"""
Phase 2 — Persistent state store backed by TimescaleDB + Redis.

Replaces the in-memory deque store from Phase 1.
Falls back gracefully to in-memory storage when the backends are
unreachable so the service stays runnable in development without Docker.

Environment variables
---------------------
REDIS_URL          Redis connection string. Default: redis://localhost:6379/0
DATABASE_URL       PostgreSQL / TimescaleDB DSN.
                   Default: postgresql://postgres:postgres@localhost:5432/predictence

Schema (auto-created on first use)
-----------------------------------
Table: metrics
    id          BIGSERIAL PRIMARY KEY
    ts          TIMESTAMPTZ NOT NULL   ← TimescaleDB hypertable partition key
    cpu_percent DOUBLE PRECISION
    ram_percent DOUBLE PRECISION
    latency_ms  DOUBLE PRECISION
    error_rate  DOUBLE PRECISION
    source      TEXT

Table: alerts
    id          TEXT PRIMARY KEY
    severity    TEXT
    metric      TEXT
    value       DOUBLE PRECISION
    threshold   DOUBLE PRECISION
    message     TEXT
    ts          TIMESTAMPTZ
    resolved    BOOLEAN DEFAULT FALSE
"""
from __future__ import annotations

import json
import logging
import os
from collections import deque
from datetime import datetime
from typing import Optional

from ..models.schemas import Alert, MetricPayload

logger = logging.getLogger("state_store")

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/predictence",
)

# In-memory fallbacks (Phase-1 behaviour) ─────────────────────────────────────
_MAX_METRICS = 200
_MAX_ALERTS = 500
_mem_metrics: deque[MetricPayload] = deque(maxlen=_MAX_METRICS)
_mem_alerts: deque[Alert] = deque(maxlen=_MAX_ALERTS)
_mem_latest: Optional[MetricPayload] = None

# Connection singletons ────────────────────────────────────────────────────────
_redis_client = None
_pg_pool = None   # asyncpg pool; None in sync context (used via psycopg2 fallback)
_pg_conn = None   # psycopg2 synchronous connection

# DDL executed once per process ────────────────────────────────────────────────
_schema_ready = False


# ─────────────────────────────────────────────────────────────────────────────
#  Backend initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis  # type: ignore
        client = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        client.ping()
        _redis_client = client
        logger.info("[state_store] Redis connected at %s", REDIS_URL)
    except Exception as exc:
        logger.warning("[state_store] Redis unavailable (%s) — using in-memory cache", exc)
        _redis_client = None
    return _redis_client


def _get_pg():
    global _pg_conn, _schema_ready
    if _pg_conn is not None:
        return _pg_conn
    try:
        import psycopg2  # type: ignore
        import psycopg2.extras  # type: ignore

        conn = psycopg2.connect(DATABASE_URL, connect_timeout=3)
        conn.autocommit = True
        _pg_conn = conn
        logger.info("[state_store] TimescaleDB connected at %s", DATABASE_URL)
        _ensure_schema(conn)
    except Exception as exc:
        logger.warning("[state_store] TimescaleDB unavailable (%s) — using in-memory store", exc)
        _pg_conn = None
    return _pg_conn


def _ensure_schema(conn) -> None:
    global _schema_ready
    if _schema_ready:
        return
    ddl = """
    CREATE TABLE IF NOT EXISTS metrics (
        id          BIGSERIAL,
        ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        cpu_percent DOUBLE PRECISION,
        ram_percent DOUBLE PRECISION,
        latency_ms  DOUBLE PRECISION,
        error_rate  DOUBLE PRECISION,
        source      TEXT
    );

    -- Make it a TimescaleDB hypertable (no-op if already done or extension absent)
    DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
            PERFORM create_hypertable(
                'metrics', 'ts',
                if_not_exists => TRUE,
                migrate_data   => TRUE
            );
        END IF;
    END $$;

    CREATE TABLE IF NOT EXISTS alerts (
        id        TEXT PRIMARY KEY,
        severity  TEXT,
        metric    TEXT,
        value     DOUBLE PRECISION,
        threshold DOUBLE PRECISION,
        message   TEXT,
        ts        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        resolved  BOOLEAN DEFAULT FALSE
    );

    CREATE INDEX IF NOT EXISTS idx_metrics_ts   ON metrics (ts DESC);
    CREATE INDEX IF NOT EXISTS idx_alerts_ts    ON alerts  (ts DESC);
    CREATE INDEX IF NOT EXISTS idx_alerts_unres ON alerts  (resolved) WHERE resolved = FALSE;
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    _schema_ready = True
    logger.info("[state_store] Schema ensured")


# ─────────────────────────────────────────────────────────────────────────────
#  Public API (identical contract to Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

def push_metrics(m: MetricPayload) -> None:
    """Persist a MetricPayload to TimescaleDB + Redis latest cache."""
    global _mem_latest
    # 1. In-memory ring (always — used by /metrics/history when PG is down)
    _mem_metrics.append(m)
    _mem_latest = m

    # 2. TimescaleDB
    conn = _get_pg()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO metrics (ts, cpu_percent, ram_percent, latency_ms, error_rate, source)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        m.timestamp or datetime.utcnow(),
                        m.cpu_percent,
                        m.ram_percent,
                        m.latency_ms,
                        m.error_rate,
                        m.source or "manual",
                    ),
                )
        except Exception as exc:
            logger.error("[state_store] metrics insert failed: %s", exc)

    # 3. Redis — cache latest snapshot (used for ultra-fast status polling)
    r = _get_redis()
    if r:
        try:
            r.set(
                "latest_metrics",
                json.dumps(m.model_dump(mode="json")),
                ex=300,  # expire after 5 min
            )
        except Exception as exc:
            logger.error("[state_store] Redis set failed: %s", exc)


def push_alerts(alerts: list[Alert]) -> None:
    """Persist alerts to TimescaleDB + Redis."""
    _mem_alerts.extend(alerts)

    conn = _get_pg()
    if conn and alerts:
        try:
            with conn.cursor() as cur:
                for a in alerts:
                    cur.execute(
                        """
                        INSERT INTO alerts (id, severity, metric, value, threshold, message, ts, resolved)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (
                            a.id,
                            a.severity,
                            a.metric,
                            a.value,
                            a.threshold,
                            a.message,
                            a.timestamp,
                            a.resolved,
                        ),
                    )
        except Exception as exc:
            logger.error("[state_store] alerts insert failed: %s", exc)

    # Invalidate Redis alert cache
    r = _get_redis()
    if r:
        try:
            r.delete("unresolved_alerts")
        except Exception:
            pass


def get_latest_metrics() -> Optional[MetricPayload]:
    """Return the most recent MetricPayload (Redis → memory fallback)."""
    r = _get_redis()
    if r:
        try:
            raw = r.get("latest_metrics")
            if raw:
                return MetricPayload.model_validate_json(raw)
        except Exception as exc:
            logger.debug("[state_store] Redis get latest failed: %s", exc)
    return _mem_latest


def get_metrics_history(limit: int = 60) -> list[MetricPayload]:
    """Return the last `limit` metric rows (TimescaleDB → memory fallback)."""
    conn = _get_pg()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT ts, cpu_percent, ram_percent, latency_ms, error_rate, source
                    FROM metrics
                    ORDER BY ts DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
            return [
                MetricPayload(
                    cpu_percent=r[1],
                    ram_percent=r[2],
                    latency_ms=r[3],
                    error_rate=r[4],
                    source=r[5],
                    timestamp=r[0],
                )
                for r in reversed(rows)
            ]
        except Exception as exc:
            logger.error("[state_store] metrics history query failed: %s", exc)

    return list(_mem_metrics)[-limit:]


def get_alerts(limit: int = 50, unresolved_only: bool = False) -> list[Alert]:
    """Return recent alerts (TimescaleDB → memory fallback)."""
    conn = _get_pg()
    if conn:
        try:
            where = "WHERE resolved = FALSE" if unresolved_only else ""
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, severity, metric, value, threshold, message, ts, resolved
                    FROM alerts
                    {where}
                    ORDER BY ts DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
            return [
                Alert(
                    id=r[0],
                    severity=r[1],
                    metric=r[2],
                    value=r[3],
                    threshold=r[4],
                    message=r[5],
                    timestamp=r[6],
                    resolved=r[7],
                )
                for r in reversed(rows)
            ]
        except Exception as exc:
            logger.error("[state_store] alerts query failed: %s", exc)

    alerts = list(_mem_alerts)
    if unresolved_only:
        alerts = [a for a in alerts if not a.resolved]
    return alerts[-limit:]


def resolve_alert(alert_id: str) -> bool:
    """Mark an alert as resolved."""
    # Update in TimescaleDB
    conn = _get_pg()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE alerts SET resolved = TRUE WHERE id = %s",
                    (alert_id,),
                )
                updated = cur.rowcount > 0
            if updated:
                # Invalidate Redis cache
                r = _get_redis()
                if r:
                    try:
                        r.delete("unresolved_alerts")
                    except Exception:
                        pass
                return True
        except Exception as exc:
            logger.error("[state_store] resolve_alert failed: %s", exc)

    # Fallback: update in-memory
    for alert in _mem_alerts:
        if alert.id == alert_id:
            alert.resolved = True
            return True
    return False
