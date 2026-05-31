"""
Phase 2 — Real Prometheus scraper.

Replaces the synthetic simulator (prometheus_sim.py) with a live
prometheus_api_client pull.  The original generate() function is kept
as a fallback so the /metrics/simulate endpoint continues to work in
development / demo environments where Prometheus is not reachable.

Environment variables
---------------------
PROMETHEUS_URL      Base URL of your Prometheus instance.
                    Default: http://localhost:9090
PROMETHEUS_TIMEOUT  Per-request timeout in seconds. Default: 10
PROMETHEUS_STEP     Step/resolution for range queries.  Default: 15s

PromQL queries used (configurable via PROMETHEUS_QUERIES_* env vars):
    cpu_percent   : 100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)
    ram_percent   : 100 * (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)
    latency_ms    : histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[1m])) by (le)) * 1000
    error_rate    : 100 * sum(rate(http_requests_total{status=~"5.."}[1m])) / sum(rate(http_requests_total[1m]))
"""
from __future__ import annotations

import logging
import math
import os
import random
from datetime import datetime
from typing import Optional

from ..models.schemas import MetricPayload

logger = logging.getLogger("prometheus_scraper")

# --------------------------------------------------------------------------- #
#  Configuration                                                               #
# --------------------------------------------------------------------------- #
PROMETHEUS_URL: str = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
PROMETHEUS_TIMEOUT: float = float(os.getenv("PROMETHEUS_TIMEOUT", "10"))

# PromQL expressions — override via environment if your metric names differ
_QUERIES: dict[str, str] = {
    "cpu_percent": os.getenv(
        "PROMETHEUS_QUERY_CPU",
        '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)',
    ),
    "ram_percent": os.getenv(
        "PROMETHEUS_QUERY_RAM",
        "100 * (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)",
    ),
    "latency_ms": os.getenv(
        "PROMETHEUS_QUERY_LATENCY",
        'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[1m])) by (le)) * 1000',
    ),
    "error_rate": os.getenv(
        "PROMETHEUS_QUERY_ERROR_RATE",
        '100 * sum(rate(http_requests_total{status=~"5.."}[1m])) / sum(rate(http_requests_total[1m]))',
    ),
}


# --------------------------------------------------------------------------- #
#  Prometheus client helper                                                    #
# --------------------------------------------------------------------------- #

def _query_instant(metric_name: str) -> Optional[float]:
    """
    Execute a single instant PromQL query.
    Returns the scalar float result, or None on any error.
    """
    try:
        from prometheus_api_client import PrometheusConnect  # type: ignore

        prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)
        result = prom.custom_query(query=_QUERIES[metric_name])
        if result:
            return float(result[0]["value"][1])
        logger.warning("[prometheus] empty result for %s", metric_name)
        return None
    except ImportError:
        logger.error(
            "[prometheus] prometheus_api_client not installed — "
            "run: pip install prometheus-api-client"
        )
        return None
    except Exception as exc:
        logger.error("[prometheus] query failed for %s: %s", metric_name, exc)
        return None


def scrape() -> Optional[MetricPayload]:
    """
    Scrape all four metrics from a live Prometheus instance.

    Returns a MetricPayload on success, None if any metric is unavailable.
    Caller should fall back to generate() when this returns None.
    """
    values: dict[str, Optional[float]] = {k: _query_instant(k) for k in _QUERIES}

    if any(v is None for v in values.values()):
        missing = [k for k, v in values.items() if v is None]
        logger.warning("[prometheus] metrics unavailable: %s — using simulator", missing)
        return None

    # Clamp to valid schema ranges
    return MetricPayload(
        cpu_percent=max(0.0, min(100.0, values["cpu_percent"])),   # type: ignore[arg-type]
        ram_percent=max(0.0, min(100.0, values["ram_percent"])),   # type: ignore[arg-type]
        latency_ms=max(0.0, values["latency_ms"]),                 # type: ignore[arg-type]
        error_rate=max(0.0, min(100.0, values["error_rate"] or 0.0)),
        source="prometheus",
        timestamp=datetime.utcnow(),
    )


# --------------------------------------------------------------------------- #
#  Simulator fallback (kept from Phase 1 — used by /metrics/simulate)         #
# --------------------------------------------------------------------------- #
_tick = 0


def generate(scenario: str = "normal") -> MetricPayload:
    """
    Synthetic metric generator — unchanged from Phase 1.
    Used by /metrics/simulate and as a fallback when Prometheus is unreachable.
    """
    global _tick
    _tick += 1
    if scenario == "random":
        scenario = random.choice(["normal", "normal", "cpu_spike", "latency", "cascade"])

    t = _tick * 0.1

    if scenario == "normal":
        return MetricPayload(
            cpu_percent=40 + 15 * math.sin(t) + random.gauss(0, 3),
            ram_percent=55 + 5 * math.sin(t * 0.5) + random.gauss(0, 2),
            latency_ms=120 + 40 * abs(math.sin(t * 0.7)) + random.gauss(0, 15),
            error_rate=0.5 + random.uniform(0, 1.5),
            source="prometheus_sim",
            timestamp=datetime.utcnow(),
        )
    if scenario == "cpu_spike":
        return MetricPayload(
            cpu_percent=min(98, 82 + random.gauss(0, 5)),
            ram_percent=70 + random.gauss(0, 3),
            latency_ms=350 + random.gauss(0, 50),
            error_rate=2 + random.uniform(0, 2),
            source="prometheus_sim",
            timestamp=datetime.utcnow(),
        )
    if scenario == "latency":
        return MetricPayload(
            cpu_percent=55 + random.gauss(0, 5),
            ram_percent=60 + random.gauss(0, 3),
            latency_ms=600 + random.gauss(0, 100),
            error_rate=6 + random.uniform(0, 4),
            source="prometheus_sim",
            timestamp=datetime.utcnow(),
        )
    if scenario == "cascade":
        return MetricPayload(
            cpu_percent=min(99, 91 + random.gauss(0, 3)),
            ram_percent=min(99, 93 + random.gauss(0, 2)),
            latency_ms=1100 + random.gauss(0, 150),
            error_rate=min(30, 12 + random.uniform(0, 5)),
            source="prometheus_sim",
            timestamp=datetime.utcnow(),
        )
    return generate("normal")
