"""
backend/core/metrics_exporter.py
==================================
Exposes a ``/metrics`` endpoint in Prometheus text format so the
FastAPI backend can be scraped by Prometheus for self-monitoring.

Gauges exported
---------------
predictence_alerts_total{severity}        — unresolved alert count by severity
predictence_ml_trained                    — 1 if IF model is trained, 0 otherwise
predictence_ml_training_samples           — current training sample count
predictence_ml_contamination              — current contamination parameter
predictence_label_store_total             — total operator-labelled alerts
predictence_label_store_true_positives    — TP labels
predictence_label_store_false_positives   — FP labels
predictence_label_contamination_adaptive  — 1 if adaptive mode active, 0 otherwise
predictence_latest_cpu_percent            — most recent CPU reading
predictence_latest_ram_percent            — most recent RAM reading
predictence_latest_latency_ms             — most recent latency p95
predictence_latest_error_rate             — most recent error rate

All metrics carry a ``job="predictence_backend"`` label so Prometheus
can distinguish them from Node Exporter / app metrics even without
relabelling rules.

Usage (added to main.py)
------------------------
    from .core.metrics_exporter import metrics_router
    app.include_router(metrics_router, tags=["observability"])
"""
from __future__ import annotations

import logging
import time
from typing import Callable

from fastapi import APIRouter, Response

log = logging.getLogger("metrics_exporter")

metrics_router = APIRouter()

_JOB = "predictence_backend"


# ---------------------------------------------------------------------------
# Tiny text-format builder (no prometheus_client dependency needed)
# ---------------------------------------------------------------------------

class _TextBuilder:
    """Builds a Prometheus text exposition string incrementally."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def gauge(
        self,
        name: str,
        value: float,
        help_text: str,
        labels: dict[str, str] | None = None,
    ) -> None:
        extra = dict(labels or {})
        extra["job"] = _JOB
        label_str = ",".join(f'{k}="{v}"' for k, v in extra.items())
        self._lines.append(f"# HELP {name} {help_text}")
        self._lines.append(f"# TYPE {name} gauge")
        self._lines.append(f"{name}{{{label_str}}} {_fmt(value)}")

    def build(self) -> str:
        return "\n".join(self._lines) + "\n"


def _fmt(v: float) -> str:
    """Format a float for Prometheus — integers without decimal point."""
    if v == float("inf"):
        return "+Inf"
    if v == float("-inf"):
        return "-Inf"
    if v != v:          # NaN
        return "NaN"
    if v == int(v):
        return str(int(v))
    return f"{v:.6g}"


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def _collect() -> str:
    b = _TextBuilder()
    ts_ms = int(time.time() * 1000)  # noqa: F841 — available for future use

    # ── ML detector ───────────────────────────────────────────────────────────
    try:
        from ..ml.anomaly_detector import detector
        b.gauge("predictence_ml_trained",          float(detector.is_trained),
                "1 if Isolation Forest model is trained")
        b.gauge("predictence_ml_training_samples", float(detector.training_samples),
                "Number of samples used for IF training")
        b.gauge("predictence_ml_contamination",    detector.current_contamination,
                "Current contamination parameter of the Isolation Forest")
    except Exception as exc:
        log.warning("[metrics_exporter] detector metrics unavailable: %s", exc)

    # ── Label store (feedback loop) ──────────────────────────────────────────
    try:
        from ..ml.label_store import store as label_store
        s = label_store.summary()
        b.gauge("predictence_label_store_total",
                float(s["total_labels"]),
                "Total operator-labelled alerts recorded")
        b.gauge("predictence_label_store_true_positives",
                float(s["true_positives"]),
                "Alerts labelled as true positives")
        b.gauge("predictence_label_store_false_positives",
                float(s["false_positives"]),
                "Alerts labelled as false positives")
        b.gauge("predictence_label_contamination_adaptive",
                1.0 if s["adaptive"] else 0.0,
                "1 if contamination is being set adaptively from operator feedback")
        b.gauge("predictence_label_contamination_estimate",
                s["contamination_estimate"],
                "Contamination estimate derived from operator feedback labels")
    except Exception as exc:
        log.warning("[metrics_exporter] label_store metrics unavailable: %s", exc)

    # ── Latest ingested metrics ───────────────────────────────────────────────
    try:
        from ..core.state_store import get_latest_metrics
        latest = get_latest_metrics()
        if latest:
            b.gauge("predictence_latest_cpu_percent", latest.cpu_percent,
                    "Most recent CPU percent reading")
            b.gauge("predictence_latest_ram_percent", latest.ram_percent,
                    "Most recent RAM percent reading")
            b.gauge("predictence_latest_latency_ms",  latest.latency_ms,
                    "Most recent latency p95 reading in milliseconds")
            b.gauge("predictence_latest_error_rate",  latest.error_rate,
                    "Most recent HTTP error rate percent")
    except Exception as exc:
        log.warning("[metrics_exporter] latest metrics unavailable: %s", exc)

    # ── Active alerts ─────────────────────────────────────────────────────────
    try:
        from ..core.state_store import get_alerts
        unresolved = get_alerts(limit=500, unresolved_only=True)
        warning_count  = sum(1 for a in unresolved if a.severity == "WARNING")
        critical_count = sum(1 for a in unresolved if a.severity == "CRITICAL")
        b.gauge("predictence_alerts_total", float(warning_count),
                "Unresolved alert count by severity",
                labels={"severity": "WARNING"})
        b.gauge("predictence_alerts_total", float(critical_count),
                "Unresolved alert count by severity",
                labels={"severity": "CRITICAL"})
    except Exception as exc:
        log.warning("[metrics_exporter] alert metrics unavailable: %s", exc)

    return b.build()


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@metrics_router.get(
    "/metrics",
    response_class=Response,
    summary="Prometheus-format metrics for self-monitoring",
    include_in_schema=True,
)
def prometheus_metrics() -> Response:
    """
    Scrape endpoint compatible with Prometheus ``/metrics`` convention.
    Add this backend as a target in ``prometheus.yml`` under job ``fastapi``
    (already present in ``docker/prometheus.yml``).
    """
    try:
        body = _collect()
        return Response(
            content=body,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )
    except Exception as exc:
        log.error("[metrics_exporter] collection failed: %s", exc)
        return Response(
            content=f"# ERROR collecting metrics: {exc}\n",
            media_type="text/plain; version=0.0.4; charset=utf-8",
            status_code=500,
        )
