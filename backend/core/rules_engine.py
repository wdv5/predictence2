"""
Phase 2 rules_engine.py

evaluate() now delegates to the Isolation Forest detector.
All other helpers (recommend_action, compute_status) are unchanged
so the API contract stays identical.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import NamedTuple

from ..models.schemas import Alert, MetricPayload

# ------------------------------------------------------------------ #
#  Phase-1 rule table (kept for fallback + recommend_action mapping)  #
# ------------------------------------------------------------------ #


class Rule(NamedTuple):
    metric: str
    threshold: float
    operator: str
    severity: str
    message_template: str
    recommended_action: str


RULES: list[Rule] = [
    Rule("cpu_percent",  80,  "gt", "WARNING",  "CPU {val:.1f}% > 80%",         "scale_up"),
    Rule("cpu_percent",  90,  "gt", "CRITICAL", "CPU {val:.1f}% > 90%",         "scale_up"),
    Rule("ram_percent",  85,  "gt", "WARNING",  "RAM {val:.1f}% > 85%",         "clear_cache"),
    Rule("ram_percent",  95,  "gt", "CRITICAL", "RAM {val:.1f}% > 95%",         "restart_service"),
    Rule("latency_ms",  500,  "gt", "WARNING",  "Latency {val:.0f}ms > 500ms",  "clear_cache"),
    Rule("latency_ms", 1000,  "gt", "CRITICAL", "Latency {val:.0f}ms > 1000ms", "scale_up"),
    Rule("error_rate",    5,  "gt", "WARNING",  "Error rate {val:.1f}% > 5%",   "alert_team"),
    Rule("error_rate",   10,  "gt", "CRITICAL", "Error rate {val:.1f}% > 10%",  "alert_team"),
]

ACTION_PRIORITY = {"CRITICAL": 2, "WARNING": 1}

# ------------------------------------------------------------------ #
#  Phase-2 addition: per-metric action mapping used by recommend_action#
# ------------------------------------------------------------------ #
_METRIC_TO_ACTION: dict[str, str] = {
    "cpu_percent": "scale_up",
    "ram_percent": "clear_cache",
    "latency_ms":  "clear_cache",
    "error_rate":  "alert_team",
    "system":      "alert_team",   # aggregate anomaly
}


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #


def evaluate(metrics: MetricPayload) -> list[Alert]:
    """
    Phase 2: delegate to the Isolation Forest detector.
    Falls back to rule-based evaluation while the model warms up.
    """
    # Import here to avoid circular imports at module load time
    from ..ml.anomaly_detector import detector

    # Always feed the detector so it accumulates training data
    detector.ingest(metrics)

    return detector.score(metrics)


def recommend_action(alerts: list[Alert]) -> list[str]:
    """
    Recommend remediating actions for a set of alerts.

    Works for both Phase-1 (rule-based) and Phase-2 (ML) alerts:
    - ML alerts carry metric names in alert.metric.
    - Rule alerts match against the RULES table.
    """
    actions: set[str] = set()
    for alert in alerts:
        # Try exact rule match first (covers Phase-1 fallback alerts)
        for rule in RULES:
            if rule.metric == alert.metric and rule.severity == alert.severity:
                actions.add(rule.recommended_action)
                break
        else:
            # Phase-2 ML alert — use metric→action mapping
            action = _METRIC_TO_ACTION.get(alert.metric)
            if action:
                actions.add(action)
    return list(actions)


def compute_status(alerts: list[Alert]) -> str:
    if not alerts:
        return "OK"
    if any(a.severity == "CRITICAL" for a in alerts):
        return "CRITICAL"
    return "WARNING"


# ------------------------------------------------------------------ #
#  Rule-based evaluate kept as private helper (used by detector       #
#  fallback — imported directly by anomaly_detector.py)              #
# ------------------------------------------------------------------ #


def _rule_evaluate(metrics: MetricPayload) -> list[Alert]:
    """Original Phase-1 rule evaluator (threshold-based)."""
    triggered: dict[str, Alert] = {}
    for rule in RULES:
        value = getattr(metrics, rule.metric)
        if rule.operator == "gt" and value <= rule.threshold:
            continue
        if rule.operator == "lt" and value >= rule.threshold:
            continue
        existing = triggered.get(rule.metric)
        if existing and ACTION_PRIORITY[existing.severity] >= ACTION_PRIORITY[rule.severity]:
            continue
        alert = Alert(
            id=str(uuid.uuid4()),
            severity=rule.severity,
            metric=rule.metric,
            value=value,
            threshold=rule.threshold,
            message=rule.message_template.format(val=value),
            timestamp=metrics.timestamp or datetime.utcnow(),
        )
        triggered[rule.metric] = alert
    return list(triggered.values())
