"""
Phase 2 — metrics API router.

Changes from Phase 1:
- /metrics/simulate still works (uses generate() from prometheus_scraper)
- /metrics/scrape  NEW — pulls from live Prometheus, falls back to sim
- /metrics/ml/status  NEW — exposes Isolation Forest training state
- All other endpoints unchanged
"""
from datetime import datetime

from fastapi import APIRouter, Query

from ..core import action_layer
from ..core import predictive_monitor
from ..core import rules_engine
from ..core import state_store
from ..core.prometheus_scraper import generate, scrape
from ..models.schemas import MetricPayload, SystemStatus

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
#  Shared ingest logic
# ─────────────────────────────────────────────────────────────────────────────

def _ingest(payload: MetricPayload) -> dict:
    state_store.push_metrics(payload)
    predictive_monitor.persist_metric_row(payload)
    alerts = rules_engine.evaluate(payload)
    state_store.push_alerts(alerts)

    actions_taken = []
    if alerts:
        for action in rules_engine.recommend_action(alerts):
            result = action_layer.execute(action, triggered_by=str([a.metric for a in alerts]))
            actions_taken.append(result.model_dump())

    return {
        "received": payload.model_dump(),
        "alerts_triggered": len(alerts),
        "alerts": [a.model_dump() for a in alerts],
        "actions": actions_taken,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoints (Phase-1 contract preserved)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/ingest", summary="Receive real or Prometheus metrics")
def ingest_metrics(payload: MetricPayload):
    return _ingest(payload)


@router.get(
    "/scrape",
    summary="[Phase 2] Pull metrics from live Prometheus; fallback to simulator",
)
def scrape_metrics():
    """
    Tries to pull all four metrics from the configured Prometheus instance.
    Falls back to a simulated 'normal' sample when Prometheus is unreachable.
    """
    payload = scrape()
    if payload is None:
        payload = generate("normal")
    return _ingest(payload)


@router.get("/simulate", summary="Generate and ingest a simulated metric sample")
def simulate_metric(
    scenario: str = Query(
        "random",
        enum=["normal", "cpu_spike", "latency", "cascade", "random"],
    )
):
    payload = generate(scenario)
    return _ingest(payload)


@router.get("/history", summary="Get recent metric history")
def get_history(limit: int = Query(60, le=200)):
    history = state_store.get_metrics_history(limit)
    return [m.model_dump() for m in history]


@router.get("/status", response_model=SystemStatus)
def get_status():
    alerts = state_store.get_alerts(limit=20, unresolved_only=True)
    return SystemStatus(
        status=rules_engine.compute_status(alerts),
        alerts=alerts,
        latest_metrics=state_store.get_latest_metrics(),
        timestamp=datetime.utcnow(),
    )


@router.get(
    "/ml/status",
    summary="[Phase 2] Isolation Forest model training status",
)
def ml_status():
    from ..ml.anomaly_detector import detector

    return {
        "trained": detector.is_trained,
        "training_samples": detector.training_samples,
        "min_samples_needed": 50,
        "mode": "isolation_forest" if detector.is_trained else "rule_fallback",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Predictive / Prophet endpoints (unchanged from Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/predict/cpu", summary="Predict CPU usage for the next 24 hours")
def predict_cpu(
    threshold: float = Query(90.0, ge=0, le=100),
    trigger_webhook: bool = Query(True),
    force_webhook: bool = Query(False),
):
    forecast = predictive_monitor.predict_cpu_next_24h(threshold=threshold)
    webhook = None
    should_trigger = bool(forecast.get("threshold_exceeded")) or force_webhook
    webhook_attempted = should_trigger and trigger_webhook
    if trigger_webhook and should_trigger:
        webhook_payload = {
            **forecast,
            "trigger_reason": (
                "forced_test"
                if force_webhook and not forecast.get("threshold_exceeded")
                else "threshold_breach"
            ),
            "forced": force_webhook,
        }
        webhook = predictive_monitor.trigger_n8n_threshold_webhook(webhook_payload)
    return {
        "forecast": forecast,
        "webhook": webhook,
        "should_trigger_webhook": should_trigger,
        "webhook_attempted": webhook_attempted,
        "trigger_webhook_received": trigger_webhook,
        "webhook_url": predictive_monitor.N8N_PREDICTIVE_WEBHOOK_URL,
    }


@router.post(
    "/predict/cpu/test-webhook",
    summary="Force-send a test predictive webhook to n8n",
)
def test_predictive_webhook():
    test_payload = {
        "trained": True,
        "history_points": 0,
        "threshold": 90.0,
        "predictions": [],
        "threshold_exceeded": False,
        "first_breach": None,
        "trigger_reason": "manual_test_endpoint",
        "forced": True,
    }
    webhook = predictive_monitor.trigger_n8n_threshold_webhook(test_payload)
    return {
        "webhook": webhook,
        "webhook_attempted": True,
        "webhook_url": predictive_monitor.N8N_PREDICTIVE_WEBHOOK_URL,
    }


@router.get(
    "/debug/prophet-dataset",
    summary="Preview Prophet-compatible dataset",
)
def debug_prophet_dataset(limit: int = Query(50, ge=1, le=500)):
    df = predictive_monitor.load_prophet_dataset(metric="cpu_percent")
    rows = [
        {"ds": row.ds.isoformat(), "y": float(row.y)}
        for row in df.tail(limit).itertuples(index=False)
    ]
    return {
        "csv_path": str(predictive_monitor.METRICS_CSV_PATH),
        "rows_total": len(df),
        "rows_preview": rows,
    }
