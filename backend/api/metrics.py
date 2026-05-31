"""
backend/api/metrics.py  (Phase 3 additions)
============================================
New endpoints added in Phase 3:

    POST /metrics/alerts/{alert_id}/resolve
        Mark an alert resolved AND record operator feedback (true_positive /
        false_positive).  The label is fed into label_store which
        automatically reweights the IF contamination estimate.

    GET  /metrics/ml/status
        Extended to include label_store summary.

All Phase 1 + Phase 2 endpoints are unchanged.
"""
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..core import action_layer, predictive_monitor, rules_engine, state_store
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
        "received":         payload.model_dump(),
        "alerts_triggered": len(alerts),
        "alerts":           [a.model_dump() for a in alerts],
        "actions":          actions_taken,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Phase-1 / Phase-2 endpoints (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/ingest", summary="Receive real or Prometheus metrics")
def ingest_metrics(payload: MetricPayload):
    return _ingest(payload)


@router.get("/scrape", summary="[Phase 2] Pull from live Prometheus; fallback to simulator")
def scrape_metrics():
    payload = scrape()
    if payload is None:
        payload = generate("normal")
    return _ingest(payload)


@router.get("/simulate", summary="Generate and ingest a simulated metric sample")
def simulate_metric(
    scenario: str = Query("random", enum=["normal", "cpu_spike", "latency", "cascade", "random"]),
):
    payload = generate(scenario)
    return _ingest(payload)


@router.get("/history", summary="Get recent metric history")
def get_history(limit: int = Query(60, le=200)):
    return [m.model_dump() for m in state_store.get_metrics_history(limit)]


@router.get("/status", response_model=SystemStatus)
def get_status():
    alerts = state_store.get_alerts(limit=20, unresolved_only=True)
    return SystemStatus(
        status=rules_engine.compute_status(alerts),
        alerts=alerts,
        latest_metrics=state_store.get_latest_metrics(),
        timestamp=datetime.utcnow(),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Phase-3: alert resolution + feedback loop
# ─────────────────────────────────────────────────────────────────────────────

class _ResolveRequest(BaseModel):
    """Body for the resolve endpoint."""
    label: str = "true_positive"   # "true_positive" | "false_positive"


@router.post(
    "/alerts/{alert_id}/resolve",
    summary="[Phase 3] Resolve an alert and record operator feedback",
)
def resolve_alert(alert_id: str, body: _ResolveRequest):
    """
    Marks an alert resolved in the state store **and** records whether it
    was a genuine anomaly (``true_positive``) or noise (``false_positive``).

    The label is fed into the label store, which continuously reweights
    the Isolation Forest's ``contamination`` parameter so the model
    self-calibrates over time.

    Example
    -------
    ```
    curl -X POST http://localhost:8000/metrics/alerts/<id>/resolve \\
         -H 'Content-Type: application/json' \\
         -d '{"label": "false_positive"}'
    ```
    """
    if body.label not in ("true_positive", "false_positive"):
        raise HTTPException(
            status_code=422,
            detail="label must be 'true_positive' or 'false_positive'",
        )

    # 1. Mark resolved in state store
    resolved = state_store.resolve_alert(alert_id)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id!r} not found")

    # 2. Record feedback → updates contamination estimate
    from ..ml.label_store import store as label_store
    new_contamination = label_store.record(alert_id, label=body.label)  # type: ignore[arg-type]

    # 3. Report back
    ls_summary = label_store.summary()
    return {
        "alert_id":            alert_id,
        "resolved":            True,
        "label_recorded":      body.label,
        "new_contamination":   new_contamination,
        "label_store":         ls_summary,
        "message": (
            f"Alert resolved as {body.label}. "
            f"Contamination estimate updated to {new_contamination:.4f}. "
            f"Model will use this on next retrain "
            f"({'adaptive' if ls_summary['adaptive'] else 'default until ' + str(ls_summary['min_labels_to_adapt']) + ' labels'})."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Phase-2 / Phase-3 ML status (extended)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/ml/status", summary="[Phase 3] Isolation Forest + feedback loop status")
def ml_status():
    from ..ml.anomaly_detector import detector
    from ..ml.label_store import store as label_store

    return {
        # IF model
        "trained":              detector.is_trained,
        "training_samples":     detector.training_samples,
        "min_samples_needed":   50,
        "mode":                 "isolation_forest" if detector.is_trained else "rule_fallback",
        "current_contamination": detector.current_contamination,
        # Feedback loop
        "label_store":          label_store.summary(),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Predictive / Prophet endpoints (unchanged)
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
        "forecast":                forecast,
        "webhook":                 webhook,
        "should_trigger_webhook":  should_trigger,
        "webhook_attempted":       webhook_attempted,
        "trigger_webhook_received": trigger_webhook,
        "webhook_url":             predictive_monitor.N8N_PREDICTIVE_WEBHOOK_URL,
    }


@router.post("/predict/cpu/test-webhook", summary="Force-send a test predictive webhook to n8n")
def test_predictive_webhook():
    test_payload = {
        "trained": True, "history_points": 0, "threshold": 90.0,
        "predictions": [], "threshold_exceeded": False, "first_breach": None,
        "trigger_reason": "manual_test_endpoint", "forced": True,
    }
    webhook = predictive_monitor.trigger_n8n_threshold_webhook(test_payload)
    return {
        "webhook": webhook,
        "webhook_attempted": True,
        "webhook_url": predictive_monitor.N8N_PREDICTIVE_WEBHOOK_URL,
    }


@router.get("/debug/prophet-dataset", summary="Preview Prophet-compatible dataset")
def debug_prophet_dataset(limit: int = Query(50, ge=1, le=500)):
    df = predictive_monitor.load_prophet_dataset(metric="cpu_percent")
    rows = [
        {"ds": row.ds.isoformat(), "y": float(row.y)}
        for row in df.tail(limit).itertuples(index=False)
    ]
    return {
        "csv_path":    str(predictive_monitor.METRICS_CSV_PATH),
        "rows_total":  len(df),
        "rows_preview": rows,
    }