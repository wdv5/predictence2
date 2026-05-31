"""CPU forecasting helpers for the Predictence backend.

Metrics are appended to a local CSV so Prophet can train from the same data that
flows through `/metrics/ingest`, `/metrics/simulate`, and `/metrics/scrape`.
When there is not enough history to train Prophet, the forecast endpoint returns
a graceful untrained response instead of blocking backend startup.
"""
from __future__ import annotations

import csv
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from prophet import Prophet

from ..models.schemas import MetricPayload

logger = logging.getLogger("predictive_monitor")

DATA_DIR = Path(os.getenv("PREDICTENCE_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
METRICS_CSV_PATH = Path(os.getenv("METRICS_CSV_PATH", DATA_DIR / "metrics.csv"))
N8N_PREDICTIVE_WEBHOOK_URL = os.getenv(
    "N8N_PREDICTIVE_WEBHOOK_URL",
    "http://localhost:5678/webhook/cpu-threshold-forecast",
)
N8N_ENABLED = os.getenv("N8N_ENABLED", "false").lower() == "true"
N8N_TIMEOUT = float(os.getenv("N8N_TIMEOUT", "5"))
_MIN_FORECAST_POINTS = int(os.getenv("PROPHET_MIN_POINTS", "12"))

_FIELDNAMES = [
    "timestamp",
    "cpu_percent",
    "ram_percent",
    "latency_ms",
    "error_rate",
    "source",
]


def persist_metric_row(payload: MetricPayload) -> None:
    """Append a MetricPayload to the local forecasting CSV."""
    METRICS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = METRICS_CSV_PATH.exists() and METRICS_CSV_PATH.stat().st_size > 0
    timestamp = payload.timestamp or datetime.utcnow()
    with METRICS_CSV_PATH.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": timestamp.isoformat(),
                "cpu_percent": payload.cpu_percent,
                "ram_percent": payload.ram_percent,
                "latency_ms": payload.latency_ms,
                "error_rate": payload.error_rate,
                "source": payload.source or "manual",
            }
        )


def load_prophet_dataset(metric: str = "cpu_percent"):
    """Load local metric history in Prophet's `ds`/`y` format."""
    try:
        import pandas as pd
    except Exception as exc:
        logger.error("[predictive_monitor] pandas unavailable: %s", exc)
        raise RuntimeError(f"pandas is required for forecasting: {exc}") from exc
def load_prophet_dataset(metric: str = "cpu_percent") -> pd.DataFrame:
    """Load local metric history in Prophet's `ds`/`y` format."""
    if metric not in _FIELDNAMES:
        raise ValueError(f"Unsupported metric for forecasting: {metric}")
    if not METRICS_CSV_PATH.exists():
        return pd.DataFrame(columns=["ds", "y"])

    df = pd.read_csv(METRICS_CSV_PATH)
    if df.empty or "timestamp" not in df or metric not in df:
        return pd.DataFrame(columns=["ds", "y"])

    out = pd.DataFrame(
        {
            "ds": pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_localize(None),
            "y": pd.to_numeric(df[metric], errors="coerce"),
        }
    )
    out = out.dropna().sort_values("ds")
    return out.drop_duplicates(subset=["ds"], keep="last")

def _untrained_forecast(df: pd.DataFrame, threshold: float) -> dict[str, Any]:
    return {
        "trained": False,
        "history_points": int(len(df)),
        "threshold": threshold,
        "predictions": [],
        "threshold_exceeded": False,
        "first_breach": None,
        "message": f"Need at least {_MIN_FORECAST_POINTS} metric rows to train Prophet.",
    }


def predict_cpu_next_24h(threshold: float = 90.0) -> dict[str, Any]:
    """Forecast hourly CPU usage for the next 24 hours."""
    try:
        df = load_prophet_dataset("cpu_percent")
    except RuntimeError as exc:
        return {
            "trained": False,
            "history_points": 0,
            "threshold": threshold,
            "predictions": [],
            "threshold_exceeded": False,
            "first_breach": None,
            "message": str(exc),
        }

    df = load_prophet_dataset("cpu_percent")
    if len(df) < _MIN_FORECAST_POINTS:
        return _untrained_forecast(df, threshold)

    try:
        from prophet import Prophet

        model = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False)
        model.fit(df)
        future = model.make_future_dataframe(periods=24, freq="h", include_history=False)
        forecast = model.predict(future)
    except Exception as exc:
        logger.error("[predictive_monitor] Prophet forecast failed: %s", exc)
        result = _untrained_forecast(df, threshold)
        result["message"] = f"Forecast failed: {exc}"
        return result

    predictions = []
    first_breach = None
    for row in forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].itertuples(index=False):
        yhat = max(0.0, min(100.0, float(row.yhat)))
        item = {
            "timestamp": row.ds.isoformat(),
            "cpu_percent": yhat,
            "lower": max(0.0, float(row.yhat_lower)),
            "upper": min(100.0, float(row.yhat_upper)),
        }
        predictions.append(item)
        if first_breach is None and yhat >= threshold:
            first_breach = item

    return {
        "trained": True,
        "history_points": int(len(df)),
        "threshold": threshold,
        "predictions": predictions,
        "threshold_exceeded": first_breach is not None,
        "first_breach": first_breach,
        "generated_at": datetime.utcnow().isoformat(),
        "window_end": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
    }


def trigger_n8n_threshold_webhook(payload: dict[str, Any]) -> dict[str, Any]:
    """Send a predictive alert webhook to n8n when enabled."""
    if not N8N_ENABLED:
        return {
            "enabled": False,
            "sent": False,
            "status": "simulated",
            "message": "N8N_ENABLED is false; webhook not sent.",
        }

    try:
        response = httpx.post(N8N_PREDICTIVE_WEBHOOK_URL, json=payload, timeout=N8N_TIMEOUT)
        response.raise_for_status()
        return {
            "enabled": True,
            "sent": True,
            "status": "sent",
            "status_code": response.status_code,
        }
    except Exception as exc:
        logger.error("[predictive_monitor] webhook failed: %s", exc)
        return {
            "enabled": True,
            "sent": False,
            "status": "failed",
            "message": str(exc),
        }
