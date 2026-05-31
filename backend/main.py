"""backend/main.py — Phase 3: adds /metrics Prometheus endpoint."""
from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.metrics import router as metrics_router
from .core.metrics_exporter import metrics_router as prom_router

app = FastAPI(
    title="Predictence Backend",
    description=(
        "Predictive maintenance API — rule fallback, ML anomaly detection, "
        "Prophet forecasting, feedback-loop contamination tuning, and "
        "Prometheus self-monitoring."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /metrics/… endpoints (ingest, simulate, status, alerts, predict, …)
app.include_router(metrics_router, prefix="/metrics", tags=["metrics"])

# /metrics  (bare, no prefix) — Prometheus scrape endpoint
app.include_router(prom_router, tags=["observability"])


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/", tags=["health"])
def root() -> dict:
    return {"service": "predictence_backend", "version": "3.0.0", "status": "ok"}