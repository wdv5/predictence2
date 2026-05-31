"""FastAPI application entry point for the Predictence backend."""
from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.metrics import router as metrics_router

app = FastAPI(
    title="Predictence Backend",
    description="Predictive maintenance API with rule fallback, ML anomaly detection, and forecasting.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(metrics_router, prefix="/metrics", tags=["metrics"])


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Container healthcheck endpoint."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/", tags=["health"])
def root() -> dict[str, str]:
    """Root endpoint for quick manual checks."""
    return {"service": "predictence_backend", "status": "ok"}
