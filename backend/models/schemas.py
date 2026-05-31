"""Shared Pydantic schemas for the Predictence backend API."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


Severity = Literal["OK", "WARNING", "CRITICAL"]
ActionName = Literal["scale_up", "clear_cache", "restart_service", "alert_team"]


class MetricPayload(BaseModel):
    """One snapshot of service-health metrics."""

    cpu_percent: float = Field(..., ge=0, le=100)
    ram_percent: float = Field(..., ge=0, le=100)
    latency_ms: float = Field(..., ge=0)
    error_rate: float = Field(..., ge=0, le=100)
    source: str | None = "manual"
    timestamp: datetime | None = None


class Alert(BaseModel):
    """A threshold or model-generated alert."""

    id: str
    severity: Literal["WARNING", "CRITICAL"]
    metric: str
    value: float
    threshold: float
    message: str
    timestamp: datetime
    resolved: bool = False


class ActionResult(BaseModel):
    """Result returned after the backend executes a recommended action."""

    action: ActionName | str
    status: Literal["simulated", "delegated", "failed"] = "simulated"
    message: str
    triggered_by: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SystemStatus(BaseModel):
    """Current system health summary consumed by the dashboard."""

    status: Severity
    alerts: list[Alert]
    latest_metrics: MetricPayload | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
