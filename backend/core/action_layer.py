"""Action execution layer for alert remediation recommendations.

The project currently runs in a safe simulation mode by default. When n8n is
enabled, this module delegates the action request to an n8n webhook while still
returning a stable ActionResult shape for the API and dashboard.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime

import httpx

from ..models.schemas import ActionResult

logger = logging.getLogger("action_layer")

N8N_ENABLED = os.getenv("N8N_ENABLED", "false").lower() == "true"
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/agent-alert")
N8N_TIMEOUT = float(os.getenv("N8N_TIMEOUT", "5"))

_ACTION_MESSAGES = {
    "scale_up": "Scale-up recommendation recorded.",
    "clear_cache": "Cache-clear recommendation recorded.",
    "restart_service": "Service-restart recommendation recorded.",
    "alert_team": "Team-alert recommendation recorded.",
}


def execute(action: str, triggered_by: str | None = None) -> ActionResult:
    """Execute or simulate a remediation action."""
    payload = {
        "action": action,
        "triggered_by": triggered_by,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if N8N_ENABLED:
        try:
            response = httpx.post(N8N_WEBHOOK_URL, json=payload, timeout=N8N_TIMEOUT)
            response.raise_for_status()
            return ActionResult(
                action=action,
                status="delegated",
                message=f"Delegated {action} to n8n webhook.",
                triggered_by=triggered_by,
            )
        except Exception as exc:
            logger.error("[action_layer] failed to delegate %s to n8n: %s", action, exc)
            return ActionResult(
                action=action,
                status="failed",
                message=f"Failed to delegate {action}: {exc}",
                triggered_by=triggered_by,
            )

    return ActionResult(
        action=action,
        status="simulated",
        message=_ACTION_MESSAGES.get(action, f"Recommendation recorded for {action}."),
        triggered_by=triggered_by,
    )
