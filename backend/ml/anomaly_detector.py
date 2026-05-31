"""
Phase 2 — Isolation Forest anomaly detector.

Replaces the hard-coded threshold rules in rules_engine.py::evaluate().
Trained incrementally on incoming metrics. Falls back to rule-based
evaluation when not enough data has been collected yet.

Public API (identical contract to Phase 1):
    detector.score(payload) -> list[Alert]
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..models.schemas import Alert, MetricPayload
from ..core.rules_engine import evaluate as rule_evaluate  # Phase-1 fallback

logger = logging.getLogger("anomaly_detector")

# --------------------------------------------------------------------------- #
#  Hyper-parameters                                                             #
# --------------------------------------------------------------------------- #
FEATURE_COLS = ["cpu_percent", "ram_percent", "latency_ms", "error_rate"]

MIN_SAMPLES_TO_TRAIN = 50       # rows needed before IF kicks in
RETRAIN_EVERY = 25              # retrain after this many new samples
CONTAMINATION = 0.05            # expected fraction of anomalies (~5 %)
N_ESTIMATORS = 200
RANDOM_STATE = 42
HISTORY_MAXLEN = 2_000          # rolling window kept in memory


# --------------------------------------------------------------------------- #
#  Severity calibration (replaces hard-coded rule thresholds)                  #
#  Score < sev_threshold  →  anomaly;  the more negative, the worse.           #
# --------------------------------------------------------------------------- #
CRITICAL_SCORE_THRESHOLD = -0.25
WARNING_SCORE_THRESHOLD = -0.10


class _AnomalyDetector:
    """Thread-safe, self-retraining Isolation Forest wrapper."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._history: deque[list[float]] = deque(maxlen=HISTORY_MAXLEN)
        self._model: Optional[IsolationForest] = None
        self._scaler: Optional[StandardScaler] = None
        self._samples_since_last_train = 0
        self._trained = False

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def ingest(self, payload: MetricPayload) -> None:
        """Add a new observation; retrain if the schedule says so."""
        row = self._payload_to_row(payload)
        with self._lock:
            self._history.append(row)
            self._samples_since_last_train += 1
            if (
                len(self._history) >= MIN_SAMPLES_TO_TRAIN
                and self._samples_since_last_train >= RETRAIN_EVERY
            ):
                self._retrain_unlocked()

    def score(self, payload: MetricPayload) -> list[Alert]:
        """
        Return a list of Alerts — same contract as rules_engine.evaluate().
        Falls back to rule-based logic when the model is not yet trained.
        """
        with self._lock:
            trained = self._trained
            model = self._model
            scaler = self._scaler

        if not trained:
            # Not enough data yet — defer to Phase-1 rules
            logger.debug("[IF] not trained yet — using rule fallback")
            return rule_evaluate(payload)

        row = np.array([self._payload_to_row(payload)], dtype=float)
        row_scaled = scaler.transform(row)  # type: ignore[union-attr]
        raw_score: float = float(model.score_samples(row_scaled)[0])  # type: ignore[union-attr]

        return self._build_alerts(payload, raw_score)

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def training_samples(self) -> int:
        return len(self._history)

    # ------------------------------------------------------------------ #
    #  Private                                                             #
    # ------------------------------------------------------------------ #

    def _retrain_unlocked(self) -> None:
        """Must be called while self._lock is held."""
        data = np.array(list(self._history), dtype=float)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        model = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(data_scaled)
        self._scaler = scaler
        self._model = model
        self._trained = True
        self._samples_since_last_train = 0
        logger.info(
            "[IF] retrained on %d samples (contamination=%.2f)",
            len(data),
            CONTAMINATION,
        )

    @staticmethod
    def _payload_to_row(p: MetricPayload) -> list[float]:
        return [getattr(p, col) for col in FEATURE_COLS]

    def _build_alerts(self, payload: MetricPayload, score: float) -> list[Alert]:
        """
        Map the IF anomaly score to severity-labelled Alerts.

        We still emit per-metric alerts (not a single "anomaly" alert)
        so that downstream consumers (action layer, dashboard) need zero changes.
        The anomaly score drives severity; per-metric deviation from the mean
        determines which metrics to surface.
        """
        import uuid

        ts = payload.timestamp or datetime.utcnow()

        if score > WARNING_SCORE_THRESHOLD:
            # Normal — no alerts
            return []

        severity = "CRITICAL" if score < CRITICAL_SCORE_THRESHOLD else "WARNING"

        # Identify the *contributing* metrics using z-score from scaler mean
        with self._lock:
            if self._scaler is None:
                return []
            means = self._scaler.mean_
            stds = np.sqrt(self._scaler.var_)

        row = np.array(self._payload_to_row(payload), dtype=float)
        z_scores = np.abs((row - means) / (stds + 1e-9))

        # Surface metrics whose z-score is in the top-2 contributors
        top_indices = np.argsort(z_scores)[::-1][:2]

        alerts: list[Alert] = []
        for idx in top_indices:
            if z_scores[idx] < 1.0:
                # Only flag metrics that are genuinely unusual
                continue
            metric = FEATURE_COLS[idx]
            value = float(row[idx])
            mean_val = float(means[idx])
            alerts.append(
                Alert(
                    id=str(uuid.uuid4()),
                    severity=severity,
                    metric=metric,
                    value=value,
                    threshold=float(mean_val),  # baseline mean acts as "threshold"
                    message=(
                        f"[ML] {metric}={value:.2f} is anomalous "
                        f"(IF score={score:.3f}, z={z_scores[idx]:.1f}σ)"
                    ),
                    timestamp=ts,
                )
            )

        # If no individual metric stood out but we still have an anomaly,
        # emit a single aggregate alert
        if not alerts:
            alerts.append(
                Alert(
                    id=str(uuid.uuid4()),
                    severity=severity,
                    metric="system",
                    value=score,
                    threshold=WARNING_SCORE_THRESHOLD,
                    message=f"[ML] Multivariate anomaly detected (IF score={score:.3f})",
                    timestamp=ts,
                )
            )

        return alerts


# Singleton — imported by rules_engine and metrics API
detector = _AnomalyDetector()
