"""
backend/ml/anomaly_detector.py  (Phase 3 — adaptive contamination)
===================================================================
Isolation Forest anomaly detector.

Phase 3 change: before every retrain the detector reads
``label_store.contamination_estimate`` so operator feedback
(resolved-alert labels) automatically tunes how aggressively the model
flags anomalies.  Everything else — public API, scoring thresholds,
rule fallback — is unchanged.
"""
from __future__ import annotations

import logging
import threading
import uuid
from collections import deque
from datetime import datetime
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..models.schemas import Alert, MetricPayload
from ..core.rules_engine import _rule_evaluate as rule_evaluate

log = logging.getLogger("anomaly_detector")

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
FEATURE_COLS = ["cpu_percent", "ram_percent", "latency_ms", "error_rate"]

MIN_SAMPLES_TO_TRAIN    = 50
RETRAIN_EVERY           = 25
DEFAULT_CONTAMINATION   = 0.05   # used before label_store has enough data
N_ESTIMATORS            = 200
RANDOM_STATE            = 42
HISTORY_MAXLEN          = 2_000

CRITICAL_SCORE_THRESHOLD = -0.25
WARNING_SCORE_THRESHOLD  = -0.10


class _AnomalyDetector:
    """Thread-safe, self-retraining Isolation Forest with adaptive contamination."""

    def __init__(self) -> None:
        self._lock   = threading.Lock()
        self._history: deque[list[float]] = deque(maxlen=HISTORY_MAXLEN)
        self._model:   Optional[IsolationForest] = None
        self._scaler:  Optional[StandardScaler]  = None
        self._samples_since_last_train = 0
        self._trained  = False
        self._last_contamination = DEFAULT_CONTAMINATION

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def ingest(self, payload: MetricPayload) -> None:
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
        with self._lock:
            trained = self._trained
            model   = self._model
            scaler  = self._scaler

        if not trained:
            return rule_evaluate(payload)

        row        = np.array([self._payload_to_row(payload)], dtype=float)
        row_scaled = scaler.transform(row)          # type: ignore[union-attr]
        raw_score  = float(model.score_samples(row_scaled)[0])  # type: ignore[union-attr]
        return self._build_alerts(payload, raw_score)

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def training_samples(self) -> int:
        return len(self._history)

    @property
    def current_contamination(self) -> float:
        return self._last_contamination

    # ------------------------------------------------------------------ #
    #  Private                                                             #
    # ------------------------------------------------------------------ #

    def _retrain_unlocked(self) -> None:
        """Must be called while self._lock is held."""
        # --- Phase 3: pull live contamination from label store -----------
        try:
            from .label_store import store as label_store
            contamination = label_store.contamination_estimate
        except Exception:
            contamination = DEFAULT_CONTAMINATION
        # -----------------------------------------------------------------

        data        = np.array(list(self._history), dtype=float)
        scaler      = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        model = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(data_scaled)

        self._scaler               = scaler
        self._model                = model
        self._trained              = True
        self._last_contamination   = contamination
        self._samples_since_last_train = 0

        log.info(
            "[IF] retrained on %d samples | contamination=%.4f (adaptive=%s)",
            len(data),
            contamination,
            contamination != DEFAULT_CONTAMINATION,
        )

    @staticmethod
    def _payload_to_row(p: MetricPayload) -> list[float]:
        return [getattr(p, col) for col in FEATURE_COLS]

    def _build_alerts(self, payload: MetricPayload, score: float) -> list[Alert]:
        ts = payload.timestamp or datetime.utcnow()

        if score > WARNING_SCORE_THRESHOLD:
            return []

        severity = "CRITICAL" if score < CRITICAL_SCORE_THRESHOLD else "WARNING"

        with self._lock:
            if self._scaler is None:
                return []
            means = self._scaler.mean_
            stds  = np.sqrt(self._scaler.var_)

        row     = np.array(self._payload_to_row(payload), dtype=float)
        z_scores = np.abs((row - means) / (stds + 1e-9))

        top_indices = np.argsort(z_scores)[::-1][:2]
        alerts: list[Alert] = []

        for idx in top_indices:
            if z_scores[idx] < 1.0:
                continue
            metric    = FEATURE_COLS[idx]
            value     = float(row[idx])
            mean_val  = float(means[idx])
            alerts.append(Alert(
                id=str(uuid.uuid4()),
                severity=severity,
                metric=metric,
                value=value,
                threshold=mean_val,
                message=(
                    f"[ML] {metric}={value:.2f} anomalous "
                    f"(IF score={score:.3f}, z={z_scores[idx]:.1f}σ)"
                ),
                timestamp=ts,
            ))

        if not alerts:
            alerts.append(Alert(
                id=str(uuid.uuid4()),
                severity=severity,
                metric="system",
                value=score,
                threshold=WARNING_SCORE_THRESHOLD,
                message=f"[ML] Multivariate anomaly detected (IF score={score:.3f})",
                timestamp=ts,
            ))

        return alerts


# Singleton
detector = _AnomalyDetector()