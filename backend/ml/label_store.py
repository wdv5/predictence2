"""
backend/ml/label_store.py
==========================
Feedback-loop label store.

Every time an operator resolves an alert via the API, that outcome is
recorded here.  The store exposes ``contamination_estimate`` — a
weighted rolling fraction of alerts that were *real* anomalies vs.
false positives — which the anomaly detector reads before each retrain
to adjust the Isolation Forest ``contamination`` hyper-parameter.

Design goals
------------
- Zero new runtime dependencies (stdlib + numpy only).
- Thread-safe; safe to call from FastAPI request handlers.
- Graceful degradation: if no feedback has been collected yet, falls
  back to the hard-coded DEFAULT_CONTAMINATION so the model still trains.
- Persistent across restarts via a small JSON file written atomically.

Feedback model
--------------
Each resolved alert carries a *label*:

    "true_positive"  — the alert was a real incident; anomaly confirmed
    "false_positive" — the alert was noise; model was over-sensitive

contamination_estimate = (
    weighted sum of true_positive labels
    ──────────────────────────────────
    weighted sum of all labels
)

Recent labels carry more weight (exponential decay, half-life = 50
labels) so the estimate tracks concept drift rather than ancient history.

Usage
-----
    from backend.ml.label_store import store

    # Record operator feedback when an alert is resolved
    store.record("alert-uuid-123", label="true_positive")

    # Anomaly detector reads this before retraining
    c = store.contamination_estimate   # float in [MIN_CONTAMINATION, MAX_CONTAMINATION]
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Literal, Optional

import numpy as np

log = logging.getLogger("label_store")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONTAMINATION: float = 0.05   # used when < MIN_LABELS feedback exists
MIN_CONTAMINATION:     float = 0.01   # floor — model can never be fully blind
MAX_CONTAMINATION:     float = 0.20   # ceiling — never flag >20 % as anomalies
MIN_LABELS_TO_ADAPT:   int   = 10     # need at least this many labels to override default
DECAY_HALF_LIFE:       float = 50.0   # labels; older labels down-weighted with this half-life
MAX_HISTORY:           int   = 500    # rolling window size

Label = Literal["true_positive", "false_positive"]

_DATA_DIR = Path(os.getenv("PREDICTENCE_DATA_DIR",
                            Path(__file__).resolve().parents[2] / "backend" / "data"))
_STORE_PATH = _DATA_DIR / "label_store.json"


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

class _LabelRecord:
    __slots__ = ("alert_id", "label", "ts", "weight")

    def __init__(self, alert_id: str, label: Label, ts: datetime, weight: float = 1.0):
        self.alert_id = alert_id
        self.label    = label
        self.ts       = ts
        self.weight   = weight   # filled in by _recompute_weights()

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "label":    self.label,
            "ts":       self.ts.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "_LabelRecord":
        return cls(
            alert_id=d["alert_id"],
            label=d["label"],
            ts=datetime.fromisoformat(d["ts"]),
        )


# ---------------------------------------------------------------------------
# Label store
# ---------------------------------------------------------------------------

class LabelStore:
    """Thread-safe feedback store with adaptive contamination estimation."""

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._history: Deque[_LabelRecord] = deque(maxlen=MAX_HISTORY)
        self._contamination: float = DEFAULT_CONTAMINATION
        self._load()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def record(self, alert_id: str, label: Label) -> float:
        """
        Record operator feedback for a resolved alert.

        Parameters
        ----------
        alert_id : str
            The ``Alert.id`` that was resolved.
        label : "true_positive" | "false_positive"
            Whether the alert represented a real anomaly.

        Returns
        -------
        float
            The updated contamination estimate after incorporating this label.
        """
        rec = _LabelRecord(alert_id=alert_id, label=label, ts=datetime.utcnow())
        with self._lock:
            # Avoid double-recording the same alert
            if any(r.alert_id == alert_id for r in self._history):
                log.debug("[label_store] alert %s already recorded — skipping", alert_id)
                return self._contamination

            self._history.append(rec)
            self._recompute_unlocked()
            self._persist_unlocked()

        log.info(
            "[label_store] recorded %s for alert %s | contamination → %.4f",
            label, alert_id, self._contamination,
        )
        return self._contamination

    @property
    def contamination_estimate(self) -> float:
        """Current contamination estimate, safe to read from any thread."""
        with self._lock:
            return self._contamination

    @property
    def label_count(self) -> int:
        with self._lock:
            return len(self._history)

    @property
    def true_positive_rate(self) -> Optional[float]:
        """Weighted TP / (TP + FP), or None if < MIN_LABELS_TO_ADAPT labels."""
        with self._lock:
            if len(self._history) < MIN_LABELS_TO_ADAPT:
                return None
            return self._contamination

    def summary(self) -> dict:
        """Return a JSON-serialisable summary for the /metrics/ml/status endpoint."""
        with self._lock:
            total = len(self._history)
            tp = sum(1 for r in self._history if r.label == "true_positive")
            fp = total - tp
            return {
                "total_labels":         total,
                "true_positives":       tp,
                "false_positives":      fp,
                "contamination_estimate": self._contamination,
                "adaptive":             total >= MIN_LABELS_TO_ADAPT,
                "min_labels_to_adapt":  MIN_LABELS_TO_ADAPT,
            }

    # ------------------------------------------------------------------ #
    #  Private                                                             #
    # ------------------------------------------------------------------ #

    def _recompute_unlocked(self) -> None:
        """Recompute contamination from history.  Must hold self._lock."""
        n = len(self._history)
        if n < MIN_LABELS_TO_ADAPT:
            self._contamination = DEFAULT_CONTAMINATION
            return

        # Exponential decay weights: most recent label has weight 1,
        # label k steps ago has weight exp(-k * ln2 / HALF_LIFE)
        indices = np.arange(n, dtype=float)          # 0 = oldest
        ages    = (n - 1) - indices                  # 0 = most recent
        weights = np.exp(-ages * math.log(2) / DECAY_HALF_LIFE)

        labels  = np.array([1.0 if r.label == "true_positive" else 0.0
                            for r in self._history])

        weighted_tp    = float(np.dot(weights, labels))
        weighted_total = float(weights.sum())

        raw = weighted_tp / weighted_total if weighted_total > 0 else DEFAULT_CONTAMINATION
        self._contamination = float(np.clip(raw, MIN_CONTAMINATION, MAX_CONTAMINATION))

    def _persist_unlocked(self) -> None:
        """Atomically write history to JSON.  Must hold self._lock."""
        try:
            _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _STORE_PATH.with_suffix(".json.tmp")
            payload = {
                "version": 1,
                "contamination": self._contamination,
                "history": [r.to_dict() for r in self._history],
            }
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(_STORE_PATH)
        except Exception as exc:
            log.warning("[label_store] persist failed: %s", exc)

    def _load(self) -> None:
        """Load persisted history on startup."""
        if not _STORE_PATH.exists():
            return
        try:
            payload = json.loads(_STORE_PATH.read_text())
            records = [_LabelRecord.from_dict(d) for d in payload.get("history", [])]
            with self._lock:
                self._history = deque(records, maxlen=MAX_HISTORY)
                self._recompute_unlocked()
            log.info(
                "[label_store] loaded %d labels from %s | contamination=%.4f",
                len(self._history), _STORE_PATH, self._contamination,
            )
        except Exception as exc:
            log.warning("[label_store] could not load %s: %s", _STORE_PATH, exc)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
store = LabelStore()
