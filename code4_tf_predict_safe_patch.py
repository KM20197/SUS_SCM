# Reconstructed helper module.
# Functional equivalent of the safe prediction behavior currently embedded
# in Code4_colab_fixed_v12_actionscale_minpatch.py.

from __future__ import annotations

"""
Safe prediction helper for the LSTM+RF forecaster used in the Code4 v12 runner.

Purpose
-------
This module isolates the prediction fallback logic already embedded in the
main runner. It does not change the experimental design. It only guards
forecast inference against runtime failures in the LSTM or Random Forest
submodels and returns a safe non-negative fallback.

Status
------
Functional reconstruction for repository organization and readability.
If the main runner does not import this module yet, adding this file alone
does not change any experiment outputs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PredictSafeLogger:
    enabled: bool = False
    max_prints: int = 3
    counts: Dict[str, int] = field(default_factory=dict)

    def log(self, tag: str, exc: BaseException) -> None:
        if not self.enabled:
            return
        n = self.counts.get(tag, 0)
        self.counts[tag] = n + 1
        if n < self.max_prints:
            print(f"[warn] {tag}: {type(exc).__name__}: {exc}")


def _safe_nonnegative_mean(values: List[float], fallback: float) -> float:
    vals = [float(v) for v in values if np.isfinite(v)]
    if vals:
        return float(max(0.0, np.mean(vals)))
    return float(max(0.0, fallback))


def safe_predict_lstm_rf(
    hist: np.ndarray,
    seq_len: int,
    scaler_X,
    scaler_y,
    lstm_model,
    rf_model,
    fallback: float,
    logger: Optional[PredictSafeLogger] = None,
) -> float:
    """
    Safe one-step prediction for the LSTM+RF ensemble.

    Rules
    -----
    1. Empty history -> fallback
    2. Short history -> mean(history)
    3. Try LSTM prediction
    4. Try RF prediction
    5. If both fail -> fallback
    6. Never return a negative value
    """
    hist = np.asarray(hist, dtype=float)

    if hist.size == 0:
        return float(max(0.0, fallback))

    if hist.size < seq_len:
        return float(max(0.0, np.mean(hist)))

    seq = hist[-seq_len:].astype(float)
    preds: List[float] = []

    try:
        x_s = scaler_X.transform(seq.reshape(1, -1)).reshape(1, seq_len, 1)
        p_lstm = scaler_y.inverse_transform(
            lstm_model.predict(x_s, verbose=0)
        )[0][0]
        preds.append(float(max(0.0, p_lstm)))
    except Exception as exc:
        if logger is not None:
            logger.log("lstm_predict_one", exc)

    try:
        p_rf = float(rf_model.predict(seq.reshape(1, -1))[0])
        preds.append(float(max(0.0, p_rf)))
    except Exception as exc:
        if logger is not None:
            logger.log("rf_predict_one", exc)

    return _safe_nonnegative_mean(preds, fallback=fallback)