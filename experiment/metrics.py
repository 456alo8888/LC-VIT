from __future__ import annotations

from typing import Dict

import numpy as np


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    diff = y_pred - y_true

    mse = float(np.mean(np.square(diff)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    eps = 1e-8
    mape = float(np.mean(np.abs(diff) / np.maximum(np.abs(y_true), eps)) * 100.0)
    ss_res = float(np.sum(np.square(diff)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = float(1.0 - (ss_res / max(ss_tot, eps)))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "loss": mse,
    }
