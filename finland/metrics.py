"""The two scoring helpers the Finnish pipeline needs (copied from Predicting-Nba research/evaluate.py)."""

import numpy as np
import pandas as pd

EPS = 1e-12


def expected_calibration_error(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def classification_metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return {"n": int(len(y)), "accuracy": float(np.mean((p >= 0.5) == (y == 1))), "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), "ece": expected_calibration_error(y, p)}


def calibration_table(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)) - 1, 0, bins - 1)
    return pd.DataFrame([{"bin": b, "mean_pred": p[idx == b].mean(), "observed": y[idx == b].mean(), "count": int((idx == b).sum())}
                         for b in range(bins) if (idx == b).any()])
