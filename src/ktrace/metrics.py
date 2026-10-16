"""Metrics for probability predictions of 0/1 answers.

* AUC: can the model rank right answers above wrong ones? Ignores calibration.
* Log loss: are the probabilities good? A scheduler acts on the probability itself
  ("review when recall < 0.85"), so calibration matters as much as ranking here.
* ECE (expected calibration error): average gap between predicted and observed rates in 10
  equal-width bins, weighted by bin size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

P_CLIP = 1e-6


def auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    pc = np.clip(p, P_CLIP, 1.0 - P_CLIP)
    return float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)))


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Reliability-chart data: mean prediction vs observed rate per probability bin."""
    bins = np.minimum((np.asarray(p) * n_bins).astype(int), n_bins - 1)
    df = pd.DataFrame({"bin": bins, "p": p, "y": y})
    table = df.groupby("bin").agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return table.reset_index()


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    table = calibration_table(y, p, n_bins)
    weights = table["n"] / table["n"].sum()
    return float((weights * (table["mean_pred"] - table["observed"]).abs()).sum())


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auc": auc(y, p),
        "log_loss": log_loss(y, p),
        "ece": expected_calibration_error(y, p),
        "n": float(len(y)),
    }


def per_group_auc(frame: pd.DataFrame, columns: list[str], by: str = "skill_id") -> pd.DataFrame:
    """AUC of each prediction column within each group (e.g. IELTS topic)."""
    rows = {
        group: {col: auc(g["correct"].to_numpy(), g[col].to_numpy()) for col in columns}
        for group, g in frame.groupby(by)
    }
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()
