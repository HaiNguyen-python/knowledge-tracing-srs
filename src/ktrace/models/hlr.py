"""Half-life regression (Settles & Meeder, 2016): the model behind Duolingo's practice scheduler.

Recall decays with time since the last review, Δ (days), with a half-life h that the model
learns from the learner's history x:

    h = 2^(w · x)          p = 2^(-Δ / h)

When Δ = h, recall is 50%. Each feature in x stretches or shrinks the half-life: a positive weight
on correct answers means memory lasts longer after each success.

Differences from the paper, on purpose:

* Loss: the paper fits the fraction of correct answers in a session with squared error. Here
  every row is one 0/1 answer, so the loss is log loss (plus L2 on the non-bias weights).
* First encounters have no Δ, so they get a separate small logistic model on item difficulty
  and learner rate. HLR proper is used for every review.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from ..features import FEATURES

LN2 = float(np.log(2.0))
P_EPS = 0.01  # no answer is certain: typos, lucky guesses (the paper uses 1e-4)
LOG2_H_MIN, LOG2_H_MAX = np.log2(1.0 / 96.0), np.log2(365.0)  # 15 minutes .. 1 year
HLR_INPUTS = ["bias", "sqrt_succ_item", "sqrt_fail_item", "item_difficulty", "learner_rate"]
FIRST_INPUTS = ["item_difficulty", "learner_rate"]
_IDX = {name: i for i, name in enumerate(FEATURES)}


def hlr_design(x: np.ndarray) -> np.ndarray:
    """HLR inputs from a FEATURES matrix. sqrt(1 + n) as in the paper: diminishing returns."""
    return np.column_stack(
        [
            np.ones(len(x)),
            np.sqrt(1.0 + x[:, _IDX["succ_item"]]),
            np.sqrt(1.0 + x[:, _IDX["fail_item"]]),
            x[:, _IDX["item_difficulty"]],
            x[:, _IDX["learner_rate"]],
        ]
    )


def recall_probability(w: np.ndarray, design: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """p = 2^(-Δ/h) with h = 2^(w·x), clipped to a 15-minute .. 1-year half-life."""
    log2_h = np.clip(design @ w, LOG2_H_MIN, LOG2_H_MAX)
    return np.power(2.0, -delta / np.power(2.0, log2_h))


def loss_and_grad(
    w: np.ndarray, design: np.ndarray, delta: np.ndarray, y: np.ndarray, l2: float
) -> tuple[float, np.ndarray]:
    """Mean log loss + l2 * |w[1:]|^2, and its gradient.

    ln p = -Δ ln2 · 2^(-w·x), so d ln p / dw = ln2² · (Δ/h) · x and dp/dw = p · ln2² · (Δ/h) · x.
    Where the half-life is clipped the gradient is zero."""
    z = design @ w
    inside = (z > LOG2_H_MIN) & (z < LOG2_H_MAX)
    h = np.power(2.0, np.clip(z, LOG2_H_MIN, LOG2_H_MAX))
    p = np.power(2.0, -delta / h)
    pc = np.clip(p, P_EPS, 1.0 - P_EPS)
    loss = -np.mean(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc))
    dloss_dp = (pc - y) / (pc * (1.0 - pc))
    dp_dz = np.where(inside & (p > P_EPS) & (p < 1.0 - P_EPS), p * LN2**2 * delta / h, 0.0)
    grad = design.T @ (dloss_dp * dp_dz) / len(y)
    reg = w.copy()
    reg[0] = 0.0
    return float(loss + l2 * reg @ reg), grad + 2.0 * l2 * reg


class HLR:
    name = "HLR (half-life regression)"

    def __init__(self, l2: float = 1e-3) -> None:
        self.l2 = l2
        self.w = np.zeros(len(HLR_INPUTS))
        self.first_coef = np.zeros(len(FIRST_INPUTS))
        self.first_intercept = 0.0

    def fit(self, frame: pd.DataFrame) -> HLR:
        x = frame[FEATURES].to_numpy(dtype=float)
        y = frame["correct"].to_numpy(dtype=float)
        review = x[:, _IDX["first_seen"]] == 0
        design, delta = hlr_design(x[review]), x[review, _IDX["delta_days"]]
        start = np.zeros(len(HLR_INPUTS))
        start[0] = 1.0  # a 2-day half-life is a sensible starting point
        result = minimize(
            loss_and_grad,
            start,
            args=(design, delta, y[review], self.l2),
            jac=True,
            method="L-BFGS-B",
        )
        self.w = result.x
        first = LogisticRegression().fit(self._first_design(x[~review]), y[~review])
        self.first_coef = first.coef_[0]
        self.first_intercept = float(first.intercept_[0])
        return self

    @staticmethod
    def _first_design(x: np.ndarray) -> np.ndarray:
        return x[:, [_IDX[name] for name in FIRST_INPUTS]]

    def half_life(self, x: np.ndarray) -> np.ndarray:
        """Predicted half-life in days for each FEATURES row."""
        return np.power(2.0, np.clip(hlr_design(x) @ self.w, LOG2_H_MIN, LOG2_H_MAX))

    def predict_matrix(self, x: np.ndarray) -> np.ndarray:
        """P(correct) for a FEATURES matrix (used by the scheduler and the API)."""
        review_p = recall_probability(self.w, hlr_design(x), x[:, _IDX["delta_days"]])
        logit = self._first_design(x) @ self.first_coef + self.first_intercept
        first_p = 1.0 / (1.0 + np.exp(-logit))
        p = np.where(x[:, _IDX["first_seen"]] == 1, first_p, review_p)
        return np.clip(p, P_EPS, 1.0 - P_EPS)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_matrix(frame[FEATURES].to_numpy(dtype=float))

    def to_dict(self) -> dict[str, Any]:
        return {
            "l2": self.l2,
            "w": dict(zip(HLR_INPUTS, map(float, self.w), strict=True)),
            "first_coef": dict(zip(FIRST_INPUTS, map(float, self.first_coef), strict=True)),
            "first_intercept": self.first_intercept,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HLR:
        model = cls(l2=float(data["l2"]))
        model.w = np.array([data["w"][name] for name in HLR_INPUTS])
        model.first_coef = np.array([data["first_coef"][name] for name in FIRST_INPUTS])
        model.first_intercept = float(data["first_intercept"])
        return model
