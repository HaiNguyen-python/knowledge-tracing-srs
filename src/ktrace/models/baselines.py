"""Baselines every real model must beat."""

from __future__ import annotations

import numpy as np
import pandas as pd


class GlobalMean:
    """The majority-class baseline as a probability: always predict the training success rate.

    A hard 0/1 majority vote would have infinite log loss, so the rate is used instead. Its AUC
    is 0.5 by definition: it cannot rank answers at all."""

    name = "Global mean (majority class)"

    def __init__(self) -> None:
        self.rate = 0.5

    def fit(self, frame: pd.DataFrame) -> GlobalMean:
        self.rate = float(frame["correct"].mean())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.rate)


class ItemMean:
    """Success rate of each word in training, shrunk towards the global rate.

    `prior_weight` acts like that many extra answers at the global rate, so rare words do not
    get extreme estimates."""

    name = "Item mean"

    def __init__(self, prior_weight: float = 5.0) -> None:
        self.prior_weight = prior_weight
        self.rate = 0.5
        self.item_rate: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame) -> ItemMean:
        self.rate = float(frame["correct"].mean())
        stats = frame.groupby("item_id")["correct"].agg(["sum", "count"])
        shrunk = (stats["sum"] + self.prior_weight * self.rate) / (
            stats["count"] + self.prior_weight
        )
        self.item_rate = {str(k): float(v) for k, v in shrunk.items()}
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["item_id"].map(self.item_rate).fillna(self.rate).to_numpy(dtype=float)
