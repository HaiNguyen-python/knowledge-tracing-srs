"""Recall models. All share one small interface so evaluation treats them the same way.

`fit(frame)` learns from training rows. `predict(frame)` returns P(correct) for every row, using
only what was known before that row: the history features (see `features.py`) or, for BKT, the
earlier answers of the same learner in the same frame.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class RecallModel(Protocol):
    name: str

    def fit(self, frame: pd.DataFrame) -> RecallModel: ...

    def predict(self, frame: pd.DataFrame) -> np.ndarray: ...
