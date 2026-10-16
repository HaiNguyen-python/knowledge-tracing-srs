"""Performance Factors Analysis style logistic model (Pavlik, Cen & Koedinger, 2009), extended.

Classic PFA: logit P(correct) = item easiness + gamma * prior successes + rho * prior failures,
counted per skill. Here the counts are taken per word and per topic, and three more inputs are
added: time since the word was last seen (log), whether it is a first encounter, and the
learner's overall success rate. Counts enter as log(1 + n) because the 10th review adds less
than the 2nd.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def pfa_design(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.log1p(frame["succ_item"]),
            np.log1p(frame["fail_item"]),
            np.log1p(frame["succ_skill"]),
            np.log1p(frame["fail_skill"]),
            np.log1p(frame["delta_days"]),
            frame["first_seen"],
            frame["item_difficulty"],
            frame["learner_rate"],
        ]
    )


class PFA:
    name = "PFA (logistic)"

    def __init__(self, c: float = 1.0) -> None:
        self.pipeline: Pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=c))

    def fit(self, frame: pd.DataFrame) -> PFA:
        self.pipeline.fit(pfa_design(frame), frame["correct"].to_numpy())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(pfa_design(frame))[:, 1]
