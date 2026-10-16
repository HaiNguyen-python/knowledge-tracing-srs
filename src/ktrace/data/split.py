"""Train/test split by learner AND by time.

Two kinds of leakage are possible in knowledge tracing:

* **Learner leakage:** the same learner in train and test lets a model memorise that person.
  Fix: test learners are never used for fitting.
* **Time leakage:** fitting on answers given after the moment being predicted. Fix: training
  rows end at a cutoff time and scored rows start at it.

Test learners' answers *before* the cutoff are kept as context: a model may read a learner's
history to predict their next answer (that is the task), but never fits on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame  # train learners, t < cutoff: the only rows models fit on
    test: pd.DataFrame  # every row of the test learners (history + scored rows)
    cutoff: float  # test rows with t >= cutoff are scored


def split_by_learner_and_time(
    log: pd.DataFrame, test_fraction: float = 0.3, cutoff_quantile: float = 0.6, seed: int = 0
) -> Split:
    learners = np.sort(log["user_id"].unique())
    rng = np.random.default_rng(seed)
    n_test = max(1, round(len(learners) * test_fraction))
    test_learners = set(rng.choice(learners, size=n_test, replace=False))
    cutoff = float(log["t"].quantile(cutoff_quantile))
    is_test = log["user_id"].isin(test_learners)
    train = log[~is_test & (log["t"] < cutoff)].reset_index(drop=True)
    test = log[is_test].reset_index(drop=True)
    return Split(train=train, test=test, cutoff=cutoff)
