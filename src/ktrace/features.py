"""History features for one prediction: what a learner has done *before* this moment.

Every feature row is computed by `LearnerHistory.features` *before* `LearnerHistory.update` sees
the answer, so a row can never contain its own label or anything later. Training, evaluation,
the scheduler and the API all use this one class, so features cannot drift between them.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import pandas as pd

FEATURES = [
    "succ_item",  # earlier correct answers on this word
    "fail_item",  # earlier wrong answers on this word
    "succ_skill",  # earlier correct answers on any word of this topic
    "fail_skill",
    "delta_days",  # days since this word was last seen (0 when never seen)
    "first_seen",  # 1 if the learner has never seen this word
    "item_difficulty",  # estimated from training learners only
    "learner_rate",  # smoothed log-odds of the learner's earlier answers
]


def smoothed_log_odds(successes: float, failures: float) -> float:
    """log((s + 1) / (f + 1)): a success rate that stays finite for 0 or 1 answers."""
    return math.log((successes + 1.0) / (failures + 1.0))


class LearnerHistory:
    """Running counts for one learner, updated one answer at a time."""

    def __init__(self, difficulty: Mapping[str, float], default_difficulty: float) -> None:
        self._difficulty = difficulty
        self._default = default_difficulty
        # item or skill -> [successes, failures]
        self._item: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
        self._skill: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
        self._last: dict[str, float] = {}
        self._total = [0, 0]

    def features(self, item: str, skill: str, t: float) -> list[float]:
        s_i, f_i = self._item.get(item, [0, 0])
        s_k, f_k = self._skill.get(skill, [0, 0])
        last = self._last.get(item)
        return [
            float(s_i),
            float(f_i),
            float(s_k),
            float(f_k),
            0.0 if last is None else max(t - last, 0.0),
            1.0 if last is None else 0.0,
            self._difficulty.get(item, self._default),
            smoothed_log_odds(*self._total),
        ]

    def update(self, item: str, skill: str, t: float, correct: int) -> None:
        slot = 0 if correct else 1
        self._item[item][slot] += 1
        self._skill[skill][slot] += 1
        self._total[slot] += 1
        self._last[item] = t


def estimate_difficulty(log: pd.DataFrame) -> tuple[dict[str, float], float]:
    """Item difficulty = -smoothed log-odds of correct answers, from the given (training) rows.

    Returns the map and a default (the median) for words no training learner has answered."""
    counts = log.groupby("item_id")["correct"].agg(["sum", "count"])
    diff = {
        str(item): -smoothed_log_odds(row["sum"], row["count"] - row["sum"])
        for item, row in counts.iterrows()
    }
    default = float(np.median(list(diff.values()))) if diff else 0.0
    return diff, default


def sort_log(log: pd.DataFrame) -> pd.DataFrame:
    """Time order within each learner; a stable sort keeps the original order on ties."""
    return log.sort_values(["user_id", "t"], kind="stable").reset_index(drop=True)


def build_features(
    log: pd.DataFrame, difficulty: Mapping[str, float], default_difficulty: float
) -> pd.DataFrame:
    """Add the FEATURES columns to a log. Each row only uses earlier rows of the same learner."""
    log = sort_log(log)
    values = np.empty((len(log), len(FEATURES)))
    histories: dict[str, LearnerHistory] = {}
    columns = zip(
        log["user_id"], log["item_id"], log["skill_id"], log["t"], log["correct"], strict=True
    )
    for i, (user, item, skill, t, correct) in enumerate(columns):
        history = histories.get(user)
        if history is None:
            history = histories[user] = LearnerHistory(difficulty, default_difficulty)
        values[i] = history.features(item, skill, t)
        history.update(item, skill, t, int(correct))
    return pd.concat([log, pd.DataFrame(values, columns=FEATURES)], axis=1)
