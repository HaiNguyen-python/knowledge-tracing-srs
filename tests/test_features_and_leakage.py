import numpy as np
import pandas as pd
import pytest

from ktrace.benchmark import run_benchmark
from ktrace.data.split import split_by_learner_and_time
from ktrace.features import FEATURES, build_features, estimate_difficulty


def test_features_count_only_earlier_answers():
    log = pd.DataFrame(
        {
            "user_id": ["a", "a", "a", "b"],
            "item_id": ["w1", "w1", "w2", "w1"],
            "skill_id": ["health", "health", "health", "health"],
            "t": [0.5, 2.0, 2.1, 0.7],
            "correct": [0, 1, 1, 1],
        }
    )
    f = build_features(log, {"w1": 0.4}, default_difficulty=-1.0)
    a = f[f["user_id"] == "a"].reset_index(drop=True)
    assert a.loc[0, ["succ_item", "fail_item", "first_seen", "delta_days"]].tolist() == [0, 0, 1, 0]
    assert a.loc[1, ["succ_item", "fail_item", "first_seen"]].tolist() == [0, 1, 0]
    assert a.loc[1, "delta_days"] == pytest.approx(1.5)
    assert a.loc[2, ["succ_skill", "fail_skill", "first_seen"]].tolist() == [1, 1, 1]
    assert a.loc[2, "item_difficulty"] == -1.0  # unknown word -> default
    # Learner b's history is separate from a's.
    b = f[f["user_id"] == "b"].iloc[0]
    assert b["first_seen"] == 1 and b["learner_rate"] == 0.0


def test_features_do_not_change_when_the_future_changes(log):
    diff, default = estimate_difficulty(log)
    cutoff = log["t"].quantile(0.5)
    before = build_features(log, diff, default)
    tampered = log.copy()
    future = tampered["t"] >= cutoff
    tampered.loc[future, "correct"] = 1 - tampered.loc[future, "correct"]
    after = build_features(tampered, diff, default)
    # Up to and including the first changed rows: their own label changed, their features not.
    upto = (before["t"] <= cutoff).to_numpy()
    np.testing.assert_array_equal(before.loc[upto, FEATURES], after.loc[upto, FEATURES])
    assert not np.array_equal(before.loc[~upto, FEATURES], after.loc[~upto, FEATURES])


def test_split_separates_learners_and_time(log):
    split = split_by_learner_and_time(log, seed=0)
    assert not set(split.train["user_id"]) & set(split.test["user_id"])
    assert split.train["t"].max() < split.cutoff
    assert (split.test["t"] >= split.cutoff).sum() > 0


def test_test_labels_never_reach_training(log):
    """Flip every scored answer of the held-out learners: the fitted model must not change."""
    bench = run_benchmark(log, seed=0)
    split = bench.split
    scored = log["user_id"].isin(set(split.test["user_id"])) & (log["t"] >= split.cutoff)
    flipped = log.assign(correct=np.where(scored, 1 - log["correct"], log["correct"]))
    again = run_benchmark(flipped, seed=0)
    np.testing.assert_array_equal(bench.bundle.model.w, again.bundle.model.w)
    assert bench.bundle.difficulty == again.bundle.difficulty
