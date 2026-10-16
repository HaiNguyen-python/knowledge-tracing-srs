import numpy as np
import pandas as pd
import pytest

from ktrace.benchmark import CEILING, beats_baselines
from ktrace.models.baselines import GlobalMean, ItemMean

REQUIRED = ["BKT (per skill)", "PFA (logistic)", "HLR (half-life regression)"]


@pytest.mark.parametrize("model", REQUIRED)
def test_model_beats_both_baselines_on_held_out_learners(bench, model):
    assert beats_baselines(bench, model), bench.metrics.loc[[model, "Item mean"]]


def test_no_model_beats_the_true_recall_ceiling_by_more_than_noise(bench):
    m = bench.metrics
    best = m.drop(index=CEILING)["log_loss"].min()
    assert m.loc[CEILING, "log_loss"] <= best + 0.005


def test_baselines():
    train = pd.DataFrame({"item_id": ["a"] * 4 + ["b"], "correct": [1, 1, 1, 1, 0]})
    item = ItemMean(prior_weight=1.0).fit(train)
    p = item.predict(pd.DataFrame({"item_id": ["a", "b", "new"]}))
    assert p[0] == pytest.approx((4 + 0.8) / 5)  # shrunk towards the global rate 0.8
    assert p[2] == pytest.approx(0.8)  # unseen word -> global rate
    assert np.all(GlobalMean().fit(train).predict(train) == 0.8)
