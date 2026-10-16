from dataclasses import dataclass, field

import numpy as np
import pytest

from ktrace.scheduler import (
    SM2,
    CourseConfig,
    Leitner,
    compare_schedulers,
    mean_ci,
    paired_gain,
    scheduler_factories,
    select_reviews,
    simulate_course,
    summarize,
)

SMALL = CourseConfig(n_learners=4, n_days=12, new_per_day=6, budget=7)


def test_select_reviews_takes_words_just_below_the_target_within_budget():
    p = np.array([0.95, 0.84, 0.10, 0.60, 0.80, 0.86])
    assert select_reviews(p, budget=2, target=0.85).tolist() == [1, 4]
    assert select_reviews(p, budget=10, target=0.85).tolist() == [1, 4, 3, 2]
    assert select_reviews(p, budget=0, target=0.85).tolist() == []


def test_leitner_boxes_move_up_on_success_and_reset_on_failure():
    s = Leitner()
    s.observe(1, 0.0, 1)  # new word answered right -> box 1 (2-day interval)
    assert s.choose(1.0, 10) == [] and s.choose(2.0, 10) == [1]
    s.observe(1, 2.0, 1)
    s.observe(1, 6.0, 0)
    assert s.box[1] == 0 and s.choose(7.0, 10) == [1]


def test_sm2_intervals_grow_1_6_then_by_ease_and_lapse_lowers_ease():
    s = SM2()
    for t in (0.0, 1.0, 7.0):
        s.observe(1, t, 1)
    card = s.cards[1]
    assert card.interval == pytest.approx(6 * 2.5) and card.reps == 3
    s.observe(1, 22.0, 0)
    assert (card.interval, card.reps, card.ease) == (1.0, 0, pytest.approx(2.3))


@dataclass
class BudgetSpy:
    """Wraps a scheduler and records every choice it makes."""

    inner: object
    name: str = "spy"
    seen: set[int] = field(default_factory=set)
    choices: list[list[int]] = field(default_factory=list)

    def observe(self, item, t, correct):
        self.seen.add(item)
        self.inner.observe(item, t, correct)

    def choose(self, t, budget):
        chosen = self.inner.choose(t, budget)
        self.choices.append(list(chosen))
        assert len(chosen) <= budget
        assert len(set(chosen)) == len(chosen)
        assert set(chosen) <= self.seen  # only words the learner has already met
        return chosen


@pytest.mark.parametrize(
    "name", ["Leitner boxes", "SM-2", "Model-based (HLR)", "Model-based (true recall)"]
)
def test_every_scheduler_respects_the_budget(bench, catalog, name):
    spies: list[BudgetSpy] = []
    base = scheduler_factories(bench.bundle, catalog, target=0.85)[name]

    def factory(world, learner):
        spies.append(BudgetSpy(base(world, learner)))
        return spies[-1]

    result = simulate_course(catalog, factory, seed=1, cfg=SMALL)
    assert 0 < result["reviews_per_day"] <= SMALL.budget
    assert max(len(c) for spy in spies for c in spy.choices) <= SMALL.budget
    assert 0.0 < result["retention"] < 1.0


def test_comparison_is_deterministic_and_paired(bench, catalog):
    a = compare_schedulers(catalog, bench.bundle, seeds=[1, 2], cfg=SMALL)
    b = compare_schedulers(catalog, bench.bundle, seeds=[1, 2], cfg=SMALL)
    assert a.equals(b)
    summary = summarize(a)
    assert list(summary.index) == list(a["scheduler"].unique())
    assert (summary["ci_low"] <= summary["retention"]).all()
    mean, low, high = paired_gain(a, "Model-based (HLR)", "Leitner boxes")
    assert low <= mean <= high


def test_mean_ci():
    mean, low, high = mean_ci(np.array([1.0, 2.0, 3.0]))
    assert mean == 2.0 and low == pytest.approx(2 - 4.303 * 1 / np.sqrt(3), rel=1e-3)
    assert mean_ci(np.array([5.0])) == (5.0, 5.0, 5.0)
