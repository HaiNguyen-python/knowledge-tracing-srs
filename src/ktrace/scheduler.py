"""Daily review schedulers and the course simulation that compares them under one budget.

Every scheduler sees only what an app would see: which words were shown, when, and whether the
answer was right. Each day it picks at most `budget` words to review. Three families:

* **Leitner boxes:** a right answer moves a word up one box (longer interval), a wrong answer
  sends it back to box 1. Same intervals for everybody.
* **SM-2** (SuperMemo 2, the ancestor of Anki's scheduler): the interval grows by an "ease
  factor" after each right answer; a lapse restarts the word and lowers its ease.
* **Model-based:** predict today's recall of every word and review the words that have just
  dropped below a target (0.85): forgotten enough that reviewing is useful (the spacing
  effect), not so far that the word is lost.

`simulate_course` runs a cohort of simulated learners through the same course with each
scheduler and measures recall on a final test one week after the course.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from scipy import stats

from .predictor import LearnerPredictor, ModelBundle
from .simulator import MINUTE, Catalog, World, WorldConfig, course_order

DUE_TOLERANCE = 1e-6  # sessions happen at the same time each day; "due in 1 day" means today


def select_reviews(p_recall: np.ndarray, budget: int, target: float) -> np.ndarray:
    """Positions of the words to review: predicted recall below `target`, closest first.

    Shared by the model-based scheduler and the API, so both make the same choice."""
    below = np.flatnonzero(p_recall < target)
    order = below[np.argsort(-p_recall[below], kind="stable")]
    return order[: max(budget, 0)]


class Scheduler(Protocol):
    name: str

    def observe(self, item: int, t: float, correct: int) -> None: ...

    def choose(self, t: float, budget: int) -> list[int]: ...


@dataclass
class Leitner:
    intervals: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)
    name: str = "Leitner boxes"
    box: dict[int, int] = field(default_factory=dict)
    last: dict[int, float] = field(default_factory=dict)

    def observe(self, item: int, t: float, correct: int) -> None:
        current = self.box.get(item, 0)
        self.box[item] = min(current + 1, len(self.intervals) - 1) if correct else 0
        self.last[item] = t

    def choose(self, t: float, budget: int) -> list[int]:
        """Due words, lowest box first (the least known), then the longest overdue."""
        due = [
            i for i, b in self.box.items() if self.last[i] + self.intervals[b] <= t + DUE_TOLERANCE
        ]
        due.sort(key=lambda i: (self.box[i], self.last[i] + self.intervals[self.box[i]]))
        return due[:budget]


@dataclass
class _Card:
    reps: int = 0
    ease: float = 2.5
    interval: float = 1.0
    last: float = 0.0


@dataclass
class SM2:
    """SM-2 with binary grades: right = quality 4 (ease unchanged), wrong = a lapse.

    On a lapse SM-2 restarts the word with a 1-day interval; like Anki, the ease also drops by
    0.2 (never below 1.3), so words that keep failing come back more often."""

    name: str = "SM-2"
    cards: dict[int, _Card] = field(default_factory=dict)

    def observe(self, item: int, t: float, correct: int) -> None:
        card = self.cards.setdefault(item, _Card())
        if correct:
            card.interval = (
                1.0 if card.reps == 0 else 6.0 if card.reps == 1 else card.interval * card.ease
            )
            card.reps += 1
        else:
            card.reps, card.interval = 0, 1.0
            card.ease = max(1.3, card.ease - 0.2)
        card.last = t

    def choose(self, t: float, budget: int) -> list[int]:
        """Due words, most overdue (relative to their interval) first."""
        due = [i for i, c in self.cards.items() if c.last + c.interval <= t + DUE_TOLERANCE]
        due.sort(key=lambda i: -(t - self.cards[i].last) / self.cards[i].interval)
        return due[:budget]


class RecallSource(Protocol):
    def observe(self, item: int, t: float, correct: int) -> None: ...

    def predict(self, items: Sequence[int], t: float) -> np.ndarray: ...


class HLRSource:
    """Recall predicted by the fitted model from the learner's visible history."""

    def __init__(self, bundle: ModelBundle, catalog: Catalog) -> None:
        self.catalog = catalog
        self.predictor = LearnerPredictor(bundle)

    def observe(self, item: int, t: float, correct: int) -> None:
        cat = self.catalog
        self.predictor.observe(cat.item_ids[item], cat.skills[item], t, correct)

    def predict(self, items: Sequence[int], t: float) -> np.ndarray:
        cat = self.catalog
        ids = [cat.item_ids[i] for i in items]
        return self.predictor.predict(ids, [cat.skills[i] for i in items], t)


class OracleSource:
    """The simulator's true recall: the best any model could do. A ceiling, not a product."""

    def __init__(self, world: World, learner: int) -> None:
        self.world, self.learner = world, learner

    def observe(self, item: int, t: float, correct: int) -> None:
        return None

    def predict(self, items: Sequence[int], t: float) -> np.ndarray:
        return self.world.p_correct(self.learner, np.asarray(items, dtype=int), t)


@dataclass
class ModelBased:
    source: RecallSource
    target: float = 0.85
    name: str = "Model-based"
    seen: list[int] = field(default_factory=list)

    def observe(self, item: int, t: float, correct: int) -> None:
        if item not in self.seen:
            self.seen.append(item)
        self.source.observe(item, t, correct)

    def choose(self, t: float, budget: int) -> list[int]:
        if not self.seen:
            return []
        p = self.source.predict(self.seen, t)
        return [self.seen[k] for k in select_reviews(p, budget, self.target)]


@dataclass(frozen=True)
class CourseConfig:
    n_learners: int = 30
    n_days: int = 30
    new_per_day: int = 5
    budget: int = 20  # reviews per day, on top of the new words
    test_delay_days: float = 7.0  # final test one week after the last study day
    target: float = 0.85
    session_time: float = 0.75  # every learner studies at 18:00


SchedulerFactory = Callable[[World, int], Scheduler]


def scheduler_factories(
    bundle: ModelBundle, catalog: Catalog, target: float
) -> dict[str, SchedulerFactory]:
    return {
        "Leitner boxes": lambda world, learner: Leitner(),
        "SM-2": lambda world, learner: SM2(),
        "Model-based (HLR)": lambda world, learner: ModelBased(
            HLRSource(bundle, catalog), target, "Model-based (HLR)"
        ),
        "Model-based (true recall)": lambda world, learner: ModelBased(
            OracleSource(world, learner), target, "Model-based (true recall)"
        ),
    }


def simulate_course(
    catalog: Catalog,
    factory: SchedulerFactory,
    seed: int,
    cfg: CourseConfig,
    world_config: WorldConfig | None = None,
) -> dict[str, float]:
    """One cohort, one scheduler. Returns final-test recall and review counts.

    Recall on the final test is the *expected* score (mean true P(correct) over the course
    words), not a sampled one: same meaning, less noise."""
    world = World(catalog, cfg.n_learners, seed, world_config or WorldConfig())
    order = course_order(catalog)
    retention, reviews = [], []
    for learner in range(cfg.n_learners):
        scheduler = factory(world, learner)
        n_reviews = 0
        for day in range(cfg.n_days):
            t = day + cfg.session_time
            chosen = scheduler.choose(t, cfg.budget)
            new = order[day * cfg.new_per_day : (day + 1) * cfg.new_per_day]
            for item in [*chosen, *new]:
                correct, _ = world.answer(learner, int(item), t)
                scheduler.observe(int(item), t, correct)
                t += MINUTE
            n_reviews += len(chosen)
        studied = order[: cfg.n_days * cfg.new_per_day]
        t_test = cfg.n_days - 1 + cfg.session_time + cfg.test_delay_days
        retention.append(float(world.p_correct(learner, studied, t_test).mean()))
        reviews.append(n_reviews / cfg.n_days)
    return {"retention": float(np.mean(retention)), "reviews_per_day": float(np.mean(reviews))}


def compare_schedulers(
    catalog: Catalog,
    bundle: ModelBundle,
    seeds: Sequence[int],
    cfg: CourseConfig | None = None,
    world_config: WorldConfig | None = None,
) -> pd.DataFrame:
    """Every scheduler on the same cohorts (same seeds). One row per (seed, scheduler)."""
    cfg = cfg or CourseConfig()
    rows = []
    for name, factory in scheduler_factories(bundle, catalog, cfg.target).items():
        for seed in seeds:
            result = simulate_course(catalog, factory, seed, cfg, world_config)
            rows.append({"seed": seed, "scheduler": name, **result})
    return pd.DataFrame(rows)


def mean_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and a t-distribution confidence interval across seeds."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    half = float(stats.t.ppf(0.5 + level / 2, len(values) - 1) * stats.sem(values))
    return mean, mean - half, mean + half


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name, g in results.groupby("scheduler", sort=False):
        mean, low, high = mean_ci(g["retention"].to_numpy())
        rows[name] = {
            "retention": mean,
            "ci_low": low,
            "ci_high": high,
            "reviews_per_day": float(g["reviews_per_day"].mean()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def paired_gain(results: pd.DataFrame, a: str, b: str) -> tuple[float, float, float]:
    """Retention of `a` minus `b` on the same seeds, with a 95% CI (paired design)."""
    wide = results.pivot(index="seed", columns="scheduler", values="retention")
    return mean_ci((wide[a] - wide[b]).to_numpy())
