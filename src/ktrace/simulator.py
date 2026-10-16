"""The synthetic "world": learners, IELTS vocabulary and memory that decays over time.

The simulator is the ground truth for every number in this repo. Nothing in the models reads its
hidden state; they only see the practice log it writes (who answered which word, when, and
whether the answer was right). The effects below were planted on purpose so they can be found:

* Learners differ in ability and words differ in difficulty (grouped by IELTS topic).
* Memory decays exponentially: recall = 2^(-days_since_review / half_life), the same shape as
  half-life regression (Settles & Meeder, 2016).
* A successful review multiplies the half-life, and the gain is larger when the word was harder
  to recall at that moment (the spacing effect). Reviewing a word you remember perfectly
  teaches you almost nothing.
* A failed review shrinks the half-life (a lapse), but never below the half-life of a freshly
  studied word: the learner sees the answer again, so the memory clock restarts.
* Answers are noisy: a small guess rate and a small slip rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TOPICS: tuple[str, ...] = (
    "environment",
    "education",
    "health",
    "technology",
    "work",
    "travel",
    "crime",
    "media",
)
LOG_COLUMNS = ["user_id", "item_id", "skill_id", "t", "correct", "p_true"]
MINUTE = 1.0 / 1440.0


@dataclass(frozen=True)
class WorldConfig:
    """Parameters of the memory model. Times are in days."""

    guess: float = 0.05  # correct answer although the word is forgotten
    slip: float = 0.03  # wrong answer although the word is remembered
    prior_offset: float = 1.5  # higher = fewer words already known before studying them
    h_new: float = 1.5  # half-life after first studying an unknown word
    h_known: float = 6.0  # half-life when the learner already knew the word
    growth: float = 3.0  # largest extra half-life multiplier from one successful review
    spacing_power: float = 0.5  # how strongly the gain depends on how hard recall was
    lapse_factor: float = 0.4  # half-life multiplier after a failed review
    ability_effect: float = 0.4  # effect of (ability - difficulty) on half-life, log scale
    h_min: float = 0.25
    h_max: float = 365.0


@dataclass(frozen=True)
class Catalog:
    """The words of a course: id, IELTS topic (the "skill") and true difficulty."""

    item_ids: tuple[str, ...]
    skills: tuple[str, ...]
    difficulty: np.ndarray

    def __len__(self) -> int:
        return len(self.item_ids)


def make_catalog(n_per_topic: int = 15, seed: int = 0) -> Catalog:
    """Words grouped by topic. Topics have their own mean difficulty (crime words are not as easy
    as travel words), so a per-skill model such as BKT has something real to learn."""
    rng = np.random.default_rng(seed)
    topic_offset = rng.normal(0.0, 0.5, len(TOPICS))
    ids, skills, diffs = [], [], []
    for k, topic in enumerate(TOPICS):
        for j in range(n_per_topic):
            ids.append(f"{topic[:4]}_{j:03d}")
            skills.append(topic)
            diffs.append(topic_offset[k] + rng.normal(0.0, 0.8))
    return Catalog(tuple(ids), tuple(skills), np.asarray(diffs))


@dataclass
class World:
    """Hidden memory state of a cohort of learners studying one catalog."""

    catalog: Catalog
    n_learners: int
    seed: int
    config: WorldConfig = field(default_factory=WorldConfig)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.ability = rng.normal(0.0, 1.0, self.n_learners)
        # One random stream per learner: two schedulers run on the same seed then face the
        # same learners with the same luck, so their results can be compared pair by pair.
        self.rngs = [np.random.default_rng([self.seed, k]) for k in range(self.n_learners)]
        cfg = self.config
        gap = self.ability[:, None] - self.catalog.difficulty[None, :]
        # Able learners and easy words keep memories longer (a multiplier on the half-life) and
        # are more likely to be known before studying.
        self.scale = np.exp(cfg.ability_effect * gap)
        self.prior = 1.0 / (1.0 + np.exp(-(gap - cfg.prior_offset)))
        self.half_life = np.full(gap.shape, np.nan)
        self.last_seen = np.full(gap.shape, np.nan)

    def memory(self, learner: int, items: np.ndarray, t: float) -> np.ndarray:
        """Probability that each word is in memory at time t (before answering)."""
        h = self.half_life[learner, items]
        seen = ~np.isnan(h)
        dt = np.maximum(t - self.last_seen[learner, items], 0.0)
        decayed = np.power(2.0, -np.where(seen, dt, 0.0) / np.where(seen, h, 1.0))
        return np.where(seen, decayed, self.prior[learner, items])

    def p_correct(self, learner: int, items: np.ndarray, t: float) -> np.ndarray:
        """Probability of a correct answer: memory plus guess and slip noise."""
        cfg = self.config
        return cfg.guess + (1.0 - cfg.guess - cfg.slip) * self.memory(learner, items, t)

    def answer(self, learner: int, item: int, t: float) -> tuple[int, float]:
        """The learner answers one word at time t. Returns (correct, p_true) and updates memory.

        After every answer the learner sees the right answer, so the memory clock restarts.
        Scalar maths (not numpy) because this runs once per simulated answer."""
        cfg = self.config
        h = float(self.half_life[learner, item])
        seen = not math.isnan(h)
        if seen:
            m = 2.0 ** (-max(t - float(self.last_seen[learner, item]), 0.0) / h)
        else:
            m = float(self.prior[learner, item])
        p = cfg.guess + (1.0 - cfg.guess - cfg.slip) * m
        correct = int(self.rngs[learner].random() < p)
        scale = float(self.scale[learner, item])
        if not seen:
            new_h = (cfg.h_known if correct else cfg.h_new) * scale
        elif correct:
            new_h = h * (1.0 + cfg.growth * scale * (1.0 - m) ** cfg.spacing_power)
        else:
            # A lapse shrinks the half-life, but the learner sees the answer again, which is at
            # least as good as studying the word for the first time.
            new_h = max(h * cfg.lapse_factor, cfg.h_new * scale)
        self.half_life[learner, item] = min(max(new_h, cfg.h_min), cfg.h_max)
        self.last_seen[learner, item] = t
        return correct, p


def course_order(catalog: Catalog) -> np.ndarray:
    """Words are taught topic by topic, in catalog order, like units in a textbook."""
    return np.arange(len(catalog))


def generate_log(
    catalog: Catalog,
    n_learners: int = 400,
    n_days: int = 30,
    seed: int = 1,
    config: WorldConfig | None = None,
) -> pd.DataFrame:
    """Practice logs from learners who review words at random, as many real apps let them.

    Random reviews give a wide spread of gaps between reviews (from hours to weeks), which a
    model needs to learn how fast memory decays. Each learner has their own pace and habit."""
    world = World(catalog, n_learners, seed, config or WorldConfig())
    rng = np.random.default_rng(seed + 10_000)
    rows: list[tuple[str, str, str, float, int, float]] = []
    order = course_order(catalog)
    for learner in range(n_learners):
        user = f"learner_{learner:04d}"
        p_active = rng.uniform(0.5, 0.95)
        new_per_day = int(rng.integers(3, 8))
        introduced: list[int] = []
        for day in range(n_days):
            if rng.random() > p_active:
                continue
            t = day + rng.uniform(0.3, 0.9)
            n_reviews = min(len(introduced), int(rng.poisson(10)))
            reviews: list[int] = []
            if n_reviews:
                # Recently seen words are picked more often, as in most vocabulary apps.
                pool = np.asarray(introduced)
                weights = 1.0 / (1.0 + t - world.last_seen[learner, pool])
                reviews = list(
                    rng.choice(pool, size=n_reviews, replace=False, p=weights / weights.sum())
                )
            fresh = order[len(introduced) : len(introduced) + new_per_day]
            for item in [*reviews, *fresh]:
                correct, p = world.answer(learner, int(item), t)
                rows.append((user, catalog.item_ids[item], catalog.skills[item], t, correct, p))
                t += MINUTE
            introduced.extend(int(i) for i in fresh)
    return pd.DataFrame(rows, columns=LOG_COLUMNS)
