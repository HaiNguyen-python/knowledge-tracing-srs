import numpy as np
import pandas as pd
import pytest

from ktrace.simulator import World, WorldConfig, generate_log, make_catalog


def test_same_seed_same_log_different_seed_different_log(catalog):
    a = generate_log(catalog, n_learners=20, n_days=10, seed=5)
    b = generate_log(catalog, n_learners=20, n_days=10, seed=5)
    c = generate_log(catalog, n_learners=20, n_days=10, seed=6)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)
    assert make_catalog(seed=0).difficulty.tolist() == make_catalog(seed=0).difficulty.tolist()


def test_memory_halves_after_one_half_life(catalog):
    world = World(catalog, n_learners=1, seed=0)
    world.answer(0, 3, t=0.0)
    h = world.half_life[0, 3]
    assert world.memory(0, np.array([3]), t=h)[0] == pytest.approx(0.5)
    assert world.memory(0, np.array([3]), t=2 * h)[0] == pytest.approx(0.25)


def _half_life_after_success(memory_at_review: float) -> float:
    """Half-life gain of one successful review at a given memory level (no guessing noise)."""
    cfg = WorldConfig(guess=1.0, slip=0.0)  # every answer is correct
    world = World(make_catalog(), n_learners=1, seed=0, config=cfg)
    world.answer(0, 0, t=0.0)
    h = world.half_life[0, 0]
    world.answer(0, 0, t=-h * np.log2(memory_at_review))
    return world.half_life[0, 0] / h


def test_spacing_effect_is_planted():
    """Reviewing a word that is half forgotten helps more than reviewing it right away."""
    assert _half_life_after_success(0.5) > _half_life_after_success(0.9) > 1.0
    assert _half_life_after_success(0.999) == pytest.approx(1.0, abs=0.2)


def test_lapse_never_drops_below_a_fresh_study(catalog):
    cfg = WorldConfig(guess=0.0, slip=1.0)  # every answer is wrong
    world = World(catalog, n_learners=1, seed=0, config=cfg)
    for day in range(5):
        world.answer(0, 7, t=float(day))
    assert world.half_life[0, 7] == pytest.approx(cfg.h_new * world.scale[0, 7])
