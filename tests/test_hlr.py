import numpy as np
import pytest

from ktrace.features import FEATURES
from ktrace.models.hlr import HLR, hlr_design, loss_and_grad

HLR_NAME = "HLR (half-life regression)"


def _rows(delta: np.ndarray, succ: float = 2, fail: float = 0) -> np.ndarray:
    x = np.zeros((len(delta), len(FEATURES)))
    x[:, FEATURES.index("succ_item")] = succ
    x[:, FEATURES.index("fail_item")] = fail
    x[:, FEATURES.index("delta_days")] = delta
    return x


@pytest.fixture(scope="module")
def hlr(bench) -> HLR:
    return bench.bundle.model


def test_recall_decreases_with_time_and_halves_at_the_half_life(hlr):
    p = hlr.predict_matrix(_rows(np.array([0.1, 1, 2, 4, 8, 16])))
    assert np.all(np.diff(p) < 0)
    h = hlr.half_life(_rows(np.array([1.0])))[0]
    assert hlr.predict_matrix(_rows(np.array([h])))[0] == pytest.approx(0.5)


def test_successes_lengthen_and_failures_shorten_the_half_life(hlr):
    base = hlr.half_life(_rows(np.array([1.0]), succ=1, fail=0))[0]
    assert hlr.half_life(_rows(np.array([1.0]), succ=4, fail=0))[0] > base
    assert hlr.half_life(_rows(np.array([1.0]), succ=1, fail=3))[0] < base


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    x = _rows(rng.uniform(0.1, 10, 200), succ=0)
    x[:, FEATURES.index("succ_item")] = rng.integers(0, 6, 200)
    design, delta = hlr_design(x), x[:, FEATURES.index("delta_days")]
    y = (rng.random(200) < 0.6).astype(float)
    w = np.array([1.0, 0.3, -0.2, 0.1, 0.05])
    _, grad = loss_and_grad(w, design, delta, y, l2=1e-3)
    eps = 1e-6
    numeric = [
        (
            loss_and_grad(w + eps * e, design, delta, y, 1e-3)[0]
            - loss_and_grad(w - eps * e, design, delta, y, 1e-3)[0]
        )
        / (2 * eps)
        for e in np.eye(len(w))
    ]
    np.testing.assert_allclose(grad, numeric, rtol=1e-4, atol=1e-7)


def test_serialisation_round_trip_gives_identical_predictions(hlr, bench):
    restored = HLR.from_dict(hlr.to_dict())
    np.testing.assert_allclose(restored.predict(bench.scored), bench.scored[HLR_NAME])
