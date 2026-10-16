import numpy as np
import pandas as pd
import pytest

from ktrace.models.bkt import (
    BKT,
    BKTParams,
    fit_em,
    log_likelihood,
    posterior_known,
    predict_correct,
    to_sequences,
    update_known,
)

PARAMS = BKTParams(p_init=0.3, p_learn=0.1, p_guess=0.2, p_slip=0.1)


def test_update_math_matches_hand_calculation():
    # P(correct) = 0.3 * 0.9 + 0.7 * 0.2 = 0.41
    assert predict_correct(PARAMS, np.array(0.3)) == pytest.approx(0.41)
    # Correct answer: posterior = 0.27 / 0.41; then learning: post + (1 - post) * 0.1
    post = 0.27 / 0.41
    assert posterior_known(PARAMS, np.array(0.3), np.array(1)) == pytest.approx(post)
    assert update_known(PARAMS, np.array(0.3), np.array(1)) == pytest.approx(
        post + (1 - post) * 0.1
    )
    # Wrong answer: posterior = 0.3 * 0.1 / (0.03 + 0.7 * 0.8) = 0.03 / 0.59
    assert posterior_known(PARAMS, np.array(0.3), np.array(0)) == pytest.approx(0.03 / 0.59)


def _simulate_bkt(params: BKTParams, n_learners: int, length: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_learners):
        known = rng.random() < params.p_init
        for t in range(length):
            p = 1 - params.p_slip if known else params.p_guess
            rows.append((f"u{u}", "s", float(t), int(rng.random() < p)))
            known = known or rng.random() < params.p_learn
    return pd.DataFrame(rows, columns=["user_id", "skill_id", "t", "correct"])


def test_em_never_lowers_likelihood_and_recovers_parameters():
    true = BKTParams(p_init=0.25, p_learn=0.15, p_guess=0.2, p_slip=0.08)
    seq = to_sequences(_simulate_bkt(true, n_learners=400, length=20, seed=0))
    start = BKTParams(0.5, 0.3, 0.3, 0.3)
    fitted, history = fit_em(seq, init=start, max_iter=200, tol=1e-9)
    assert np.all(np.diff(history) >= -1e-8), "EM must never decrease the likelihood"
    assert history[-1] > log_likelihood(start, seq)
    assert fitted.p_learn == pytest.approx(true.p_learn, abs=0.04)
    assert fitted.p_guess == pytest.approx(true.p_guess, abs=0.05)
    assert fitted.p_slip == pytest.approx(true.p_slip, abs=0.04)


def test_prediction_only_uses_earlier_answers_and_ignores_row_order():
    frame = _simulate_bkt(PARAMS, n_learners=5, length=8, seed=1)
    model = BKT().fit(frame)
    p = model.predict(frame)
    # Changing each learner's last answer must not move any prediction (a row's own answer
    # and later answers are unknown when it is predicted).
    last = frame.groupby("user_id")["t"].transform("max") == frame["t"]
    flipped = frame.assign(correct=np.where(last, 1 - frame["correct"], frame["correct"]))
    np.testing.assert_allclose(model.predict(flipped), p)
    # The first answer of every learner is predicted from p_init alone.
    first = frame["t"] == 0
    expected = predict_correct(model.params["s"], np.array(model.params["s"].p_init))
    np.testing.assert_allclose(p[first], expected)
    # Shuffled rows give the same prediction per row (sequences are rebuilt in time order).
    shuffled = frame.sample(frac=1.0, random_state=0)
    np.testing.assert_allclose(model.predict(shuffled), p[shuffled.index])
