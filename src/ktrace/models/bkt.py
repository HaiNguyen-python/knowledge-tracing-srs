"""Bayesian Knowledge Tracing (Corbett & Anderson, 1995), fitted with EM.

BKT is a two-state hidden Markov model. For each learner and skill the hidden state is
"unknown" (0) or "known" (1). Four parameters per skill:

    p_init   P(known before the first answer)
    p_learn  P(unknown -> known after an answer)   (classic BKT never forgets)
    p_guess  P(correct | unknown)
    p_slip   P(wrong | known)

Prediction before an answer:  P(correct) = P(known) (1 - slip) + (1 - P(known)) guess.
After seeing the answer, Bayes' rule gives the posterior P(known | answer), then the learning
step: P(known next) = posterior + (1 - posterior) p_learn.

EM (Baum-Welch): the E-step runs the scaled forward-backward algorithm to get, for every answer,
P(known) given *all* answers in the sequence, and the expected number of unknown -> known
transitions. The M-step sets each parameter to its expected frequency. Each iteration cannot
lower the likelihood, which `tests/test_bkt.py` checks. See docs/how-it-works.md for the maths.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd

EPS = 1e-6


@dataclass(frozen=True)
class BKTParams:
    p_init: float = 0.3
    p_learn: float = 0.1
    p_guess: float = 0.2
    p_slip: float = 0.1


def predict_correct(params: BKTParams, p_known: np.ndarray) -> np.ndarray:
    return p_known * (1.0 - params.p_slip) + (1.0 - p_known) * params.p_guess


def posterior_known(params: BKTParams, p_known: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """Bayes' rule: P(known | this answer)."""
    like_known = np.where(correct == 1, 1.0 - params.p_slip, params.p_slip)
    like_unknown = np.where(correct == 1, params.p_guess, 1.0 - params.p_guess)
    num = p_known * like_known
    return num / (num + (1.0 - p_known) * like_unknown)


def update_known(params: BKTParams, p_known: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """Posterior, then the chance of learning: P(known) before the next answer."""
    post = posterior_known(params, p_known, correct)
    return post + (1.0 - post) * params.p_learn


@dataclass(frozen=True)
class Sequences:
    """Answers of many learners as a padded matrix (one row per learner, time along columns)."""

    obs: np.ndarray  # (n, T) 0/1, 0 where padded
    mask: np.ndarray  # (n, T) True for real answers
    rows: np.ndarray  # (n, T) position of the answer in the input frame, -1 where padded


def to_sequences(frame: pd.DataFrame) -> Sequences:
    """Group a frame by learner, each learner's answers in time order (stable on ties)."""
    codes, _ = pd.factorize(frame["user_id"])
    order = np.lexsort((frame["t"].to_numpy(), codes))
    counts = np.bincount(codes)
    n, width = len(counts), int(np.max(counts)) if len(counts) else 0
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    sorted_codes = codes[order]
    col = np.arange(len(order)) - starts[sorted_codes]
    rows = np.full((n, width), -1)
    rows[sorted_codes, col] = order
    mask = rows >= 0
    obs = np.where(mask, frame["correct"].to_numpy()[np.maximum(rows, 0)], 0).astype(float)
    return Sequences(obs=obs, mask=mask, rows=rows)


def _emissions(params: BKTParams, seq: Sequences) -> np.ndarray:
    """P(answer | state) as (n, T, 2); 1 for padded cells so they carry no information."""
    unknown = np.where(seq.obs == 1, params.p_guess, 1.0 - params.p_guess)
    known = np.where(seq.obs == 1, 1.0 - params.p_slip, params.p_slip)
    e = np.stack([unknown, known], axis=-1)
    e[~seq.mask] = 1.0
    return e


def _forward(params: BKTParams, e: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scaled forward pass. Returns (alpha, scale, prior_known).

    alpha[:, t] = P(state_t | answers up to t); scale[:, t] = P(answer t | answers before t);
    prior_known[:, t] = P(known_t | answers before t), the quantity used for prediction."""
    n, width, _ = e.shape
    alpha = np.zeros_like(e)
    scale = np.ones((n, width))
    prior = np.zeros((n, width))
    belief = np.full(n, params.p_init)
    for t in range(width):
        prior[:, t] = belief
        joint = np.stack([1.0 - belief, belief], axis=-1) * e[:, t]
        scale[:, t] = joint.sum(axis=1)
        alpha[:, t] = joint / scale[:, t, None]
        known = alpha[:, t, 1]
        belief = known + (1.0 - known) * params.p_learn
    return alpha, scale, prior


def log_likelihood(params: BKTParams, seq: Sequences) -> float:
    _, scale, _ = _forward(params, _emissions(params, seq))
    return float(np.log(scale[seq.mask]).sum())


def em_step(params: BKTParams, seq: Sequences) -> tuple[BKTParams, float]:
    """One EM iteration. Returns new parameters and the log-likelihood of the *old* ones."""
    e = _emissions(params, seq)
    alpha, scale, _ = _forward(params, e)
    n, width, _ = e.shape
    learn = params.p_learn
    trans = np.array([[1.0 - learn, learn], [0.0, 1.0]])
    beta = np.ones_like(e)
    for t in range(width - 2, -1, -1):
        nxt = e[:, t + 1] * beta[:, t + 1] / scale[:, t + 1, None]
        beta[:, t] = nxt @ trans.T
    gamma = alpha * beta  # P(state_t | all answers); rows sum to 1 thanks to the scaling
    # Expected unknown -> known transitions between t and t + 1 (only where t + 1 is real).
    xi_learn = (
        alpha[:, :-1, 0] * learn * e[:, 1:, 1] * beta[:, 1:, 1] / np.maximum(scale[:, 1:], EPS)
    )
    has_next = seq.mask[:, 1:]
    m = seq.mask
    g0, g1 = gamma[..., 0] * m, gamma[..., 1] * m
    new = BKTParams(
        p_init=float(gamma[:, 0, 1].mean()),
        p_learn=float((xi_learn * has_next).sum() / max((g0[:, :-1] * has_next).sum(), EPS)),
        p_guess=float((g0 * seq.obs).sum() / max(g0.sum(), EPS)),
        p_slip=float((g1 * (1.0 - seq.obs)).sum() / max(g1.sum(), EPS)),
    )
    return _clip(new), float(np.log(scale[m]).sum())


def _clip(p: BKTParams) -> BKTParams:
    """Keep guess and slip below 0.5. Above that, "known" would mean "more likely wrong" and the
    two states swap meaning (a known identifiability problem of BKT)."""
    return BKTParams(
        p_init=float(np.clip(p.p_init, EPS, 1 - EPS)),
        p_learn=float(np.clip(p.p_learn, EPS, 1 - EPS)),
        p_guess=float(np.clip(p.p_guess, EPS, 0.5)),
        p_slip=float(np.clip(p.p_slip, EPS, 0.5)),
    )


def fit_em(
    seq: Sequences, init: BKTParams | None = None, max_iter: int = 100, tol: float = 1e-5
) -> tuple[BKTParams, list[float]]:
    """Run EM until the log-likelihood improves by less than `tol`. Returns params and history."""
    params = init or BKTParams()
    history: list[float] = []
    for _ in range(max_iter):
        new, ll = em_step(params, seq)
        history.append(ll)
        params = new
        if len(history) > 1 and history[-1] - history[-2] < tol:
            break
    history.append(log_likelihood(params, seq))
    return params, history


@dataclass
class BKT:
    """One BKT per knowledge component (KC). KCs unseen in training get the average parameters.

    `kc="skill_id"` is classic BKT: one model per IELTS topic. `kc="item_id"` treats every
    word as its own KC, which suits vocabulary better: knowing "pollution" says little about
    knowing "deforestation"."""

    kc: str = "skill_id"
    name: str = "BKT (per skill)"
    max_iter: int = 100
    params: dict[str, BKTParams] = field(default_factory=dict)
    default: BKTParams = field(default_factory=BKTParams)

    def fit(self, frame: pd.DataFrame) -> BKT:
        for kc, group in frame.groupby(self.kc):
            self.params[str(kc)], _ = fit_em(to_sequences(group), max_iter=self.max_iter)
        fitted = list(self.params.values())
        if fitted:
            names = [f.name for f in fields(BKTParams)]
            self.default = BKTParams(
                *(float(np.mean([getattr(p, n) for p in fitted])) for n in names)
            )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """P(correct) for each row from the learner's earlier answers on the same KC."""
        out = np.empty(len(frame))
        for kc, positions in frame.groupby(self.kc).indices.items():
            group = frame.iloc[positions]
            params = self.params.get(str(kc), self.default)
            seq = to_sequences(group)
            _, _, prior = _forward(params, _emissions(params, seq))
            pred = predict_correct(params, prior)
            out[positions[seq.rows[seq.mask]]] = pred[seq.mask]
        return out
