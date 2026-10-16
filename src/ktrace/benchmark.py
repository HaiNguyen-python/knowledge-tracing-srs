"""Fit every model on training learners and score it on held-out learners' later answers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data.split import Split, split_by_learner_and_time
from .features import build_features, estimate_difficulty
from .metrics import evaluate, per_group_auc
from .models import RecallModel
from .models.baselines import GlobalMean, ItemMean
from .models.bkt import BKT
from .models.hlr import HLR
from .models.pfa import PFA
from .predictor import ModelBundle

CEILING = "True recall (simulator)"


def default_models(include_deep: bool = False) -> list[RecallModel]:
    models: list[RecallModel] = [
        GlobalMean(),
        ItemMean(),
        BKT(),
        BKT(kc="item_id", name="BKT (per word)"),
        PFA(),
        HLR(),
    ]
    if include_deep:
        from .models.dkt import DKT

        models.append(DKT())
    return models


@dataclass
class Benchmark:
    split: Split
    scored: pd.DataFrame  # scored test rows with one prediction column per model
    metrics: pd.DataFrame  # one row per model
    per_skill: pd.DataFrame  # AUC per IELTS topic and model
    model_names: list[str]
    bundle: ModelBundle  # the fitted HLR, ready for the scheduler and the API


def _metrics_row(scored: pd.DataFrame, column: str) -> dict[str, float]:
    y, p = scored["correct"].to_numpy(), scored[column].to_numpy()
    review = (scored["first_seen"] == 0).to_numpy()
    row = evaluate(y, p)
    reviews_only = evaluate(y[review], p[review])
    row["auc_reviews"] = reviews_only["auc"]
    row["log_loss_reviews"] = reviews_only["log_loss"]
    return row


def run_benchmark(
    log: pd.DataFrame, seed: int = 0, models: list[RecallModel] | None = None
) -> Benchmark:
    """Split by learner and time, build history features, fit, predict, score.

    Item difficulty is estimated from the training rows only; the test learners' features use
    that same map, so nothing about test answers reaches the features."""
    split = split_by_learner_and_time(log, seed=seed)
    difficulty, default = estimate_difficulty(split.train)
    train = build_features(split.train, difficulty, default)
    test = build_features(split.test, difficulty, default)
    mask = (test["t"] >= split.cutoff).to_numpy()
    scored = test[mask].copy()
    models = models if models is not None else default_models()
    hlr: HLR | None = None
    for model in models:
        model.fit(train)
        scored[model.name] = model.predict(test)[mask]
        if isinstance(model, HLR):
            hlr = model
    if hlr is None:
        hlr = HLR().fit(train)
    names = [m.name for m in models]
    columns = names + ([CEILING] if "p_true" in scored else [])
    if "p_true" in scored:
        scored[CEILING] = scored["p_true"]
    metrics = pd.DataFrame({c: _metrics_row(scored, c) for c in columns}).T
    return Benchmark(
        split=split,
        scored=scored.reset_index(drop=True),
        metrics=metrics,
        per_skill=per_group_auc(scored, columns),
        model_names=names,
        bundle=ModelBundle(hlr, difficulty, default),
    )


def beats_baselines(bench: Benchmark, model: str) -> bool:
    """A model counts only if it beats both baselines on AUC and on log loss."""
    m = bench.metrics
    base = [GlobalMean.name, ItemMean.name]
    return bool(
        np.all(m.loc[model, "auc"] > m.loc[base, "auc"])
        and np.all(m.loc[model, "log_loss"] < m.loc[base, "log_loss"])
    )
