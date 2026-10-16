import pytest

pytest.importorskip("torch")

from ktrace.benchmark import run_benchmark  # noqa: E402
from ktrace.models.baselines import ItemMean  # noqa: E402
from ktrace.models.dkt import DKT  # noqa: E402


def test_dkt_beats_item_mean(log):
    bench = run_benchmark(log, seed=0, models=[ItemMean(), DKT(epochs=5)])
    m = bench.metrics
    assert m.loc[DKT.name, "auc"] > m.loc[ItemMean.name, "auc"]
