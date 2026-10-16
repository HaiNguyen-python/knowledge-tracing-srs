import pytest

from ktrace.benchmark import Benchmark, run_benchmark
from ktrace.simulator import Catalog, generate_log, make_catalog


@pytest.fixture(scope="session")
def catalog() -> Catalog:
    return make_catalog()


@pytest.fixture(scope="session")
def log(catalog):
    """160 learners x 24 days: small enough for CI, big enough for stable metrics."""
    return generate_log(catalog, n_learners=160, n_days=24, seed=3)


@pytest.fixture(scope="session")
def bench(log) -> Benchmark:
    return run_benchmark(log, seed=0)
