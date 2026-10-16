import pytest
from fastapi.testclient import TestClient

from ktrace.api import create_app

T0 = "2026-03-01T18:00:00Z"


@pytest.fixture(scope="module")
def client(bench):
    return TestClient(create_app(bench.bundle))


def _event(item, day, correct, skill="environment"):
    return {
        "item_id": item,
        "skill_id": skill,
        "timestamp": f"2026-03-{day:02d}T18:00:00Z",
        "correct": correct,
    }


HISTORY = [
    _event("envi_000", 1, True),
    _event("envi_001", 1, False),
    _event("envi_000", 2, True),
    _event("envi_002", 2, True),
    _event("heal_003", 3, False, "health"),
]


def test_predict_contract_and_recall_falls_with_time(client):
    queries = [
        {
            "item_id": "envi_000",
            "skill_id": "environment",
            "timestamp": f"2026-03-{d:02d}T18:00:00Z",
        }
        for d in (5, 10, 20)
    ]
    r = client.post("/predict", json={"history": HISTORY, "queries": queries})
    assert r.status_code == 200
    p = [x["predicted_recall"] for x in r.json()["predictions"]]
    assert len(p) == 3 and all(0 < v < 1 for v in p)
    assert p[0] > p[1] > p[2]


def test_predict_ignores_events_after_the_query_time(client):
    query = [{"item_id": "envi_001", "skill_id": "environment", "timestamp": "2026-03-04T12:00Z"}]
    base = client.post("/predict", json={"history": HISTORY, "queries": query}).json()
    future = [*HISTORY, _event("envi_001", 4, True), _event("envi_001", 9, True)]
    later = client.post("/predict", json={"history": future, "queries": query}).json()
    assert base == later


def test_next_reviews_respects_budget_target_and_order(client):
    body = {"history": HISTORY, "now": "2026-03-12T18:00:00Z", "budget": 2, "target": 0.85}
    r = client.post("/next-reviews", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["candidates"] == 4
    items = data["items"]
    assert len(items) <= 2
    p = [x["predicted_recall"] for x in items]
    assert all(v < 0.85 for v in p) and p == sorted(p, reverse=True)


def test_next_reviews_only_uses_history_before_now(client):
    body = {"history": HISTORY, "now": "2026-03-02T12:00:00Z", "budget": 10}
    data = client.post("/next-reviews", json=body).json()
    assert data["candidates"] == 2  # only the two words seen on 1 March


@pytest.mark.parametrize(
    "body",
    [
        {"history": [], "now": T0, "target": 1.5},
        {"history": [], "now": T0, "budget": -1},
        {"history": [{"item_id": "x"}], "now": T0},
    ],
)
def test_invalid_requests_are_rejected(client, body):
    assert client.post("/next-reviews", json=body).status_code == 422


def test_empty_history_gives_no_reviews(client):
    data = client.post("/next-reviews", json={"now": T0}).json()
    assert data == {"now": "2026-03-01T18:00:00Z", "candidates": 0, "items": []}
