"""FastAPI service: recall predictions and today's review list for one learner.

The client sends the learner's answer history with every request, so the service is stateless:
no database, easy to scale and to test. The same `LearnerHistory` features and the same
`select_reviews` rule as the offline simulation are used, so the API cannot drift from the
numbers in docs/results.md.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from .predictor import ModelBundle
from .scheduler import select_reviews

SECONDS_PER_DAY = 86_400.0


class Event(BaseModel):
    item_id: str
    skill_id: str
    timestamp: datetime
    correct: bool


class Query(BaseModel):
    item_id: str
    skill_id: str
    timestamp: datetime


class PredictRequest(BaseModel):
    history: list[Event] = Field(default_factory=list)
    queries: list[Query] = Field(min_length=1)


class Prediction(BaseModel):
    item_id: str
    timestamp: datetime
    predicted_recall: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]


class NextReviewsRequest(BaseModel):
    history: list[Event] = Field(default_factory=list)
    now: datetime
    budget: int = Field(20, ge=0, le=500)
    target: float = Field(0.85, gt=0.0, lt=1.0)


class ReviewItem(BaseModel):
    item_id: str
    skill_id: str
    predicted_recall: float


class NextReviewsResponse(BaseModel):
    now: datetime
    candidates: int  # words the learner has seen before `now`
    items: list[ReviewItem]  # what to review today, most urgent first


def to_days(ts: datetime) -> float:
    """Days since the Unix epoch. Timestamps without a time zone are read as UTC."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.timestamp() / SECONDS_PER_DAY


def _sorted(history: list[Event]) -> list[Event]:
    return sorted(history, key=lambda e: to_days(e.timestamp))


def predict(bundle: ModelBundle, request: PredictRequest) -> PredictResponse:
    """Each query only uses the events strictly before its own timestamp."""
    events = _sorted(request.history)
    order = sorted(range(len(request.queries)), key=lambda k: to_days(request.queries[k].timestamp))
    learner = bundle.learner()
    results: dict[int, float] = {}
    pos = 0
    for k in order:
        q = request.queries[k]
        t = to_days(q.timestamp)
        while pos < len(events) and to_days(events[pos].timestamp) < t:
            e = events[pos]
            learner.observe(e.item_id, e.skill_id, to_days(e.timestamp), int(e.correct))
            pos += 1
        results[k] = float(learner.predict([q.item_id], [q.skill_id], t)[0])
    return PredictResponse(
        predictions=[
            Prediction(item_id=q.item_id, timestamp=q.timestamp, predicted_recall=results[k])
            for k, q in enumerate(request.queries)
        ]
    )


def next_reviews(bundle: ModelBundle, request: NextReviewsRequest) -> NextReviewsResponse:
    now = to_days(request.now)
    learner = bundle.learner()
    skills: dict[str, str] = {}
    for e in _sorted(request.history):
        t = to_days(e.timestamp)
        if t >= now:
            break
        learner.observe(e.item_id, e.skill_id, t, int(e.correct))
        skills[e.item_id] = e.skill_id
    items = list(skills)
    p = learner.predict(items, [skills[i] for i in items], now)
    chosen = select_reviews(p, request.budget, request.target) if items else np.empty(0, int)
    return NextReviewsResponse(
        now=request.now,
        candidates=len(items),
        items=[
            ReviewItem(item_id=items[k], skill_id=skills[items[k]], predicted_recall=float(p[k]))
            for k in chosen
        ],
    )


def create_app(bundle: ModelBundle) -> FastAPI:
    app = FastAPI(title="ktrace", version="0.1.0")
    app.state.bundle = bundle

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict_endpoint(body: PredictRequest, request: Request) -> PredictResponse:
        return predict(request.app.state.bundle, body)

    @app.post("/next-reviews", response_model=NextReviewsResponse)
    def next_reviews_endpoint(body: NextReviewsRequest, request: Request) -> NextReviewsResponse:
        return next_reviews(request.app.state.bundle, body)

    return app
