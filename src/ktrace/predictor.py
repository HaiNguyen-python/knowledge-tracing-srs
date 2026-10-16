"""A fitted recall model packaged with what it needs at serving time.

`ModelBundle` = HLR weights + the item difficulty map from training. It is saved as one small JSON
file, so the API needs no pickle and the artifact is readable in a code review.
`LearnerPredictor` replays one learner's history through the same `LearnerHistory` used in
training, then asks the model for P(recall) of any word at any time.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import LearnerHistory
from .models.hlr import HLR


@dataclass
class ModelBundle:
    model: HLR
    difficulty: dict[str, float]
    default_difficulty: float

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.to_dict(),
            "difficulty": self.difficulty,
            "default_difficulty": self.default_difficulty,
        }
        path.write_text(json.dumps(payload, indent=1))

    @classmethod
    def load(cls, path: Path) -> ModelBundle:
        data = json.loads(path.read_text())
        return cls(
            model=HLR.from_dict(data["model"]),
            difficulty={str(k): float(v) for k, v in data["difficulty"].items()},
            default_difficulty=float(data["default_difficulty"]),
        )

    def learner(self) -> LearnerPredictor:
        return LearnerPredictor(self)


class LearnerPredictor:
    """Online predictions for one learner: call `observe` after each answer, `predict` any time."""

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        self.history = LearnerHistory(bundle.difficulty, bundle.default_difficulty)

    def observe(self, item: str, skill: str, t: float, correct: int) -> None:
        self.history.update(item, skill, t, correct)

    def predict(self, items: Sequence[str], skills: Sequence[str], t: float) -> np.ndarray:
        if not items:
            return np.empty(0)
        x = np.array([self.history.features(i, s, t) for i, s in zip(items, skills, strict=True)])
        return self.bundle.model.predict_matrix(x)
