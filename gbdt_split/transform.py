import json
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


@dataclass
class TargetTransformState:
    mean: Optional[float] = None
    std: Optional[float] = None


class TargetTransformer:
    def __init__(self):
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None

    def fit(self, y: np.ndarray) -> "TargetTransformer":
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        self.mean_ = float(y.mean())
        self.std_ = float(y.std())
        if self.std_ < 1e-8:
            self.std_ = 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        transformed = (y - self.mean_) / (self.std_ + 1e-8)
        return np.clip(transformed, -5.0, 5.0).astype(np.float32)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        return (y * self.std_ + self.mean_).astype(np.float32)

    def to_state(self) -> TargetTransformState:
        return TargetTransformState(mean=self.mean_, std=self.std_)

    @classmethod
    def from_state(cls, state: TargetTransformState) -> "TargetTransformer":
        transformer = cls()
        transformer.mean_ = state.mean
        transformer.std_ = state.std
        return transformer


def save_transformer(transformer: TargetTransformer, path: str) -> None:
    state = transformer.to_state()
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(asdict(state), fp)


def load_transformer(path: str) -> TargetTransformer:
    with open(path, "r", encoding="utf-8") as fp:
        state = TargetTransformState(**json.load(fp))
    return TargetTransformer.from_state(state)
