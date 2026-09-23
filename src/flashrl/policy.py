"""A tiny deterministic policy used to run the complete flow without a model."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path


class ToyPolicy:
    """Tabular policy: enough to exercise logprobs, updates, and checkpoints."""

    def __init__(self, vocabulary: list[int], version: int = 0, logits: dict[str, float] | None = None):
        self.vocabulary = vocabulary
        self.version = version
        self.logits = logits or {str(token): 0.0 for token in vocabulary}

    def _probs(self) -> dict[int, float]:
        values = {token: math.exp(self.logits[str(token)]) for token in self.vocabulary}
        total = sum(values.values())
        return {token: value / total for token, value in values.items()}

    def generate(self, seed: int, max_tokens: int = 4, stop_token: int = 0, start: list[int] | None = None):
        rng = random.Random(seed + self.version * 100003)
        probs = self._probs()
        response: list[int] = []
        logprobs: list[float] = []
        for _ in range(max_tokens):
            draw = rng.random()
            cumulative = 0.0
            token = self.vocabulary[-1]
            for candidate, probability in probs.items():
                cumulative += probability
                if draw <= cumulative:
                    token = candidate
                    break
            response.append(token)
            logprobs.append(math.log(max(probs[token], 1e-12)))
            if token == stop_token:
                break
        return response, logprobs

    def update(self, token: int, advantage: float, learning_rate: float = 0.05) -> None:
        """REINFORCE-style update with a stable mean-logit baseline."""
        for candidate in self.vocabulary:
            direction = 1.0 if candidate == token else -1.0 / max(len(self.vocabulary) - 1, 1)
            self.logits[str(candidate)] += learning_rate * advantage * direction
        self.version += 1

    def save(self, path: str | Path) -> None:
        payload = {"version": self.version, "vocabulary": self.vocabulary, "logits": self.logits}
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ToyPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["vocabulary"], payload["version"], payload["logits"])

