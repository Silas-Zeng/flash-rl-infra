"""Rollout backends: local simulator plus an optional SGLang HTTP adapter."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .policy import ToyPolicy
from .schemas import Prompt, Trajectory


@dataclass
class RolloutResult:
    trajectory: Trajectory
    next_token_index: int = 0


class LocalRolloutBackend:
    def __init__(self, policy: ToyPolicy, max_tokens: int = 4):
        self.policy = policy
        self.max_tokens = max_tokens

    def generate(self, prompt: Prompt, trajectory_id: str, interrupted_after: int | None = None) -> RolloutResult:
        started = time.perf_counter()
        response, logprobs = self.policy.generate(prompt.seed, self.max_tokens)
        interrupted = interrupted_after is not None and interrupted_after < len(response)
        if interrupted:
            response = response[:interrupted_after]
            logprobs = logprobs[:interrupted_after]
        trajectory = Trajectory(
            run_id=prompt.run_id,
            trajectory_id=trajectory_id,
            prompt_id=prompt.prompt_id,
            group_id=prompt.group_id,
            prompt_ids=prompt.input_ids,
            response_ids=response,
            rollout_logprobs=logprobs,
            finish_reason="interrupted" if interrupted else "length",
            policy_version=self.policy.version,
            generated_tokens=len(response),
            wall_time_ms=(time.perf_counter() - started) * 1000,
            interrupted=interrupted,
        )
        return RolloutResult(trajectory, len(response))


class SGLangRolloutBackend:
    """Small adapter for a running SGLang OpenAI-compatible endpoint.

    It deliberately keeps transport details out of the core schema. The exact
    response shape can vary by SGLang version, so parsing is isolated here.
    """

    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: Prompt, trajectory_id: str) -> RolloutResult:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Install flash-rl-infra[http] to use SGLangRolloutBackend") from exc
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "input_ids": prompt.input_ids,
            "max_tokens": 128,
            "temperature": 1.0,
            "seed": prompt.seed,
            "return_logprob": True,
            "metadata": {"run_id": prompt.run_id, "prompt_id": prompt.prompt_id, "group_id": prompt.group_id},
        }
        response = httpx.post(f"{self.base_url}/v1/completions", json=payload, timeout=self.timeout)
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        choice = body["choices"][0]
        meta = body.get("meta_info", {})
        pairs = meta.get("output_token_logprobs") or []
        response_ids = [int(pair[1]) for pair in pairs]
        logprobs = [float(pair[0]) for pair in pairs]
        if not response_ids:
            text = choice.get("text", "")
            raise RuntimeError("SGLang response did not expose output_token_logprobs; text-only data is unsafe for RL")
        policy_version = int(meta.get("weight_version", prompt.policy_version))
        trajectory = Trajectory(
            run_id=prompt.run_id,
            trajectory_id=trajectory_id,
            prompt_id=prompt.prompt_id,
            group_id=prompt.group_id,
            prompt_ids=prompt.input_ids,
            response_ids=response_ids,
            rollout_logprobs=logprobs,
            finish_reason=choice.get("finish_reason", "stop"),
            policy_version=policy_version,
            generated_tokens=len(response_ids),
            wall_time_ms=(time.perf_counter() - started) * 1000,
        )
        return RolloutResult(trajectory)

