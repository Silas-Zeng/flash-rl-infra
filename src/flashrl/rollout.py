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
            "sampling_params": {"max_new_tokens": 128, "temperature": 1.0, "seed": prompt.seed},
            "return_logprob": True,
            "metadata": {"run_id": prompt.run_id, "prompt_id": prompt.prompt_id, "group_id": prompt.group_id},
        }
        response = httpx.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        meta = body.get("meta_info", {})
        pairs = meta.get("output_token_logprobs") or []
        if not isinstance(pairs, list) or any(not isinstance(pair, (list, tuple)) or len(pair) != 2 for pair in pairs):
            raise RuntimeError("SGLang response contains malformed output_token_logprobs")
        response_ids = [int(pair[1]) for pair in pairs]
        logprobs = [float(pair[0]) for pair in pairs]
        if not response_ids:
            raise RuntimeError("SGLang response did not expose output_token_logprobs; text-only data is unsafe for RL")
        completion_tokens = meta.get("completion_tokens")
        if completion_tokens is not None and int(completion_tokens) != len(response_ids):
            raise RuntimeError("SGLang completion token count does not match output_token_logprobs")
        finish_reason = meta.get("finish_reason", "length")
        if isinstance(finish_reason, dict):
            finish_reason = finish_reason.get("type", "length")
        raw_version = meta.get("weight_version")
        if raw_version is None:
            policy_version = prompt.policy_version
        else:
            try:
                policy_version = int(raw_version)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "SGLang weight_version must be a numeric policy step; "
                    "pass a monotonic training version when publishing weights"
                ) from exc
        trajectory = Trajectory(
            run_id=prompt.run_id,
            trajectory_id=trajectory_id,
            prompt_id=prompt.prompt_id,
            group_id=prompt.group_id,
            prompt_ids=prompt.input_ids,
            response_ids=response_ids,
            rollout_logprobs=logprobs,
            finish_reason=str(finish_reason),
            policy_version=policy_version,
            generated_tokens=len(response_ids),
            wall_time_ms=(time.perf_counter() - started) * 1000,
        )
        return RolloutResult(trajectory)

