"""Versioned records that cross the rollout, reward, and learner boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


SCHEMA_VERSION = 1


class SchemaError(ValueError):
    """Raised when a record violates a data-plane invariant."""


@dataclass(frozen=True)
class Prompt:
    run_id: str
    prompt_id: str
    group_id: str
    text: str
    input_ids: list[int]
    seed: int
    policy_version: int
    dataset: str = "default"
    schema_version: ClassVar[int] = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.run_id or not self.prompt_id or not self.group_id:
            raise SchemaError("prompt identifiers must be non-empty")
        if not self.input_ids:
            raise SchemaError("input_ids must not be empty")
        if self.policy_version < 0:
            raise SchemaError("policy_version must be non-negative")


@dataclass(frozen=True)
class Trajectory:
    run_id: str
    trajectory_id: str
    prompt_id: str
    group_id: str
    prompt_ids: list[int]
    response_ids: list[int]
    rollout_logprobs: list[float]
    finish_reason: str
    policy_version: int
    generated_tokens: int
    wall_time_ms: float
    interrupted: bool = False
    routing_replay: list[int] | None = None
    schema_version: ClassVar[int] = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.trajectory_id or not self.prompt_id:
            raise SchemaError("trajectory identifiers must be non-empty")
        if len(self.response_ids) != len(self.rollout_logprobs):
            raise SchemaError("response_ids and rollout_logprobs must be aligned")
        if self.generated_tokens != len(self.response_ids):
            raise SchemaError("generated_tokens must equal response length")
        if self.policy_version < 0:
            raise SchemaError("policy_version must be non-negative")


@dataclass(frozen=True)
class Reward:
    run_id: str
    trajectory_id: str
    score: float
    components: dict[str, float] = field(default_factory=dict)
    verifier_version: str = "local-v1"
    schema_version: ClassVar[int] = SCHEMA_VERSION


@dataclass(frozen=True)
class Experience:
    run_id: str
    trajectory_id: str
    sequence: list[int]
    action_mask: list[int]
    old_logprobs: list[float]
    reward: float
    advantage: float
    policy_version: int
    stale_tokens: int = 0
    schema_version: ClassVar[int] = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not (len(self.sequence) == len(self.action_mask) == len(self.old_logprobs)):
            raise SchemaError("sequence, action_mask, and old_logprobs must be aligned")
        if self.stale_tokens < 0 or self.stale_tokens > sum(self.action_mask):
            raise SchemaError("stale_tokens must be within the action-token count")


@dataclass(frozen=True)
class CheckpointManifest:
    run_id: str
    checkpoint_id: str
    policy_version: int
    path: str
    trainer_step: int
    consumed_trajectory_count: int
    parent_checkpoint: str | None = None
    schema_version: ClassVar[int] = SCHEMA_VERSION


def encode(record: Any) -> dict[str, Any]:
    """Serialize a record without leaking ClassVar fields into JSON."""
    payload = asdict(record)
    payload["record_type"] = type(record).__name__
    payload["schema_version"] = SCHEMA_VERSION
    return payload

