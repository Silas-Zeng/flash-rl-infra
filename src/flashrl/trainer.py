"""Experience construction and a tiny learner for the local vertical slice."""

from __future__ import annotations

import json
from pathlib import Path

from .policy import ToyPolicy
from .schemas import Experience, Reward, Trajectory


def build_experience(trajectory: Trajectory, reward: Reward, current_policy_version: int) -> Experience:
    stale = max(0, current_policy_version - trajectory.policy_version)
    # Stale response tokens are excluded by the same mask used by the learner.
    action_mask = [1] * len(trajectory.response_ids)
    stale_tokens = min(stale, len(action_mask))
    for index in range(stale_tokens):
        action_mask[index] = 0
    advantage = reward.score
    return Experience(
        run_id=trajectory.run_id,
        trajectory_id=trajectory.trajectory_id,
        sequence=trajectory.prompt_ids + trajectory.response_ids,
        action_mask=[0] * len(trajectory.prompt_ids) + action_mask,
        old_logprobs=[0.0] * len(trajectory.prompt_ids) + trajectory.rollout_logprobs,
        reward=reward.score,
        advantage=advantage,
        policy_version=trajectory.policy_version,
        stale_tokens=stale_tokens,
    )


class ToyLearner:
    def __init__(self, policy: ToyPolicy, checkpoint_dir: str | Path, learning_rate: float = 0.05):
        self.policy = policy
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.learning_rate = learning_rate
        self.step = 0

    def train(self, experiences: list[Experience]) -> Path:
        for experience in experiences:
            for token, mask in zip(experience.sequence, experience.action_mask):
                if mask:
                    self.policy.update(token, experience.advantage, self.learning_rate)
        self.step += 1
        path = self.checkpoint_dir / f"step_{self.step:04d}.json"
        self.policy.save(path)
        manifest = {
            "checkpoint_id": path.stem,
            "policy_version": self.policy.version,
            "trainer_step": self.step,
            "experience_count": len(experiences),
            "path": str(path),
        }
        path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path

