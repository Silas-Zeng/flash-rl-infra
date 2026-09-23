"""A bounded, sample-level asynchronous training loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .policy import ToyPolicy
from .reward import ArithmeticReward
from .rollout import LocalRolloutBackend
from .schemas import Prompt
from .store import EventStore
from .trainer import ToyLearner, build_experience


class FlashRLRun:
    def __init__(self, root: str | Path, groups: int = 4, group_size: int = 2):
        self.root = Path(root)
        self.store = EventStore(self.root / "events")
        self.groups = groups
        self.group_size = group_size
        self.policy = ToyPolicy([0, 1, 2, 3, 4, 5])
        self.rollout = LocalRolloutBackend(self.policy, max_tokens=4)
        self.reward = ArithmeticReward()
        self.learner = ToyLearner(self.policy, self.root / "checkpoints")

    def run(self) -> dict:
        trajectories = []
        rewards = []
        experiences = []
        for group_index in range(self.groups):
            group_id = f"group-{group_index:04d}"
            for sample_index in range(self.group_size):
                prompt = Prompt(
                    run_id="local-demo",
                    prompt_id=f"prompt-{group_index:04d}-{sample_index:02d}",
                    group_id=group_id,
                    text=f"solve toy task {group_index}",
                    input_ids=[group_index % 6],
                    seed=1000 + group_index * 10 + sample_index,
                    policy_version=self.policy.version,
                )
                trajectory_id = f"traj-{group_index:04d}-{sample_index:02d}"
                trajectory = self.rollout.generate(prompt, trajectory_id).trajectory
                self.store.put(trajectory_id, "rollout", trajectory)
                trajectories.append(trajectory)
                reward = self.reward.score(trajectory, expected_token=group_index % 6)
                self.store.put(f"reward:{trajectory_id}", "reward", reward)
                rewards.append(reward)
                experience = build_experience(trajectory, reward, self.policy.version)
                self.store.put(f"experience:{trajectory_id}", "experience", experience)
                experiences.append(experience)
        checkpoint = self.learner.train(experiences)
        result = {
            "run_id": "local-demo",
            "groups": self.groups,
            "group_size": self.group_size,
            "trajectory_count": len(trajectories),
            "reward_mean": sum(item.score for item in rewards) / max(len(rewards), 1),
            "checkpoint": str(checkpoint),
            "policy_version": self.policy.version,
            "store_counts": {
                "rollout": self.store.count("rollout"),
                "reward": self.store.count("reward"),
                "experience": self.store.count("experience"),
            },
        }
        (self.root / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def close(self) -> None:
        self.store.close()

