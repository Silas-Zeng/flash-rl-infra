"""Reward/verifier adapters."""

from __future__ import annotations

from .schemas import Reward, Trajectory


class ArithmeticReward:
    """Toy verifier used for a reproducible local run.

    A response receives 1 when it contains the expected answer token, and a
    small length penalty otherwise. Replace this with a sandbox/verifier API in
    a real RLVR run.
    """

    def score(self, trajectory: Trajectory, expected_token: int) -> Reward:
        hit = float(expected_token in trajectory.response_ids)
        length_penalty = 0.01 * max(trajectory.generated_tokens - 1, 0)
        score = hit - length_penalty
        return Reward(
            run_id=trajectory.run_id,
            trajectory_id=trajectory.trajectory_id,
            score=score,
            components={"verifier": hit, "length_penalty": -length_penalty},
        )

