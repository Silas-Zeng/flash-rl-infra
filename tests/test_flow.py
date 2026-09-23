from pathlib import Path

from flashrl.orchestrator import FlashRLRun
from flashrl.schemas import SchemaError, Trajectory


def test_local_flow_preserves_lineage(tmp_path: Path):
    run = FlashRLRun(tmp_path / "run", groups=2, group_size=2)
    try:
        result = run.run()
    finally:
        run.close()
    assert result["trajectory_count"] == 4
    assert result["store_counts"] == {"rollout": 4, "reward": 4, "experience": 4}
    assert result["policy_version"] > 0
    assert Path(result["checkpoint"]).exists()


def test_trajectory_alignment_is_checked():
    try:
        Trajectory("r", "t", "p", "g", [1], [2], [], "length", 0, 0, 1.0)
    except SchemaError:
        return
    raise AssertionError("misaligned trajectory should be rejected")

