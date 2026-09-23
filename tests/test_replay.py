import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from flashrl.compression import BoundedKVReplay  # noqa: E402


def _append_fixture(replay: BoundedKVReplay) -> None:
    replay.append(
        "sample-0",
        0,
        policy_version=3,
        kv_bytes=128,
        payload={"tokens": [11, 12], "opaque": b"kv-0"},
    )
    replay.append(
        "sample-0",
        1,
        policy_version=3,
        kv_bytes=256,
        payload=torch.tensor([[1.0, 2.0]], dtype=torch.float16),
    )


def test_replay_payload_round_trip_preserves_order_payload_and_rng():
    replay = BoundedKVReplay(max_tokens=3, seed=20260924)
    _append_fixture(replay)
    payload = replay.to_payload()
    expected_random = replay.rng.random()

    restored = BoundedKVReplay.from_payload(payload)

    assert restored.max_tokens == replay.max_tokens
    assert restored.evictions == replay.evictions
    assert restored.items()[0][0] == ("sample-0", 0)
    assert restored.get("sample-0", 0)["payload"] == {"tokens": [11, 12], "opaque": b"kv-0"}
    assert torch.equal(
        restored.get("sample-0", 1)["payload"],
        torch.tensor([[1.0, 2.0]], dtype=torch.float16),
    )
    assert restored.rng.random() == expected_random


def test_replay_persist_restore_is_atomic_and_creates_parent(tmp_path: Path):
    replay = BoundedKVReplay(max_tokens=2, seed=7)
    replay.append("sample", 0, policy_version=1, kv_bytes=8, payload=b"state")
    path = tmp_path / "nested" / "replay.json"

    assert replay.persist(path) == path
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["format"] == "flashrl.bounded_kv_replay"
    assert not list(path.parent.glob("*.tmp"))

    restored = BoundedKVReplay.restore(path)
    assert restored.get("sample", 0)["payload"] == b"state"
    assert restored.rng.getstate() == replay.rng.getstate()


def test_replay_rejects_invalid_snapshot():
    with pytest.raises(ValueError, match="invalid bounded KV replay format"):
        BoundedKVReplay.from_payload({"version": 1})

    replay = BoundedKVReplay(max_tokens=1)
    with pytest.raises(ValueError, match="states exceed max_tokens"):
        BoundedKVReplay.from_payload(
            {
                "format": "flashrl.bounded_kv_replay",
                "version": 1,
                "max_tokens": 1,
                "states": [
                    {"sample_id": "a", "token_index": 0, "policy_version": 0, "kv_bytes": 1},
                    {"sample_id": "b", "token_index": 0, "policy_version": 0, "kv_bytes": 1},
                ],
            }
        )
    assert replay.tokens == 0


def test_replay_payload_follows_fifo_eviction():
    replay = BoundedKVReplay(max_tokens=2)
    replay.append("sample", 0, policy_version=1, kv_bytes=1, payload="old")
    replay.append("sample", 1, policy_version=1, kv_bytes=1, payload="middle")
    replay.append("sample", 2, policy_version=1, kv_bytes=1, payload="new")

    assert replay.get("sample", 0) is None
    assert replay.get("sample", 1)["payload"] == "middle"
    assert replay.get("sample", 2)["payload"] == "new"
    assert replay.evictions == 1
