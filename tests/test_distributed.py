import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from flashrl.distributed import run_distributed
from flashrl.schemas import Trajectory
from flashrl.ablation import compare_modes
from flashrl.compression import (
    CrossLayerKVCache,
    TieredKVCache,
    compress_tensor,
    decompress_tensor,
    run_compression_benchmark,
)
from flashrl.speculative import MTPHead, verify_candidates
from flashrl.attention import CSA2Mode, CSA2Reference, CEDPlan, CEDReference
from flashrl.optim import run_optimizer_smoke


def test_single_process_distributed_contract(tmp_path: Path):
    summary = run_distributed(
        tmp_path / "distributed",
        groups=2,
        group_size=2,
        steps=1,
        max_tokens=2,
    )
    assert summary is not None
    assert summary["world_size"] == 1
    assert summary["samples"] == 4
    assert summary["policy_version"] == 1
    assert (tmp_path / "distributed" / "checkpoints" / "step-0001.pt").exists()
    lines = (tmp_path / "distributed" / "rollouts" / "rank-0000.jsonl").read_text().splitlines()
    assert len(lines) == 4
    record = json.loads(lines[0])
    trajectory = Trajectory(
        run_id=record["run_id"],
        trajectory_id=record["trajectory_id"],
        prompt_id=record["prompt_id"],
        group_id=record["group_id"],
        prompt_ids=record["prompt_ids"],
        response_ids=record["response_ids"],
        rollout_logprobs=record["rollout_logprobs"],
        finish_reason=record["finish_reason"],
        policy_version=record["policy_version"],
        generated_tokens=record["generated_tokens"],
        wall_time_ms=record["wall_time_ms"],
        interrupted=record["interrupted"],
    )
    assert trajectory.generated_tokens == 2


def test_distributed_reference_optimizer_mode(tmp_path: Path):
    summary = run_distributed(
        tmp_path / "distributed-optim",
        groups=1,
        group_size=2,
        steps=1,
        max_tokens=2,
        optimizer_mode="flash_reference",
    )
    assert summary is not None
    assert summary["optimizer_mode"] == "flash_reference"


def test_ablation_compares_the_same_workload(tmp_path: Path):
    result = compare_modes(samples=8, seed=7, output=tmp_path / "ablation.json")
    baseline = result["baseline"]
    flash = result["flash"]
    assert baseline["tokens"] == flash["tokens"]
    assert baseline["accepted_tokens"] == 0
    assert flash["proposed_tokens"] > 0
    assert flash["stale_tokens_masked"] > 0
    assert flash["max_inflight_observed"] <= 8
    assert (tmp_path / "ablation.json").exists()
    assert any(item["feature"] == "native_fp4_cuda_kernel" for item in result["coverage_limits"])


def test_fp4_and_cross_layer_cache_are_recoverable():
    torch.manual_seed(3)
    tensor = torch.randn(4, 8)
    packed = compress_tensor(tensor, kind="fp4", group_size=4)
    restored = decompress_tensor(packed)
    assert packed.compressed_nbytes < tensor.numel() * 2
    assert restored.shape == tensor.shape
    assert float((tensor - restored).pow(2).mean()) < 0.2

    cache = CrossLayerKVCache(kind="int8", group_size=4)
    cache.put(0, tensor, tensor, share_key="pair-0")
    cache.put(1, tensor, tensor, share_key="pair-0")
    assert cache.canonical_layers == 1
    assert cache.layers == 2
    assert torch.equal(cache.get(0)[0], cache.get(1)[0])

    tiers = TieredKVCache(kind="int8", group_size=4)
    tiers.put(0, tensor, tensor, persistent=True)
    tiers.put(1, tensor, tensor, persistent=False)
    before = tiers.compressed_nbytes
    tiers.clear_temporary()
    assert tiers.compressed_nbytes < before


def test_compression_benchmark_reports_memory_and_replay():
    result = run_compression_benchmark(layers=2, tokens=4, kv_heads=1, head_dim=8, group_size=4)
    assert result["formats"]["fp4"]["compression_ratio"] > 1.0
    assert result["cross_layer_fp4"]["referenced_layers"] == 1
    assert result["bounded_kv_replay"]["evictions"] > 0


def test_mtp_head_and_confidence_verifier():
    head = MTPHead(hidden_size=8, vocab_size=16, future_tokens=2)
    hidden = torch.randn(3, 8)
    targets = torch.randint(0, 16, (3, 2))
    assert head.loss(hidden, targets).item() >= 0
    trace = verify_candidates(torch.log(torch.tensor([0.9, 0.8, 0.2])), torch.tensor([1, 2, 3]), min_probability=0.5)
    assert trace.accepted == 2
    assert trace.rejected_at == 2


def test_csa2_modes_and_ced_projection():
    torch.manual_seed(4)
    query = torch.randn(3, 8)
    global_k = torch.randn(12, 8)
    global_v = torch.randn(12, 8)
    csa = CSA2Reference([CSA2Mode.FULL, CSA2Mode.REINDEX, CSA2Mode.REUSE], top_k=4, block_size=3)
    outputs = [csa.run_layer(layer, query, global_k, global_v)[0] for layer in range(3)]
    assert all(output.shape == query.shape for output in outputs)
    plan = CEDPlan(encoder_layers=2, decoder_layers=2, swa_window=4)
    keys, values = CEDReference(plan).project_decoder_global_kv(
        torch.randn(5, 8), [torch.randn(8, 8) for _ in range(2)], [torch.randn(8, 8) for _ in range(2)]
    )
    assert len(keys) == len(values) == 2


def test_reference_optimizer_groups_update_finitely():
    result = run_optimizer_smoke()
    assert result["updated"] is True
    assert result["optimizers"] == ["adamw", "head_wise_muon", "sinkhorn_momentum"]
    assert result["loss_after"] < result["loss_before"]

