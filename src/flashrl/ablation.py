"""Deterministic algorithm-level ablation for the Flash-style data path.

The simulator keeps the model-independent parts measurable on any machine.
Its decode cost is a virtual cost unit, so the comparison is not confused with
hardware benchmark numbers.  The same fields are present in the distributed
JSONL records: bounded in-flight work, token interruption/resume, stale-token
masking, length-bias control, and speculative acceptance.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AblationConfig:
    name: str
    bounded_inflight: bool = False
    interrupt_resume: bool = False
    stale_token_masking: bool = False
    length_bias_control: bool = False
    speculative_decode: bool = False
    max_inflight: int = 8
    interrupt_every: int = 4
    draft_tokens: int = 4
    stale_prefix_tokens: int = 1
    stale_every: int = 4


BASELINE = AblationConfig(name="baseline", max_inflight=10_000)
FLASH = AblationConfig(
    name="flash",
    bounded_inflight=True,
    interrupt_resume=True,
    stale_token_masking=True,
    length_bias_control=True,
    speculative_decode=True,
)


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    return numerator / (denom_x * denom_y) if denom_x and denom_y else 0.0


def run_ablation(
    config: AblationConfig,
    *,
    samples: int = 64,
    seed: int = 20260923,
) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("samples must be positive")
    if config.max_inflight < 1 or config.interrupt_every < 1 or config.draft_tokens < 1:
        raise ValueError("ablation limits must be positive")
    started = time.perf_counter()
    rng = random.Random(seed)
    total_tokens = 0
    accepted_tokens = 0
    proposed_tokens = 0
    target_forwards = 0
    stale_masked = 0
    interruptions = 0
    resumed_tokens = 0
    virtual_cost = 0.0
    rewards: list[float] = []
    task_rewards: list[float] = []
    lengths: list[float] = []
    for sample_id in range(samples):
        # Fixed per-sample lengths let both modes see the same workload.
        target_length = 8 + ((sample_id * 11 + seed) % 13)
        # Keep answer quality independent of length so the baseline's
        # correlation comes from its length bonus alone.
        quality = 0.72
        total_tokens += target_length
        lengths.append(float(target_length))

        if config.interrupt_resume:
            interruptions += max(0, math.ceil(target_length / config.interrupt_every) - 1)
            resumed_tokens += max(0, target_length - config.interrupt_every)

        if config.stale_token_masking and sample_id % config.stale_every == 0:
            stale_masked += min(config.stale_prefix_tokens, target_length)

        remaining = target_length
        while remaining:
            if config.speculative_decode:
                proposal = min(config.draft_tokens, remaining)
                proposed_tokens += proposal
                accepted = sum(rng.random() < quality for _ in range(proposal))
                accepted_tokens += accepted
                rejected = proposal - accepted
                # One target verification covers a draft block; rejected
                # tokens need a target correction step.
                target_forwards += 1 + rejected
                virtual_cost += proposal * 0.22 + 0.75 + rejected * 0.55
                progress = accepted + (1 if rejected else 0)
            else:
                target_forwards += remaining
                virtual_cost += remaining
                progress = remaining
            remaining -= min(progress, remaining)

        raw_reward = quality + 0.01 * target_length
        task_rewards.append(quality)
        rewards.append(raw_reward - (0.01 * target_length if config.length_bias_control else 0.0))

    max_inflight = min(config.max_inflight, samples) if config.bounded_inflight else samples
    windows = math.ceil(samples / max_inflight)
    virtual_cost += windows * 0.1 + interruptions * 0.05
    result = {
        "mode": config.name,
        "config": asdict(config),
        "samples": samples,
        "tokens": total_tokens,
        "accepted_tokens": accepted_tokens,
        "proposed_tokens": proposed_tokens,
        "acceptance_rate": accepted_tokens / proposed_tokens if proposed_tokens else 0.0,
        "target_forwards": target_forwards,
        "target_forward_reduction": 1.0 - target_forwards / total_tokens,
        "virtual_decode_cost_units": virtual_cost,
        "virtual_tokens_per_cost_unit": total_tokens / virtual_cost if virtual_cost else 0.0,
        "max_inflight_observed": max_inflight,
        "dispatch_windows": windows,
        "interruptions": interruptions,
        "resumed_tokens": resumed_tokens,
        "stale_tokens_masked": stale_masked,
        "stale_token_ratio": stale_masked / total_tokens,
        "mean_reward": sum(rewards) / samples,
        "mean_task_reward": sum(task_rewards) / samples,
        "length_reward_correlation": _pearson(lengths, rewards),
        "wall_time_ms": (time.perf_counter() - started) * 1000,
    }
    return result


def compare_modes(*, samples: int = 64, seed: int = 20260923, output: str | Path | None = None) -> dict[str, Any]:
    baseline = run_ablation(BASELINE, samples=samples, seed=seed)
    flash = run_ablation(FLASH, samples=samples, seed=seed)
    delta: dict[str, float] = {}
    for key in (
        "virtual_tokens_per_cost_unit",
        "acceptance_rate",
        "mean_reward",
        "mean_task_reward",
        "stale_token_ratio",
    ):
        delta[key] = float(flash[key] - baseline[key])
    try:
        from .compression import run_compression_benchmark

        compression = {"available": True, **run_compression_benchmark()}
    except ImportError as exc:
        compression = {"available": False, "reason": str(exc)}
    try:
        from .speculative import run_mtp_smoke

        mtp = {"available": True, **run_mtp_smoke()}
    except ImportError as exc:
        mtp = {"available": False, "reason": str(exc)}
    try:
        from .attention import run_attention_smoke

        attention = {"available": True, **run_attention_smoke()}
    except ImportError as exc:
        attention = {"available": False, "reason": str(exc)}
    try:
        from .optim import run_optimizer_smoke

        optimizers = {"available": True, **run_optimizer_smoke()}
    except ImportError as exc:
        optimizers = {"available": False, "reason": str(exc)}
    result = {
        "baseline": baseline,
        "flash": flash,
        "delta_flash_minus_baseline": delta,
        "compression": compression,
        "mtp": mtp,
        "attention": attention,
        "optimizers": optimizers,
        "coverage_limits": [
            {"feature": "native_fp4_cuda_kernel", "status": "not_available", "reason": "requires vendor kernel and compatible GPU"},
            {"feature": "trained_csa2_architecture", "status": "not_available", "reason": "requires the trained model and checkpoint"},
            {"feature": "mega_mhc_and_full_engram_training", "status": "reference_only", "reason": "private architecture/training recipe is unavailable"},
            {"feature": "dspark_opd_teacher_fleet", "status": "control_flow_only", "reason": "requires teacher checkpoints and serving fleet"},
        ],
    }
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

