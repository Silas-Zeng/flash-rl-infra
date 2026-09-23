"""Small multi-GPU RL data-plane prototype.

The module deliberately keeps the model tiny.  The distributed contracts are
the important part: each rank owns a prompt shard, rewards are reduced before
the policy update, DDP synchronizes gradients, and rank 0 publishes a
versioned checkpoint.  A real SGLang rollout service can replace
``_rollout_one`` without changing the data or synchronization protocol.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedEnv:
    rank: int
    world_size: int
    local_rank: int
    backend: str
    device: torch.device


class TinyActor(nn.Module):
    """A compact causal policy used to validate the full distributed path."""

    def __init__(self, vocab_size: int = 32, hidden_size: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.embedding(input_ids))


def init_distributed(backend: str = "auto", init_method: str | None = None) -> DistributedEnv:
    """Initialize torch.distributed from the standard torchrun environment."""

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    use_cuda = torch.cuda.is_available()
    if backend == "auto":
        backend = "nccl" if use_cuda else "gloo"
    if backend == "nccl" and not use_cuda:
        raise RuntimeError("NCCL was requested but CUDA is unavailable")
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        kwargs: dict[str, Any] = {"backend": backend, "rank": rank, "world_size": world_size}
        if init_method is not None:
            kwargs["init_method"] = init_method
        dist.init_process_group(**kwargs)
    return DistributedEnv(rank, world_size, local_rank, backend, device)


def _jsonl_append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _rollout_one(
    actor: nn.Module,
    prompt_token: int,
    *,
    max_tokens: int,
    vocab_size: int,
    seed: int,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    """Generate a response and retain behavior-policy log probabilities."""

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    sequence = [prompt_token]
    response: list[int] = []
    old_logprobs: list[float] = []
    with torch.no_grad():
        for _ in range(max_tokens):
            inputs = torch.tensor([sequence], dtype=torch.long, device=device)
            logits = actor(inputs)[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            token = torch.multinomial(log_probs.exp(), 1, generator=generator)
            token_id = int(token.item())
            response.append(token_id)
            old_logprobs.append(float(log_probs[0, token_id].item()))
            sequence.append(token_id)
    if len(response) != max_tokens:
        raise RuntimeError("rollout returned an unexpected token count")
    if any(token < 0 or token >= vocab_size for token in response):
        raise RuntimeError("rollout produced a token outside the policy vocabulary")
    return response, old_logprobs


def _sample_loss(
    actor: nn.Module,
    prompt_token: int,
    response: list[int],
    advantage: float,
    device: torch.device,
) -> torch.Tensor:
    """REINFORCE loss over response tokens; prompt transitions are masked."""

    sequence = torch.tensor([[prompt_token, *response]], dtype=torch.long, device=device)
    logits = actor(sequence[:, :-1])
    targets = sequence[:, 1:]
    token_logprobs = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # With a one-token prompt every transition predicts a response token.
    return -(token_logprobs.mean() * float(advantage))


def _reduce_group_baselines(
    env: DistributedEnv,
    group_sums: torch.Tensor,
    group_counts: torch.Tensor,
) -> torch.Tensor:
    if env.world_size > 1:
        dist.all_reduce(group_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(group_counts, op=dist.ReduceOp.SUM)
    return group_sums / group_counts.clamp_min(1.0)


def run_distributed(
    output: str | Path,
    *,
    groups: int = 4,
    group_size: int = 2,
    steps: int = 2,
    max_tokens: int = 4,
    vocab_size: int = 32,
    hidden_size: int = 64,
    learning_rate: float = 0.05,
    optimizer_mode: str = "adamw",
    seed: int = 20260923,
    backend: str = "auto",
    init_method: str | None = None,
) -> dict[str, Any] | None:
    """Run a sharded rollout -> reward -> synchronized update loop.

    ``groups`` and ``group_size`` describe GRPO-style groups.  Samples are
    assigned by global sample index so changing the number of ranks does not
    change the dataset or produce duplicate trajectory ids.
    """

    if groups < 1 or group_size < 1 or steps < 1:
        raise ValueError("groups, group_size and steps must be positive")
    if optimizer_mode not in {"adamw", "flash_reference"}:
        raise ValueError("optimizer_mode must be 'adamw' or 'flash_reference'")
    env = init_distributed(backend, init_method=init_method)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "rollouts").mkdir(exist_ok=True)
    if env.rank == 0:
        (output_path / "checkpoints").mkdir(exist_ok=True)

    torch.manual_seed(seed + env.rank)
    actor = TinyActor(vocab_size=vocab_size, hidden_size=hidden_size).to(env.device)
    ddp_actor: nn.Module = (
        DistributedDataParallel(actor, device_ids=[env.local_rank])
        if env.world_size > 1 and env.device.type == "cuda"
        else DistributedDataParallel(actor)
        if env.world_size > 1
        else actor
    )
    if optimizer_mode == "adamw":
        optimizers = {"adamw": torch.optim.AdamW(ddp_actor.parameters(), lr=learning_rate)}
    else:
        # The reference grouping operates on the underlying actor and updates
        # the same parameters wrapped by DDP.  It is intentionally opt-in:
        # production runs should replace it with a fused implementation.
        from .optim import build_flash_optimizers

        optimizers = build_flash_optimizers(actor.named_parameters(), lr=learning_rate)
        if not optimizers:
            raise RuntimeError("flash_reference produced no trainable parameter groups")
    policy_version = 0
    rank_rollout_path = output_path / "rollouts" / f"rank-{env.rank:04d}.jsonl"
    local_reward_sum = 0.0
    local_sample_count = 0

    for step in range(steps):
        # A fresh batch is collected before the update.  In a production run
        # this queue is bounded and can be filled by an asynchronous backend.
        pending: list[dict[str, Any]] = []
        group_sums = torch.zeros(groups, dtype=torch.float64, device=env.device)
        group_counts = torch.zeros(groups, dtype=torch.float64, device=env.device)
        for group_id in range(groups):
            for sample_id in range(group_size):
                global_index = step * groups * group_size + group_id * group_size + sample_id
                if global_index % env.world_size != env.rank:
                    continue
                prompt_token = 1 + ((group_id * group_size + sample_id) % (vocab_size - 1))
                expected_token = (prompt_token + 1) % vocab_size
                response, old_logprobs = _rollout_one(
                    ddp_actor,
                    prompt_token,
                    max_tokens=max_tokens,
                    vocab_size=vocab_size,
                    seed=seed + global_index + step * 100_000,
                    device=env.device,
                )
                reward = (1.0 if expected_token in response else 0.0) - 0.01 * len(response)
                group_sums[group_id] += reward
                group_counts[group_id] += 1.0
                pending.append(
                    {
                        "schema_version": 1,
                        "record_type": "Trajectory",
                        "run_id": output_path.name,
                        "trajectory_id": f"step-{step:04d}-g-{group_id:04d}-s-{sample_id:04d}",
                        "prompt_id": f"step-{step:04d}-g-{group_id:04d}-s-{sample_id:04d}",
                        "rank": env.rank,
                        "step": step,
                        "group_id": f"g-{group_id:04d}",
                        "group_index": group_id,
                        "sample_id": sample_id,
                        "prompt_ids": [prompt_token],
                        "response_ids": response,
                        "rollout_logprobs": old_logprobs,
                        "finish_reason": "length",
                        "generated_tokens": len(response),
                        "wall_time_ms": 0.0,
                        "interrupted": False,
                        "tokenizer_hash": "toy-v1",
                        "reward": reward,
                        "policy_version": policy_version,
                        "stale_tokens": 0,
                    }
                )

        baselines = _reduce_group_baselines(env, group_sums, group_counts)
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        for record in pending:
            advantage = float(record["reward"] - baselines[record["group_index"]].item())
            record["advantage"] = advantage
            losses.append(
                _sample_loss(
                    ddp_actor,
                    record["prompt_ids"][0],
                    record["response_ids"],
                    advantage,
                    env.device,
                )
            )
            _jsonl_append(rank_rollout_path, record)
            local_reward_sum += float(record["reward"])
            local_sample_count += 1
        batch_count = torch.tensor([float(len(losses))], dtype=torch.float64, device=env.device)
        if env.world_size > 1:
            dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
        global_batch_count = max(float(batch_count.item()), 1.0)
        if losses:
            # DDP averages gradients across ranks. Weight the local sum by
            # world_size/global_count so uneven rank shards still produce the
            # same gradient as one global mean batch.
            scale = env.world_size / global_batch_count if env.world_size > 1 else 1.0 / global_batch_count
            torch.stack(losses).sum().mul_(scale).backward()
        else:
            # This happens only when there are more ranks than samples.  A
            # zero loss still participates in DDP's collective graph.
            (ddp_actor(torch.zeros((1, 1), dtype=torch.long, device=env.device)).sum() * 0.0).backward()
        for optimizer in optimizers.values():
            optimizer.step()

        version_tensor = torch.tensor([policy_version + 1 if env.rank == 0 else 0], dtype=torch.long, device=env.device)
        if env.world_size > 1:
            dist.broadcast(version_tensor, src=0)
        policy_version = int(version_tensor.item())
        if env.world_size > 1:
            dist.barrier()
        if env.rank == 0:
            checkpoint = output_path / "checkpoints" / f"step-{step + 1:04d}.pt"
            torch.save(
                {
                    "model": actor.state_dict(),
                    "optimizer_mode": optimizer_mode,
                    "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                    "policy_version": policy_version,
                    "step": step + 1,
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                },
                checkpoint,
            )
            manifest = {
                "schema_version": 1,
                "policy_version": policy_version,
                "checkpoint": str(checkpoint.relative_to(output_path)),
                "world_size": env.world_size,
                "step": step + 1,
                "created_at": time.time(),
            }
            (output_path / "checkpoints" / f"step-{step + 1:04d}.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
        if env.world_size > 1:
            dist.barrier()

    reward_stats = torch.tensor([local_reward_sum, float(local_sample_count)], dtype=torch.float64, device=env.device)
    if env.world_size > 1:
        dist.all_reduce(reward_stats, op=dist.ReduceOp.SUM)
    summary = None
    if env.rank == 0:
        summary = {
            "schema_version": 1,
            "world_size": env.world_size,
            "backend": env.backend,
            "device": str(env.device),
            "groups": groups,
            "group_size": group_size,
            "steps": steps,
            "samples": int(reward_stats[1].item()),
            "mean_reward": float(reward_stats[0].item() / max(reward_stats[1].item(), 1.0)),
            "policy_version": policy_version,
            "optimizer_mode": optimizer_mode,
        }
        (output_path / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if env.world_size > 1:
        dist.barrier()
    return summary


def close_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

