# FlashRL Infra

[English](README.md) | [简体中文](README.zh-CN.md)

FlashRL is a small, runnable RL data plane inspired by the public
DeepSeek-V4.1-Flash report. It first focuses on the pieces that can be
reproduced on one workstation, then scales the same contracts across torchrun
ranks: bounded rollout, reward and verification, replayable records,
staleness-aware experiences, synchronized learner updates, and checkpoint
publication.

The report describes an asynchronous post-training system that keeps rollout
concurrency bounded, dispatches at sample and group granularity, pauses
generation at token boundaries, persists rollout state, masks stale tokens, and resumes
unfinished samples after a checkpoint switch. This repository models those
contracts without claiming to reproduce DeepSeek's private production stack.

## Quick start

```powershell
cd flash-rl-infra
uv pip install -e . --system
python -m pytest -q
python -m flashrl.cli run --output runs/local-demo --groups 4 --group-size 2
python -m flashrl.cli inspect --run-dir runs/local-demo
```

The local run uses a tiny tabular policy so it works without a model download or
GPU. It writes append-only JSONL stages and an SQLite idempotency ledger under
`runs/local-demo/events`, then creates a checkpoint under
`runs/local-demo/checkpoints`.

## Flash-style ablation

The public report describes several post-training and serving controls that
can be isolated independently: bounded in-flight rollout, sample/group
dispatch, token-boundary interruption and resume, stale-token masking,
length-bias control, and speculative draft/target verification. The project
includes an algorithm-level simulator so the same workload is compared with
these controls disabled and enabled:

```powershell
python -m flashrl.cli ablation --samples 128 --output runs/ablation/comparison.json
```

The output contains `virtual_tokens_per_cost_unit`, `acceptance_rate`,
`target_forward_reduction`, `max_inflight_observed`, `stale_token_ratio`,
`mean_reward`, `mean_task_reward`, and `length_reward_correlation` for `baseline` and `flash`,
plus a delta section. The virtual decode cost is a model-independent algorithmic
unit; it is useful for checking direction, but it is not a wall-clock GPU
benchmark. The same comparison is available
through `python -m flashrl.distributed_cli ablation`.

Compression and coverage details are in
[docs/coverage.md](docs/coverage.md). The ablation JSON also includes a
runnable reference report for FP16, INT8, FP4-style packed KV, cross-layer
references, tiered KV, Engram sharding, INT8 gradient payloads, and bounded KV
replay with versioned payload snapshots. It does not claim to be a native
DeepSeek kernel or trained model.
It also runs a small MTP head training update and standard speculative
acceptance (`min(1, p/q)`, residual correction, and bonus-token) smoke test,
plus a dense CED/CSA2 Full-Reindex-Reuse reference smoke test.
The report also runs a finite-step smoke for the reference head-wise Muon and
Sinkhorn-balanced parameter groups. These are inspectable PyTorch updates, not
the fused kernels used by a production pre-training run.

## Multi-GPU first path

The distributed prototype is intentionally small, but it exercises the
important boundaries before a rented GPU is attached:

- each rank receives a deterministic shard of trajectory IDs, so retries do
  not create duplicate samples;
- rewards are reduced by GRPO-style group before the policy update;
- DDP synchronizes gradients and rank 0 publishes a versioned checkpoint;
- every rank writes its own immutable rollout JSONL shard;
- the run summary records world size, backend, policy version and mean reward.

Install the torch extra and launch with the same command on one node:

```powershell
uv pip install -e ".[torch,dev]" --system
torchrun --standalone --nproc_per_node 2 -m flashrl.distributed_cli run `
  --output runs/multigpu --groups 4 --group-size 2 --steps 2
Get-Content runs/multigpu/run_summary.json
```

To exercise the reference optimizer grouping in the same multi-rank loop, add
`--optimizer-mode flash_reference`. The default `adamw` path is the stable
baseline used by the smoke command.

The checked-in helpers are `scripts/run_multigpu.ps1` and
`scripts/run_multigpu.sh`. On rented NVIDIA machines, use
`--nproc_per_node` equal to the number of visible GPUs; the command selects
NCCL automatically. The output directory must be on a filesystem visible to
rank 0 and the rollout ranks so that the published checkpoint and per-rank
shards can be inspected together.

For the current Windows CPU development environment, PyTorch's CPU wheel may
be built without libuv, which prevents `torchrun --standalone` from creating a
rendezvous store. The same two-rank data and gradient path can be checked with
the file-store launcher:

```powershell
python scripts/run_multigpu_local.py --nproc 2 --output runs/multigpu-local
```

This workaround is only for local Gloo validation; a Linux CUDA rental should
use the torchrun/NCCL command above.

## Data flow

```text
Prompt -> Rollout -> Reward/Verifier -> Experience -> Learner
   ^                                                        |
   |---------------- checkpoint / policy_version ------------|
```

Every trajectory carries `run_id`, `trajectory_id`, `group_id`, `token` IDs,
token-level rollout logprobs, and `policy_version`. Derived experiences are
rebuildable. Writes use an idempotency key, so at-least-once retries do not
duplicate a sample.

## Real SGLang path

Install the optional HTTP dependency and start a compatible SGLang server. The
`SGLangRolloutBackend` sends tokenized prompts with `return_logprob=true` and
requires `output_token_logprobs`; it refuses text-only responses because text
cannot safely reconstruct the action-token logprob alignment. After a learner
checkpoint, `flashrl.sync.update_sglang_from_disk` calls the documented
`/update_weights_from_disk` control endpoint.

After the multi-GPU contract tests pass, connect the real path as follows:

1. Replace `LocalRolloutBackend` with `SGLangRolloutBackend`.
2. Replace `ArithmeticReward` with a sandbox/verifier service.
3. Replace `ToyLearner` with a GRPO/PPO learner and keep the same schemas.
4. Run one SGLang endpoint per rollout shard, then add bounded async workers
   and checkpoint-version admission rules.
5. Measure rollout utilization, queue lag, stale-token ratio, reward, loss,
   accepted length, and end-to-end step time.

## Scope boundary

This is an executable reproduction of the data-plane contracts, not a claim to
reimplement a 552B multimodal model, CSA2 kernels, FP4 KV kernels, DSpark, or
DeepSeek's private DSec infrastructure. Those are the next experimental
backends once the data flow is stable.
