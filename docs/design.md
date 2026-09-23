# Design notes

## What is reproduced

The public DeepSeek-V4.1-Flash report exposes the following post-training
mechanics that are useful as an engineering target:

- asynchronous rollout to reduce long-tail stalls;
- a bounded number of in-flight samples;
- sample-level dispatch around a GRPO group size;
- token-boundary interruption and resume;
- policy/checkpoint version on every sample;
- length-bias controls and stale-token loss masking;
- persisted rollout progress and per-sample garbage collection.

FlashRL implements the stable contracts first. The local simulator makes these
contracts testable without pretending that a toy policy has DeepSeek's model
quality or kernels.

## Record lifecycle

```text
created -> rollout -> reward -> experience -> consumed
                         \-> failed/retry
```

Raw rollout and reward records are immutable. `Experience` is a derived record
and may be rebuilt from raw stages. A stage write is idempotent on its key, and
the SQLite ledger is only an index; the JSONL data remains inspectable.

## Real backend contracts

The rollout adapter must return response token IDs and one sampled-token logprob
per response token. It must also report the policy/weight version that actually
served the request. The learner should reject or mask tokens that exceed the
configured staleness budget. Weight publication must be atomic from the
rollout worker's perspective: abort active requests, update with an explicit
numeric `weight_version`, flush/rebuild cache, verify the new version, then
resume.

## Multi-GPU contract

`flashrl.distributed` is the first executable scale-out path. `torchrun`
provides `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`; the launcher selects Gloo on
CPU and NCCL on CUDA. A global sample index determines its rank, which makes
the shard assignment independent of process arrival order. Each rank writes
`rollouts/rank-XXXX.jsonl`, while rank 0 writes a checkpoint and manifest only
after the DDP update and a barrier.

For each batch, local rewards and counts are reduced per group. The resulting
group baseline is used to calculate the advantage before the DDP backward pass.
The loss is scaled by the global sample count so uneven rank shards still
match a global-mean update. This gives the learner a real cross-rank collective
without requiring a large model or a model download. The tiny actor is replaceable: a production actor
can keep the same `prompt_ids`, `response_ids`, `old_logprobs`,
`policy_version`, and `stale_tokens` fields while the rollout implementation
calls SGLang.

The prototype collects a finite batch before each update. A production version
should put the same records behind a bounded queue, interrupt generation only
at token boundaries, persist unfinished state, and admit a sample only when
its policy-version staleness is within the configured budget.

The distributed JSONL records use the same token-level lineage as
`Trajectory`: `prompt_ids`, `response_ids`, `rollout_logprobs`,
`generated_tokens`, `policy_version`, and `stale_tokens`. This keeps the toy
actor and the SGLang adapter on one data contract instead of creating a second
training-only format.

## Trick switches and comparison

`flashrl.ablation` exposes two named configurations. `baseline` uses full
autoregressive target decoding and an unbounded request window. `flash` turns
on a bounded window, token-boundary resume bookkeeping, a stale-token prefix
mask, length reward correction, and draft/target verification. Both modes use
the same deterministic sample lengths and quality values, so the comparison
isolates control-flow effects. `virtual_decode_cost_units` is an algorithmic cost
unit, not a claim about wall-clock GPU performance. `mean_task_reward` keeps
the underlying task quality separate from the length-biased training signal;
the distributed runner is
where the same switches should be measured with an actual SGLang endpoint.
