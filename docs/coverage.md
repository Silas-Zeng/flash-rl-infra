# Technical coverage

The implementation separates a portable reference path from features that
require DeepSeek's trained weights or custom GPU kernels.

| Report area | Status in this project | Where |
| --- | --- | --- |
| Multi-rank rollout/training, group baseline, versioned checkpoints | Runnable with torchrun, DDP, Gloo/NCCL | `src/flashrl/distributed.py` |
| Bounded in-flight dispatch and token interruption/resume | Runnable scheduling/metadata path and deterministic ablation | `src/flashrl/ablation.py` |
| Stale-token masking and length-bias control | Runnable in learner contract and ablation | `src/flashrl/trainer.py`, `src/flashrl/ablation.py` |
| Draft/target verification and acceptance metrics | Runnable probability-prefix toy verifier; not target-model speculative decoding | `src/flashrl/speculative.py` |
| MTP training heads | Runnable small differentiable module and smoke loss; no trained drafter | `src/flashrl/speculative.py` |
| FP16, symmetric INT8, packed FP4-style KV | Runnable portable reference quantizer with reconstruction/error metrics; not OCP MXFP4 | `src/flashrl/compression.py` |
| CSA2-like cross-layer KV references | Reference dictionary sharing only when tensors are already identical; no CSA2 attention architecture | `src/flashrl/compression.py` |
| Persistent KV vs temporary encoder/SWA tier | Lifecycle prototype with byte accounting; no attention read path or device migration | `src/flashrl/compression.py` |
| SWA bounded replay | Bounded metadata/eviction model; no persisted KV payload or exact resume | `src/flashrl/compression.py` |
| Sharded Engram table | Local hash ownership and storage accounting; no remote lookup, n-gram injection or training | `src/flashrl/compression.py` |
| INT8 gradient/communication payload | Codec and byte/error report only; no DDP communication hook or convergence study | `src/flashrl/compression.py` |
| CED projection planner, hierarchical sparse indexer, Full/Reindex/Reuse state machine | Runnable dense reference; no trained sparse Transformer or fused kernel | `src/flashrl/attention.py` |
| Native OCP MXFP4/QAT, FP8 SWA KV, trained CSA2, Mega-mHC, 552B weights | Not reproducible from public interfaces alone | Requires checkpoints, training recipe, kernels and hardware |
| Full DSpark/OPD/heterogeneous teacher service | Control-flow simulator only | Requires teacher checkpoints and serving fleet |
| Head-wise Muon, Sinkhorn-balanced Engram updates, multimodal sharding/overlap | Not implemented | Requires full pretraining stack and multimodal model |

Run the combined report with:

```powershell
python -m flashrl.cli ablation --samples 128 --output runs/ablation/comparison.json
```

The `compression` section reports bytes, compression ratios, reconstruction
MSE, cross-layer references, Engram shard bytes, gradient payload size, and
bounded replay evictions alongside the baseline/Flash rollout comparison.
