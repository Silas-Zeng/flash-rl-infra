# Technical coverage

The implementation separates a portable reference path from features that
require DeepSeek's trained weights or custom GPU kernels.

| Report area | Status in this project | Where |
| --- | --- | --- |
| Multi-rank rollout/training, group baseline, versioned checkpoints | Runnable with torchrun, DDP, Gloo/NCCL | `src/flashrl/distributed.py` |
| Bounded in-flight dispatch and token interruption/resume | Runnable scheduling/metadata path and deterministic ablation | `src/flashrl/ablation.py` |
| Stale-token masking and length-bias control | Runnable in learner contract and ablation | `src/flashrl/trainer.py`, `src/flashrl/ablation.py` |
| Draft/target verification and acceptance metrics | Runnable `min(1,p/q)` acceptance, residual correction, bonus-token reference; model forwards and kernel scheduling remain external | `src/flashrl/speculative.py` |
| MTP training heads | Runnable small differentiable module with one optimizer step; no trained drafter or checkpoint | `src/flashrl/speculative.py` |
| FP16, symmetric INT8, packed FP4-style KV | Runnable portable reference quantizer with reconstruction/error metrics; not OCP MXFP4 | `src/flashrl/compression.py` |
| CSA2-like cross-layer KV references | Reference dictionary sharing only when tensors are already identical; no CSA2 attention architecture | `src/flashrl/compression.py` |
| Persistent KV vs temporary encoder/SWA tier | Lifecycle prototype with byte accounting; no attention read path or device migration | `src/flashrl/compression.py` |
| SWA bounded replay | Bounded payload/eviction model with versioned JSON snapshot, tensor/bytes payloads, RNG restore, and atomic persistence; exact device KV resume remains external | `src/flashrl/compression.py` |
| Sharded Engram table | Local hash ownership and storage accounting; no remote lookup, n-gram injection or training | `src/flashrl/compression.py` |
| INT8 gradient/communication payload | Codec and byte/error report only; no DDP communication hook or convergence study | `src/flashrl/compression.py` |
| CED projection planner, hierarchical sparse indexer, Full/Reindex/Reuse state machine | Runnable dense reference; no trained sparse Transformer or fused kernel | `src/flashrl/attention.py` |
| Native OCP MXFP4/QAT, FP8 SWA KV, trained CSA2, Mega-mHC, 552B weights | Not reproducible from public interfaces alone | Requires checkpoints, training recipe, kernels and hardware |
| Full DSpark/OPD/heterogeneous teacher service | Control-flow simulator only | Requires teacher checkpoints and serving fleet |
| Head-wise Muon and Sinkhorn-balanced parameter updates | Runnable toy/reference optimizer grouping and finite-step smoke; no fused kernel or full pretraining convergence study | `src/flashrl/optim.py` |
| Multimodal sharding/communication overlap | Not implemented | Requires full multimodal pretraining stack and model |

Run the combined report with:

```powershell
python -m flashrl.cli ablation --samples 128 --output runs/ablation/comparison.json
```

The `compression` section reports bytes, compression ratios, reconstruction
MSE, cross-layer references, Engram shard bytes, gradient payload size, and
bounded replay evictions alongside the baseline/Flash rollout comparison.
