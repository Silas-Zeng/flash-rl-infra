# Changelog

## Unreleased

- Added a torchrun/DDP multi-rank rollout and training data path.
- Added baseline versus Flash-style ablation metrics for rollout scheduling,
  stale-token masking, length control, and speculative acceptance.
- Added portable reference implementations for FP16/INT8/FP4-style tensor
  packing, cross-layer KV references, tiered KV lifecycle, bounded replay,
  Engram sharding metadata, and INT8 gradient payloads.
- Added a small MTP head, confidence-prefix verifier, coverage matrix, and
  CI/test documentation.

The repository does not claim to contain DeepSeek's private checkpoints,
native kernels, or production serving infrastructure.
