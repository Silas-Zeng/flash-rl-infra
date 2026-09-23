# Engineering release plan

The repository is organized around stable boundaries rather than a model
checkpoint:

1. `schemas.py` defines prompt, trajectory, reward, experience and checkpoint
   records.
2. `store.py` provides idempotent JSONL stages and a SQLite index.
3. `rollout.py` separates the local backend from the optional SGLang HTTP
   backend.
4. `distributed.py` validates rank sharding, group reduction, DDP updates and
   checkpoint versioning.
5. `compression.py` and `speculative.py` contain portable reference operators
   with explicit error and memory accounting.
6. `attention.py` exposes dense CED/CSA2 mode and sparse-indexer semantics for
   correctness tests without claiming a fused production kernel.
7. `ablation.py` produces deterministic same-workload comparisons; its virtual
   cost is not a GPU benchmark.

Before a public release, the rented-GPU validation should add a real SGLang
rollout endpoint, record prefill/decode latency and HBM usage, verify
checkpoint weight updates, and compare task quality after each cache format.
Those measurements must be stored with the model revision, CUDA/PyTorch/SGLang
versions, GPU type, sequence length and seed.
