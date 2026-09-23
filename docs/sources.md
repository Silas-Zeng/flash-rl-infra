# Public references used for the prototype

This project copies public interface ideas and does not claim access to
DeepSeek's private implementation.

- [DeepSeek-V4.1-Flash technical report](https://arxiv.org/html/2609.19969v1):
  asynchronous post-training, bounded in-flight samples, sample-level
  dispatch, token-level interruption, persisted rollout state, stale-token
  masking, and heterogeneous-teacher OPD.
- [SGLang for RL](https://docs.sglang.io/docs/advanced_features/sglang_for_rl):
  memory-saver sleep/wake and `/update_weights_from_disk` weight publication.
- [SGLang speculative decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding):
  the serving-side extension point for draft/target verification experiments.
- [DeepSeek DeepSpec](https://github.com/deepseek-ai/DeepSpec): public
  reference code for speculative decoding research.

The current multi-GPU implementation starts with a tiny actor and Gloo/NCCL
collectives. The next backend can call SGLang for rollout while preserving the
same token IDs, sampled-token logprobs, policy version, and checkpoint
contracts.
