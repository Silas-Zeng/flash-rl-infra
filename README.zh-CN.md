# FlashRL Infra

[English](README.md) | [简体中文](README.zh-CN.md)

FlashRL 是一个可运行的强化学习数据平面，参考了公开的
DeepSeek-V4.1-Flash 技术报告。项目优先实现可以在单机复现的部分，再用
torchrun 将相同的数据契约扩展到多卡：有界 rollout、奖励与 verifier、可
回放记录、带版本陈旧度的 experience、同步 learner 更新以及 checkpoint
发布。

报告描述的是一个异步 post-training 系统：限制 rollout 并发数，按 sample
和 group 调度，在 token 边界暂停生成，持久化 rollout 状态，屏蔽陈旧 token，
并在 checkpoint 切换后恢复未完成样本。本项目实现这些数据流契约，但不声称
复现 DeepSeek 的私有生产系统。

## 快速开始

```powershell
cd flash-rl-infra
uv pip install -e . --system
python -m pytest -q
python -m flashrl.cli run --output runs/local-demo --groups 4 --group-size 2
python -m flashrl.cli inspect --run-dir runs/local-demo
```

本地流程使用轻量 tabular policy，不需要下载模型或 GPU。它会在
`runs/local-demo/events` 下写入 append-only JSONL stage 和 SQLite 幂等账本，
并在 `runs/local-demo/checkpoints` 下生成 checkpoint。

## Flash 风格 ablation

公开报告中的多项 post-training 和 serving 控制可以单独测量：有界 in-flight
rollout、sample/group 调度、token 边界中断与恢复、陈旧 token 屏蔽、长度偏置
控制，以及 draft/target verification。项目提供 algorithm-level simulator，
用同一批 workload 对比开启和关闭这些控制的结果：

```powershell
python -m flashrl.cli ablation --samples 128 --output runs/ablation/comparison.json
```

输出会为 `baseline` 和 `flash` 提供
`virtual_tokens_per_cost_unit`、`acceptance_rate`、`target_forward_reduction`、
`max_inflight_observed`、`stale_token_ratio`、`mean_reward`、`mean_task_reward`
和 `length_reward_correlation`，并给出 delta。virtual decode cost 是与模型
无关的算法成本单位，用来检查方向是否正确；真实 GPU 吞吐仍需租用 GPU 测量。
相同对比也可以通过 `python -m flashrl.distributed_cli ablation` 运行。

压缩和覆盖范围见 [docs/coverage.md](docs/coverage.md)。ablation JSON 还会输出
FP16、INT8、FP4-style packed KV、跨层引用、分层 KV、Engram 分片、INT8 梯度
payload，以及带版本化 payload snapshot 的有界 KV replay 参考结果。它不声称
实现原生 DeepSeek kernel 或训练好的 DeepSeek 模型。

它还会运行 MTP head 的一步训练更新、标准 speculative acceptance
（`min(1, p/q)`、残差修正和 bonus token）smoke test，以及 CED/CSA2
Full-Reindex-Reuse 的 dense reference smoke test。报告同时包含 head-wise Muon
和 Sinkhorn-balanced 参数分组的一步 smoke；这些是便于检查的 PyTorch 更新，
不是生产 pre-training 使用的 fused kernel。

## 多卡路径

分布式原型保持很小，但在租用 GPU 前验证了关键边界：

- 每个 rank 获得确定性的 trajectory ID 分片，重试不会产生重复样本；
- 在 policy update 前按 GRPO 风格 group 做 reward reduction；
- DDP 同步梯度，rank 0 发布带版本的 checkpoint；
- 每个 rank 写入自己的不可变 rollout JSONL 分片；
- run summary 记录 world size、backend、policy version 和 mean reward。

在单节点上安装 torch 并启动：

```powershell
uv pip install -e ".[torch,dev]" --system
torchrun --standalone --nproc_per_node 2 -m flashrl.distributed_cli run `
  --output runs/multigpu --groups 4 --group-size 2 --steps 2
Get-Content runs/multigpu/run_summary.json
```

要在同一条多卡流程中使用参考优化器分组，添加
`--optimizer-mode flash_reference`。默认的 `adamw` 是 smoke command 使用的
稳定 baseline。

仓库内置了 `scripts/run_multigpu.ps1` 和 `scripts/run_multigpu.sh`。在租用的
NVIDIA 机器上，`--nproc_per_node` 应等于可见 GPU 数，命令会自动选择 NCCL。
输出目录必须位于 rank 0 和 rollout ranks 都能访问的文件系统上，这样才能共同
检查 checkpoint 和各 rank 的分片。

当前 Windows CPU 开发环境中的 PyTorch wheel 可能没有 libuv，导致
`torchrun --standalone` 无法创建 rendezvous store。可以用 file-store launcher
检查相同的两进程数据和梯度路径：

```powershell
python scripts/run_multigpu_local.py --nproc 2 --output runs/multigpu-local
```

这个 workaround 只用于本地 Gloo 验证；Linux CUDA 租用环境应使用上面的
torchrun/NCCL 命令。

## 数据流

```text
Prompt -> Rollout -> Reward/Verifier -> Experience -> Learner
   ^                                                        |
   |---------------- checkpoint / policy_version ------------|
```

每条 trajectory 都携带 `run_id`、`trajectory_id`、`group_id`、token ID、逐 token
rollout logprob 和 `policy_version`。Derived experience 可以从原始 stage 重建。
写入使用幂等 key，因此 at-least-once 重试不会重复样本。

## 真实 SGLang 路径

安装可选 HTTP 依赖并启动兼容的 SGLang server。`SGLangRolloutBackend` 通过
`/generate` 发送 tokenized prompt 和 `return_logprob=true`，要求响应包含
`output_token_logprobs`；只有文本的响应会被拒绝，因为文本无法安全恢复
action-token 和 logprob 的对应关系。learner checkpoint 发布后，
`flashrl.sync.update_sglang_from_disk` 调用 `/update_weights_from_disk` 控制接口。

通过多卡契约测试后，可以按以下顺序接入真实流程：

1. 将 `LocalRolloutBackend` 替换为 `SGLangRolloutBackend`；
2. 将 `ArithmeticReward` 替换为 sandbox/verifier service；
3. 将 `ToyLearner` 替换为 GRPO/PPO learner，并保留相同 schemas；
4. 为每个 rollout shard 运行一个 SGLang endpoint，再加入有界异步 worker
   和 checkpoint-version admission rules；
5. 测量 rollout 利用率、队列延迟、陈旧 token 比例、reward、loss、accepted
   length 和端到端 step time。

## 范围边界

这是一个可执行的数据平面契约复现项目，不声称重新实现 552B 多模态模型、
CSA2 kernel、FP4 KV kernel、DSpark 或 DeepSeek 的私有 DSec 基础设施。数据流
稳定后，这些部分可以作为后续实验 backend 接入。
