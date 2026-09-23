# FlashRL Infra

[English](README.md) | [简体中文](README.zh-CN.md)

FlashRL是一个可运行的强化学习（RL）数据平面，参考公开的
DeepSeek-V4.1-Flash技术报告。项目先实现可以在单机复现的部分，再用
`torchrun`将相同的数据契约扩展到多卡：有界采样、奖励与验证器、可回放记录、
带策略版本的训练样本、同步学习器更新，以及检查点发布。

报告描述的是一个异步后训练系统：限制采样并发数，按样本和组调度，在
`token`（模型词元）边界暂停生成，持久化采样状态，屏蔽陈旧`token`，并在检查点
切换后恢复未完成的样本。本项目实现这些数据流契约，但不声称复现DeepSeek
的私有生产系统。

## 快速开始

```powershell
cd flash-rl-infra
uv pip install -e . --system
python -m pytest -q
python -m flashrl.cli run --output runs/local-demo --groups 4 --group-size 2
python -m flashrl.cli inspect --run-dir runs/local-demo
```

本地流程使用轻量表格策略，不需要下载模型或GPU。它会在
`runs/local-demo/events`下写入只追加的JSONL阶段记录和SQLite幂等账本，
并在`runs/local-demo/checkpoints`下生成检查点。

## Flash风格消融实验

公开报告中的多项后训练和服务控制可以单独测量：有界在途采样、样本组调度、
`token`边界中断与恢复、陈旧`token`屏蔽、长度偏置控制，以及草稿模型与目标
模型验证。项目提供算法级模拟器，用同一批工作负载对比开启和关闭这些控制后的
结果：

```powershell
python -m flashrl.cli ablation --samples 128 --output runs/ablation/comparison.json
```

输出会为`baseline`和`flash`提供
`virtual_tokens_per_cost_unit`、`acceptance_rate`、`target_forward_reduction`、
`max_inflight_observed`、`stale_token_ratio`、`mean_reward`、`mean_task_reward`
和`length_reward_correlation`，并给出差值。虚拟解码成本是与模型无关的算法成本
单位，用来检查算法方向；它不是实际GPU吞吐基准。相同对比也可以通过
`python -m flashrl.distributed_cli ablation`运行。

压缩和覆盖范围见[docs/coverage.md](docs/coverage.md)。消融实验JSON还会输出
FP16、INT8、FP4风格的KV打包、跨层引用、分层KV、Engram分片、INT8梯度
负载，以及带版本化负载快照的有界KV回放参考结果。它不声称实现原生
DeepSeek内核或训练好的DeepSeek模型。

报告还会运行`MTP`头的一步训练更新、标准投机采样验证
（`min(1, p/q)`、残差修正和额外`token`）冒烟测试，以及CED/CSA2
`Full-Reindex-Reuse`稠密参考实现的冒烟测试。报告同时包含`head-wise Muon`和
`Sinkhorn-balanced`参数分组的一步冒烟；这些是便于检查的`PyTorch`更新，不是
生产预训练使用的融合内核。

## 多卡路径

分布式原型保持很小，但在租用GPU前验证了关键边界：

- 每个`rank`获得确定性的`trajectory`编号分片，重试不会产生重复样本；
- 在策略更新前按GRPO风格的组执行奖励归约；
- DDP同步梯度，`rank 0`发布带版本的检查点；
- 每个`rank`写入自己的不可变采样`JSONL`分片；
- 运行摘要记录进程数、后端、策略版本和平均奖励。

在单节点上安装torch并启动：

```powershell
uv pip install -e ".[torch,dev]" --system
torchrun --standalone --nproc_per_node 2 -m flashrl.distributed_cli run `
  --output runs/multigpu --groups 4 --group-size 2 --steps 2
Get-Content runs/multigpu/run_summary.json
```

要在同一条多卡流程中使用参考优化器分组，添加
`--optimizer-mode flash_reference`。默认的`adamw`是冒烟命令使用的稳定基线。

仓库内置了`scripts/run_multigpu.ps1`和`scripts/run_multigpu.sh`。在租用的
NVIDIA机器上，`--nproc_per_node`应等于可见GPU数，命令会自动选择NCCL。
输出目录必须位于rank 0和各采样进程都能访问的文件系统上，这样才能共同检查
检查点和各rank的分片。

当前Windows CPU开发环境中的PyTorch wheel可能没有libuv，导致
`torchrun --standalone`无法创建`rendezvous store`。可以用文件存储启动器检查
相同的两进程数据和梯度路径：

```powershell
python scripts/run_multigpu_local.py --nproc 2 --output runs/multigpu-local
```

这个临时方案只用于本地Gloo验证；Linux CUDA租用环境应使用上面的
torchrun/NCCL命令。

## 数据流

```text
Prompt -> Rollout -> Reward/Verifier -> Experience -> Learner
   ^                                                        |
   |---------------- checkpoint / policy_version ------------|
```

每条`trajectory`都携带`run_id`、`trajectory_id`、`group_id`、`token`编号、逐个
`token`的采样`logprob`和`policy_version`。派生训练样本可以从原始阶段记录重建。
写入使用幂等键，因此至少一次投递重试不会重复样本。

## 真实SGLang路径

安装可选HTTP依赖并启动兼容的SGLang服务。`SGLangRolloutBackend`通过
`/generate`发送分词后的prompt和`return_logprob=true`，要求响应包含
`output_token_logprobs`；只有文本的响应会被拒绝，因为文本无法安全恢复动作
`token`和`logprob`的对应关系。学习器检查点发布后，
`flashrl.sync.update_sglang_from_disk`调用`/update_weights_from_disk`控制接口。

通过多卡契约测试后，可以按以下顺序接入真实流程：

1. 将`LocalRolloutBackend`替换为`SGLangRolloutBackend`；
2. 将`ArithmeticReward`替换为沙箱/验证器服务；
3. 将`ToyLearner`替换为GRPO/PPO学习器，并保留相同的数据结构；
4. 为每个采样分片运行一个SGLang服务端点，再加入有界异步工作进程和检查点
   版本接入规则；
5. 测量采样利用率、队列延迟、陈旧`token`比例、奖励、损失、接受长度和端到端
   单步耗时。

## 范围边界

这是一个可执行的数据平面契约复现项目，不声称重新实现552B多模态模型、
CSA2内核、FP4 KV内核、DSpark或DeepSeek的私有DSec基础设施。数据流
稳定后，这些部分可以作为后续实验后端接入。
