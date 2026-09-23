param(
    [int]$NProc = 2,
    [string]$Output = "runs/multigpu",
    [int]$Groups = 4,
    [int]$GroupSize = 2,
    [int]$Steps = 2
)

torchrun --standalone --nproc_per_node $NProc -m flashrl.distributed_cli run `
  --output $Output --groups $Groups --group-size $GroupSize --steps $Steps

