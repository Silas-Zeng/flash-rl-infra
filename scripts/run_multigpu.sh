#!/usr/bin/env bash
set -euo pipefail

NPROC="${NPROC:-2}"
OUTPUT="${OUTPUT:-runs/multigpu}"
torchrun --standalone --nproc_per_node "${NPROC}" -m flashrl.distributed_cli run \
  --output "${OUTPUT}" --groups "${GROUPS:-4}" --group-size "${GROUP_SIZE:-2}" \
  --steps "${STEPS:-2}"

