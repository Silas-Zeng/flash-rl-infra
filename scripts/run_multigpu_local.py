"""Windows-friendly CPU/Gloo launcher used for the local contract test.

The CPU wheel in the development environment is built without libuv, which
can make ``torchrun --standalone`` fail before workers start.  This launcher
uses a file rendezvous instead.  On Linux/CUDA, prefer ``torchrun`` and the
NCCL path in ``run_multigpu.sh``.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch.multiprocessing as mp

# Allow the launcher to run from a fresh checkout before editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flashrl.distributed import close_distributed, run_distributed


def _worker(rank: int, world_size: int, init_method: str, kwargs: dict) -> None:
    import os

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    try:
        run_distributed(**kwargs, backend="gloo", init_method=init_method)
    finally:
        close_distributed()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("--output", default="runs/multigpu-local")
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--optimizer-mode", choices=["adamw", "flash_reference"], default="adamw")
    args = parser.parse_args()
    if args.nproc < 1:
        raise ValueError("--nproc must be positive")
    # The file is created by the parent and removed by torch.distributed after
    # rendezvous.  Keep the parent directory stable on Windows.
    with tempfile.NamedTemporaryFile(prefix="flashrl-rdzv-", suffix=".rdzv", delete=False) as handle:
        rendezvous_file = Path(handle.name)
    rendezvous_file.unlink(missing_ok=True)
    init_method = f"file:///{rendezvous_file.as_posix()}"
    kwargs = {
        "output": args.output,
        "groups": args.groups,
        "group_size": args.group_size,
        "steps": args.steps,
        "optimizer_mode": args.optimizer_mode,
    }
    try:
        mp.spawn(_worker, args=(args.nproc, init_method, kwargs), nprocs=args.nproc, join=True)
    finally:
        rendezvous_file.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

