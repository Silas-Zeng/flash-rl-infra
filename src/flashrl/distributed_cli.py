"""CLI entry point for the multi-rank prototype (usually launched by torchrun)."""

from __future__ import annotations

import argparse
import json

from .ablation import compare_modes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flashrl-distributed")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the sharded rollout/training loop")
    run.add_argument("--output", default="runs/multigpu")
    run.add_argument("--groups", type=int, default=4)
    run.add_argument("--group-size", type=int, default=2)
    run.add_argument("--steps", type=int, default=2)
    run.add_argument("--max-tokens", type=int, default=4)
    run.add_argument("--vocab-size", type=int, default=32)
    run.add_argument("--hidden-size", type=int, default=64)
    run.add_argument("--learning-rate", type=float, default=0.05)
    run.add_argument("--seed", type=int, default=20260923)
    run.add_argument("--backend", choices=["auto", "gloo", "nccl"], default="auto")
    compare = sub.add_parser("ablation", help="compare baseline with Flash-style algorithmic tricks")
    compare.add_argument("--samples", type=int, default=64)
    compare.add_argument("--seed", type=int, default=20260923)
    compare.add_argument("--output", default="runs/ablation/comparison.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ablation":
        print(json.dumps(compare_modes(samples=args.samples, seed=args.seed, output=args.output), indent=2))
        return 0
    if args.command != "run":
        raise AssertionError(args.command)
    # Import torch only for the distributed command. This keeps the ablation
    # report usable on machines without a torch installation.
    from .distributed import close_distributed, run_distributed

    try:
        summary = run_distributed(
            args.output,
            groups=args.groups,
            group_size=args.group_size,
            steps=args.steps,
            max_tokens=args.max_tokens,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            backend=args.backend,
        )
        if summary is not None:
            print(summary)
        return 0
    finally:
        close_distributed()


if __name__ == "__main__":
    raise SystemExit(main())

