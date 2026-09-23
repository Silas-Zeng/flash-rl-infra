"""Command line entry points for the local vertical slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ablation import compare_modes
from .orchestrator import FlashRLRun
from .store import EventStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="flashrl")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the local rollout -> reward -> train flow")
    run.add_argument("--output", default="runs/local-demo")
    run.add_argument("--groups", type=int, default=4)
    run.add_argument("--group-size", type=int, default=2)
    inspect = sub.add_parser("inspect", help="inspect durable stage state")
    inspect.add_argument("--run-dir", default="runs/local-demo")
    inspect.add_argument("--prefix", default="")
    ablation = sub.add_parser("ablation", help="compare baseline with Flash-style data-path tricks")
    ablation.add_argument("--samples", type=int, default=64)
    ablation.add_argument("--seed", type=int, default=20260923)
    ablation.add_argument("--output", default="runs/ablation/comparison.json")
    args = parser.parse_args()
    if args.command == "run":
        workflow = FlashRLRun(args.output, args.groups, args.group_size)
        try:
            print(json.dumps(workflow.run(), indent=2))
        finally:
            workflow.close()
    elif args.command == "inspect":
        store = EventStore(Path(args.run_dir) / "events")
        try:
            print(json.dumps(store.inspect(args.prefix), indent=2))
        finally:
            store.close()
    else:
        print(json.dumps(compare_modes(samples=args.samples, seed=args.seed, output=args.output), indent=2))


if __name__ == "__main__":
    main()

