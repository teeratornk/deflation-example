"""Reproduce the fixed-physics coupled reference-retention diagnostics."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf


def comparison_configuration(args, *, capture=False):
    record = json.loads((args.optimization / "record.json").read_text())
    if not record.get("all_problems_verified") or len(record["cases"]) != 1:
        raise ValueError("Capture requires the verified single-target comparison")
    config = dict(record["configuration"])
    config.update(
        baseline_directory=str(args.baseline),
        output=str(args.output),
        initial_state_snapshot=str(args.initial_snapshot),
        initial_state_assessment=str(args.initial_assessment),
        method="jacobi",
        rank=0,
        recycle_window=1,
        capture_linear_systems=capture,
        reference_selection="thermal",
        reference_candidates=400,
        reference_selection_tolerance=1e-12,
        stage={"positions": [0], "restore": None, "resume": None},
    )
    if not capture:
        if args.policy == "jacobi" and args.rank != 0:
            raise ValueError("Jacobi comparisons require rank zero")
        if args.policy != "jacobi" and args.rank not in {20, 50, 100, 200}:
            raise ValueError("Choose a declared nonzero rank")
        config.update(
            method="reference"
            if args.policy in {"thermal", "nominal_coupled", "preconditioned_coupled"}
            else args.policy,
            rank=args.rank,
            recycle_window=max(1, args.rank),
            reference_selection=args.policy
            if args.policy in {"nominal_coupled", "preconditioned_coupled"}
            else "thermal",
            hybrid_block_max_columns=args.width,
            repetition=args.repetition,
            inner_preconditioner=getattr(args, "preconditioner", "jacobi"),
            frozen_sweeps=getattr(args, "sweeps", 3),
        )
        targets = getattr(args, "targets", None)
        if targets is not None:
            from .validation import integer

            targets = [integer(t, "Target", 0) for t in targets]
            if (
                not targets
                or len(set(targets)) != len(targets)
                or max(targets) >= config["target_count"]
            ):
                raise ValueError("Declare distinct targets from the existing population")
            if targets[0] != record["configuration"]["queries"][0]["target"]:
                raise ValueError("The sequence must start with the declared nominal target")
            config["queries"] = [
                {"target": target, "upper_K": record["configuration"]["queries"][0]["upper_K"]}
                for target in targets
            ]
            config["stage"] = None
    return config


def execute(args):
    from .coupled_sequence import run

    config = comparison_configuration(args, capture=args.command == "capture")
    report = run(OmegaConf.create(config))
    if not report["all_problems_verified"]:
        raise RuntimeError("The complete outcome is preserved but fails final verification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "run"):
        p = commands.add_parser(name)
        p.add_argument("--baseline", type=Path, required=True)
        p.add_argument("--optimization", type=Path, required=True)
        p.add_argument("--initial-snapshot", type=Path, required=True)
        p.add_argument("--initial-assessment", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if name == "run":
            p.add_argument(
                "--policy",
                choices=(
                    "jacobi",
                    "thermal",
                    "nominal_coupled",
                    "preconditioned_coupled",
                    "recycling",
                ),
                required=True,
            )
            p.add_argument("--rank", type=int, required=True)
            p.add_argument("--width", type=int, choices=(1, 5, 10, 20), default=5)
            p.add_argument("--repetition", type=int, default=0)
            p.add_argument("--preconditioner", choices=("jacobi", "frozen"), default="jacobi")
            p.add_argument("--sweeps", type=int, choices=(1, 3, 5), default=3)
            p.add_argument(
                "--targets",
                nargs="+",
                type=int,
                help="Fresh complete sequence starting with the nominal target; one reference construction per sequence",
            )
        p.set_defaults(function=execute)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
