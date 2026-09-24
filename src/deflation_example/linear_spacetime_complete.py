"""Freeze or run the bounded four-way fixed-flow space-time comparison."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from .coupled_regularization_complete import checked_settings, complete_configuration
from .reporting import file_sha256, write_report


def linear_settings(screen, path=None):
    settings = checked_settings(screen)
    settings.update(
        schema="linear-spacetime-complete-settings-v1",
        physics="prescribed_flow",
        feedback_multiplier=0.0,
        scope="Complete three-target linear-quadratic trajectory optimization at the verified isothermal computed velocity. Consistent thermal source and storage actions, all temporal coupling and the initial temperature are retained. Independent residual, thermal, conservation and weighted KKT criteria match the nonlinear comparison. Reference construction is included once per sequence; common calibration and process preparation are separate.",
        comparison_policy="Compare the four solvers within prescribed-flow physics only. The alpha/rank choice is inherited from the bounded coupled screen, not reselected using the linear outcomes. Retain every development outcome. No ratio combines linear and nonlinear optimization times.",
        field_policy="Retain all targets and all time levels. For the 64-slab trajectory, plot targets 7 and 14 at levels 7, 32 and 64 (65.625, 300 and 600 seconds); levels count from one. Plot the optimized temperature, applied control and both active bounds.",
    )
    if path is not None and json.loads(path.read_text()) != settings:
        raise ValueError("Linear settings differ from the declared matched comparison")
    return settings


def linear_configuration(record, settings, arm, repetition):
    if settings.get("physics") != "prescribed_flow" or settings.get("feedback_multiplier") != 0:
        raise ValueError("The linear control requires zero thermal feedback")
    cfg = complete_configuration(record, settings, arm, repetition)
    cfg.update(physics="prescribed_flow", feedback_multiplier=0.0)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("settings")
    freeze.add_argument("--screen", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    worker = commands.add_parser("run")
    for name in (
        "screen",
        "settings",
        "optimization",
        "baseline",
        "initial-snapshot",
        "initial-assessment",
        "output",
    ):
        worker.add_argument("--" + name, type=Path, required=True)
    worker.add_argument(
        "--arm", choices=("jacobi", "frozen", "recycling", "reference"), required=True
    )
    worker.add_argument("--repetition", type=int, required=True)
    args = parser.parse_args()
    if args.command == "settings":
        if args.output.exists():
            raise FileExistsError("Preserve the existing linear settings")
        write_report(args.output, linear_settings(args.screen))
        return
    settings = linear_settings(args.screen, args.settings)
    record_path = args.optimization / "record.json"
    cfg = linear_configuration(
        json.loads(record_path.read_text()), settings, args.arm, args.repetition
    )
    cfg.update(
        baseline_directory=str(args.baseline),
        output=str(args.output),
        initial_state_snapshot=str(args.initial_snapshot),
        initial_state_assessment=str(args.initial_assessment),
        linear_spacetime_settings_sha256=file_sha256(args.settings),
        starting_optimization_record_sha256=file_sha256(record_path),
    )
    from .linear_spacetime_sequence import run

    report = run(OmegaConf.create(cfg))
    if not report["all_problems_verified"]:
        raise RuntimeError("The complete linear outcome is retained but fails verification")


if __name__ == "__main__":
    main()
