"""Predeclared complete-trajectory ablations at unchanged prescribed-flow physics.

Every worker constructs its own reference and solves the same three targets.
The nonlinear repair campaign is independent of this fixed-flow comparison.
"""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from .linear_spacetime_complete import linear_configuration, linear_settings
from .reporting import file_sha256, write_report
from .validation import integer


RANKS = (4, 8, 16, 32)
SELECTIONS = {"low": "lowest", "ends": "alternating_low_high"}
ARM_KEYS = {
    "method",
    "rank",
    "recycle_window",
    "inner_preconditioner",
    "reference_selection",
    "reference_transfer",
    "reference_krylov_selection",
}


def ablation_settings(screen):
    """Freeze rank, selection and transfer choices before measuring their outcomes."""
    settings = linear_settings(screen)
    if settings["alpha"] != 1e-11 or settings["targets"] != [7, 15, 14]:
        raise ValueError("Use the declared regularization and target sequence")
    arms = {}

    def arm(name, method, rank=0, selection="lowest", transfer="full"):
        arms[name] = {
            "method": method,
            "rank": rank,
            "recycle_window": max(1, rank),
            "inner_preconditioner": "jacobi" if name == "jacobi" else "frozen",
            "reference_selection": "krylov_coupled" if method == "reference" else "thermal",
            "reference_transfer": transfer,
            "reference_krylov_selection": selection,
        }

    arm("jacobi", "jacobi")
    arm("frozen", "jacobi")
    for rank in RANKS:
        for label, selection in SELECTIONS.items():
            arm(f"reference-{label}-r{rank}", "reference", rank, selection)
            arm(f"sequential-{label}-r{rank}", "reference", rank, selection, "sequential")
        arm(f"recycling-r{rank}", "recycling", rank)
    settings.update(
        schema="linear-spacetime-speedup-ablation-settings-v1",
        phase="ablation",
        arms=arms,
        reference_krylov_steps=96,
        sequence_time_limit_hours=24,
        maximum_concurrent_sequences=2,
        comparison_policy="Fresh complete prescribed-flow comparisons only. Vary rank, Ritz selection and reference transfer at fixed physics and accuracy. Retain all 22 configurations and all three repetitions; do not pool historical timings or fully coupled outcomes.",
        transfer_policy="Full restriction and sequential zero extension start from the same seeded full-domain construction. Sequential transfer follows every PDAS solve, including initially converged solves, and introduces no new directions. Adaptive recycling retains deployed coarse vectors and learns new directions; it is a separate comparator.",
        construction_policy="Independently construct the seeded 96-step energy-Krylov reference once in every reference or sequential sequence. Both selection rules use the same candidate construction. Report actual numerical rank; add no replacement vectors after rank loss.",
        speedup_policy="Compute complete-time ratios only for all three verified repetitions with state and objective agreement. Compare each full-reference setting with fresh Jacobi, frozen and rank-matched recycling controls. A failed or missing comparator yields no speedup. Ablation results require fresh confirmation before a publication claim.",
    )
    names = list(arms)
    settings["schedule"] = [
        {"arm": name, "repetition": rep}
        for rep in range(settings["repetitions"])
        for name in names[7 * rep :] + names[: 7 * rep]
    ]
    return settings


def checked_ablation_settings(screen, path):
    expected = ablation_settings(screen)
    if json.loads(path.read_text()) != expected:
        raise ValueError("Ablation settings differ from the predeclared population")
    return expected


def ablation_configuration(record, settings, task):
    index = integer(task, "Task index", 0)
    if index >= len(settings["schedule"]):
        raise ValueError("Task index exceeds the declared population")
    task = settings["schedule"][index]
    cfg = linear_configuration(record, settings, task["arm"], task["repetition"])
    cfg["speedup_ablation_arm"] = task["arm"]
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
    worker.add_argument("--task", type=int, required=True)
    args = parser.parse_args()
    if args.command == "settings":
        if args.output.exists():
            raise FileExistsError("Preserve the existing ablation settings")
        write_report(args.output, ablation_settings(args.screen))
        return
    settings = checked_ablation_settings(args.screen, args.settings)
    record_path = args.optimization / "record.json"
    cfg = ablation_configuration(json.loads(record_path.read_text()), settings, args.task)
    cfg.update(
        baseline_directory=str(args.baseline),
        output=str(args.output),
        initial_state_snapshot=str(args.initial_snapshot),
        initial_state_assessment=str(args.initial_assessment),
        speedup_ablation_settings_sha256=file_sha256(args.settings),
        starting_optimization_record_sha256=file_sha256(record_path),
    )
    from .linear_spacetime_sequence import run

    report = run(OmegaConf.create(cfg))
    if not report["all_problems_verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
