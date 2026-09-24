"""Four-way nonlinear optimization after the bounded quadratic screening gate.

The selected regularization and reference rank come from the verified screen.
Every arm starts from the same assessed temperature, with empty numerical
histories. This phase establishes its own nonlinear accuracy and complete cost.
"""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from .coupled_regularization import ALPHAS, RANKS, TARGETS, configuration
from .reporting import file_sha256, write_report
from .validation import integer


def selected_settings(summary, digest):
    """Freeze one four-arm development comparison; never infer missing evidence."""
    groups = summary.get("groups", [])
    if (
        summary.get("schema") != "coupled-regularization-summary-v1"
        or summary.get("decision") != "evaluate_one_complete_optimization_comparison"
        or summary.get("all_alpha_attempts_finished") is not True
        or len(groups) != len(ALPHAS)
        or {g.get("alpha") for g in groups} != set(ALPHAS)
    ):
        raise ValueError("A completed bounded screen must select the next comparison")
    candidates = []
    for group in groups:
        rows = group.get("rows", [])
        if (
            group.get("status") != "complete"
            or len(rows) != len(RANKS)
            or {row.get("rank") for row in rows} != set(RANKS)
            or any(row.get("repetitions") != 3 for row in rows)
        ):
            raise ValueError("Retain all declared alpha/rank repetitions before the next phase")
        for row in rows:
            if row.get("gate_passed"):
                if (
                    row.get("eligible_accuracy") is not True
                    or row.get("reasons")
                    or row.get("failed_cases")
                    or row["rank"] == 0
                    or not row.get("online_saving_fraction", -1) >= 0.05
                    or not 1
                    <= row.get("constant_cost_break_even_three_quadratic_sequences", 0)
                    <= 9
                ):
                    raise ValueError("The selected row must satisfy accuracy and amortization")
                candidates.append({"alpha": group["alpha"], **row})
    candidates.sort(
        key=lambda r: (-r["predicted_saving_over_nine_sequences_seconds"], r["rank"], r["alpha"])
    )
    if not candidates or summary.get("selected") != candidates[0]:
        raise ValueError("Use the selection made by the unchanged screen ordering")
    selected = candidates[0]
    rank = selected["rank"]
    arms = {}
    for name in ("jacobi", "frozen", "recycling", "reference"):
        use_space = name in {"recycling", "reference"}
        arms[name] = {
            "method": name if use_space else "jacobi",
            "rank": rank if use_space else 0,
            "recycle_window": rank if use_space else 1,
            "inner_preconditioner": "jacobi" if name == "jacobi" else "frozen",
            "reference_selection": "krylov_coupled" if name == "reference" else "thermal",
        }
    return {
        "schema": "coupled-regularization-complete-settings-v1",
        "phase": "development",
        "screen_sha256": digest,
        "alpha": selected["alpha"],
        "targets": list(TARGETS),
        "repetitions": 3,
        "arms": arms,
        "device": "cpu",
        "threads": 8,
        "frozen_sweeps": 3,
        "reference_krylov_steps": 48,
        "reference_krylov_seed": 20260923,
        "initial_state_alpha_policy": "shared_temperature",
        "sequence_time_limit_hours": 48,
        "maximum_concurrent_sequences": 4,
        "scope": "Complete three-target nonlinear optimization from a common assessed temperature; original residual, coupled equations and nonlinear KKT criteria remain unchanged. Prior optimization and common calibration are separate. Reference construction is included once per sequence.",
        "recycling_policy": "Retain deployed coarse vectors together with the last rank new search directions; select with the current velocity-frozen inverse in the operator energy metric, transfer by zero extension between inactive sets, and update only from successful inner solves. No artificial replacement directions.",
        "confirmation_policy": "Retain every development outcome. Five new complete repetitions per arm require verified full-population solution agreement and lower reference median time than the fastest tested alternative. A capped baseline supplies no completed-solve speedup.",
    }


def complete_configuration(record, settings, arm, repetition):
    """Keep physical inputs and accuracy fixed while changing declared solver choices."""
    if not record.get("all_problems_verified") or len(record.get("cases", [])) != 1:
        raise ValueError("Use the verified nominal single-target optimization input")
    if arm not in settings["arms"]:
        raise ValueError("Choose one of the four declared solver arms")
    repeat = integer(repetition, "Development repetition", 0)
    if repeat >= settings["repetitions"]:
        raise ValueError("The development population has three repetitions")
    cfg = configuration({"configuration": record["configuration"]}, settings["alpha"])
    cfg.update(
        **settings["arms"][arm],
        device=settings["device"],
        threads=settings["threads"],
        frozen_sweeps=settings["frozen_sweeps"],
        reference_krylov_steps=settings["reference_krylov_steps"],
        reference_krylov_seed=settings["reference_krylov_seed"],
        initial_state_alpha_policy=settings["initial_state_alpha_policy"],
        regularization_screen_sha256=settings["screen_sha256"],
        queries=[{"target": target, "upper_K": 357.3} for target in settings["targets"]],
        repetition=repeat,
        stage=None,
        capture_linear_systems=False,
        warm_start=True,
        evaluation_progress=True,
        linear_progress=True,
    )
    return cfg


def checked_settings(screen, path=None):
    expected = selected_settings(json.loads(screen.read_text()), file_sha256(screen))
    if path is not None and json.loads(path.read_text()) != expected:
        raise ValueError("Complete settings differ from the frozen screening decision")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("settings")
    freeze.add_argument("--screen", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    for name in (
        "screen",
        "settings",
        "optimization",
        "baseline",
        "initial-snapshot",
        "initial-assessment",
        "output",
    ):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--arm", choices=("jacobi", "frozen", "recycling", "reference"), required=True)
    run.add_argument("--repetition", type=int, required=True)
    args = parser.parse_args()
    if args.command == "settings":
        settings = checked_settings(args.screen)
        if args.output.exists():
            raise FileExistsError("Preserve the existing frozen complete-study settings")
        write_report(args.output, settings)
        return
    settings = checked_settings(args.screen, args.settings)
    record_path = args.optimization / "record.json"
    cfg = complete_configuration(
        json.loads(record_path.read_text()), settings, args.arm, args.repetition
    )
    cfg.update(
        baseline_directory=str(args.baseline),
        output=str(args.output),
        initial_state_snapshot=str(args.initial_snapshot),
        initial_state_assessment=str(args.initial_assessment),
        regularization_complete_settings_sha256=file_sha256(args.settings),
        starting_optimization_record_sha256=file_sha256(record_path),
    )
    from .coupled_sequence import run as run_sequence

    report = run_sequence(OmegaConf.create(cfg))
    if not report["all_problems_verified"]:
        raise RuntimeError("The complete outcome is retained but fails final verification")


if __name__ == "__main__":
    main()
