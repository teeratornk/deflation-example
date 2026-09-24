"""Finish declared regularization repetitions without replacing original attempts."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_regularization import RANKS, TARGETS, configuration, solve_quadratic
from .coupled_optimize import equation_acceptance, load_problem
from .coupled_sequence import RestoredEvaluation, prepare_device
from .coupled_targets import desired_temperature
from .coupled_trace import read_arrays, read_manifest
from .reporting import environment, file_sha256, write_arrays, write_report
from .study_solvers import ArrayReference, StudySolver
from .validation import integer


SCHEMA = "coupled-regularization-continuation-v1"
RUNTIME_FIELDS = ("python", "system", "machine", "cpu_model", "numpy", "scipy", "blas")


def declared_sequences():
    return [(rank, rep) for rep in range(3) for rank in RANKS[rep:] + RANKS[:rep]]


def read_original(directory):
    record = json.loads((Path(directory) / "record.json").read_text())
    if record.get("schema") != "coupled-regularization-screen-v1":
        raise ValueError("Use an original bounded regularization screen")
    if (
        configuration({"configuration": record["configuration"]}, record["alpha"])
        != record["configuration"]
    ):
        raise ValueError("The original CPU configuration changed")
    if (
        record["ranks"] != list(RANKS)
        or record["targets"] != list(TARGETS)
        or record["repetitions"] != 3
    ):
        raise ValueError("The original population differs from the declared protocol")
    missing_sequences(record)
    return record


def missing_sequences(record):
    present = [(row["rank"], row["repetition"]) for row in record["sequences"]]
    expected = declared_sequences()
    if len(set(present)) != len(present) or any(key not in expected for key in present):
        raise ValueError("Unexpected or duplicated original sequence")
    return [key for key in expected if key not in present]


def verify_environment(original, current):
    """All original Python modules must remain byte-identical; new drivers are separate."""
    if (
        original.get("source_tree_clean") is not True
        or current.get("source_tree_clean") is not True
    ):
        raise ValueError("Both sources must be clean")
    if any(original.get(k) != current.get(k) or original.get(k) is None for k in RUNTIME_FIELDS):
        raise ValueError("Continuation requires matching CPU hardware and numerical runtime")
    prior, actual = original.get("source_sha256", {}), current.get("source_sha256", {})
    required = {
        "coupled_regularization.py",
        "coupled_optimizer.py",
        "study_solvers.py",
        "solvers.py",
    }
    if not required <= prior.keys() or any(actual.get(k) != value for k, value in prior.items()):
        raise ValueError("An original numerical source module changed")
    return {"original_module_count": len(prior), "all_original_modules_identical": True}


def solve_sequence(problem, evaluation, targets, basis, rank, repetition, cfg, construction):
    """The original three-target timing boundary, with fresh solver resources."""
    lower = (337.3 - problem.temperature_offset) / problem.temperature_scale
    upper = (357.3 - problem.temperature_offset) / problem.temperature_scale
    tick = time.perf_counter()
    selected = None if rank == 0 else ArrayReference(basis[:, :rank], {"requested_rank": rank})
    solver = StudySolver(
        "jacobi" if rank == 0 else "reference",
        reference=selected,
        rank=rank,
        window=max(1, rank),
        rtol=cfg["inner_tolerance"],
        maxiter=cfg["inner_cap"],
        refresh=cfg["inner_refresh"],
        cg_factor=0.1,
        residual_policy="refine",
    )
    sequence = {"rank": rank, "repetition": repetition, "cases": []}
    states = []
    try:
        for target, desired in zip(TARGETS, targets, strict=True):
            state, row = solve_quadratic(problem, evaluation, desired, lower, upper, solver, cfg)
            row["target"] = target
            sequence["cases"].append(row)
            states.append(state)
    finally:
        solver.close()
    solver = selected = None
    gc.collect()
    sequence["online_seconds"] = time.perf_counter() - tick
    sequence["construction_seconds_once"] = construction if rank else 0.0
    sequence["setup_inclusive_model_seconds"] = (
        sequence["online_seconds"] + sequence["construction_seconds_once"]
    )
    sequence["verified"] = all(row["verified"] for row in sequence["cases"])
    return sequence, np.stack(states)


def run(args):
    original = read_original(args.original)
    rank, repetition = integer(args.rank, "Rank"), integer(args.repetition, "Repetition")
    if (rank, repetition) not in missing_sequences(original):
        raise ValueError(
            "Continue only an unrecorded declared repetition; retain every completed outcome"
        )
    if original["status"] not in {"running", "time_limit"}:
        raise ValueError(
            "Continue a time-limited attempt, not a numerical error or completed population"
        )
    current_environment = environment()
    compatibility = verify_environment(original["environment"], current_environment)
    cfg = dict(original["configuration"], baseline_directory=str(args.baseline))
    manifest = read_manifest(args.trace)
    if file_sha256(args.trace / "manifest.json") != original["trace_sha256"]:
        raise ValueError("The initial trace changed")
    first = manifest["quadratics"][0]
    if first["sha256"] != original["initial_trajectory_sha256"] or (
        first["iteration"],
        first["attempt"],
        first["damping"],
    ) != (0, 0, 0):
        raise ValueError("Use the original undamped initial trajectory")
    args.output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": SCHEMA,
        "status": "preparing",
        "environment": current_environment,
        "original_record_sha256": file_sha256(args.original / "record.json"),
        "alpha": original["alpha"],
        "configuration": original["configuration"],
        "baseline_sha256": original["baseline_sha256"],
        "trace_sha256": original["trace_sha256"],
        "initial_trajectory_sha256": original["initial_trajectory_sha256"],
        "reference_sha256": original["reference"]["sha256"],
        "rank": rank,
        "repetition": repetition,
        "state_dofs": original["state_dofs"],
        "source_compatibility": compatibility,
        "sequences": [],
        "scope": "One missing three-quadratic repetition in a fresh process. Unchanged numerical kernels and saved full-domain reference. No optimizer history or previously restricted solution is imported.",
        "timing_scope": "Online time preserves the original three-quadratic boundary. Original reference construction is charged once to each nonzero-rank model total. Restart preparation and reference-file loading are separate measured costs; this is not a complete nonlinear optimization timing.",
    }
    write_report(args.output / "record.json", record)
    sampler = None
    try:
        tick = time.perf_counter()
        problem, baseline = load_problem(cfg)
        if (
            baseline["baseline_sha256"] != original["baseline_sha256"]
            or problem.size != original["state_dofs"]
        ):
            raise ValueError("The physical baseline or dimension changed")
        sampler, _, _, _ = prepare_device("cpu", 0.01)
        sampler.start()
        arrays = read_arrays(args.trace, first["file"], first["sha256"])
        guess = RestoredEvaluation(arrays["state"], arrays["velocity"], arrays["pressure"])
        evaluation = problem.evaluate(arrays["state"], initial=guess)
        checks = problem.verify(evaluation, local_mass=True)
        if not equation_acceptance(checks, cfg):
            raise ValueError("The reconstructed coupled equations fail verification")
        lower = (337.3 - problem.temperature_offset) / problem.temperature_scale
        upper = (357.3 - problem.temperature_offset) / problem.temperature_scale
        if np.any(evaluation.state < lower) or np.any(evaluation.state > upper):
            raise ValueError("The unchanged initial trajectory violates its bounds")
        del arrays, guess
        reference = read_arrays(
            args.original, original["reference"]["file"], original["reference"]["sha256"]
        )["basis"]
        if (
            reference.shape != (problem.size, original["reference"]["description"]["deployed_rank"])
            or not np.isfinite(reference).all()
        ):
            raise ValueError("The saved full-domain reference has invalid dimensions or entries")
        targets = [
            desired_temperature(problem, target, cfg["target_count"], cfg["target_startup_s"])
            for target in TARGETS
        ]
        record.update(
            status="solving_sequence",
            nominal_equations=checks,
            restart_preparation_seconds=time.perf_counter() - tick,
        )
        write_report(args.output / "record.json", record)
        row, increments = solve_sequence(
            problem,
            evaluation,
            targets,
            reference,
            rank,
            repetition,
            cfg,
            original["reference"]["construction_seconds"],
        )
        filename = f"increments-r{rank}-repeat{repetition}.npz"
        write_arrays(args.output / filename, increments=increments)
        row["arrays"] = {"file": filename, "sha256": file_sha256(args.output / filename)}
        record["sequences"] = [row]
        record["status"] = "complete"
    except Exception as error:
        record.update(
            status="continuation_error", error_type=type(error).__name__, error_message=str(error)
        )
        raise
    finally:
        if sampler is not None:
            record["memory"] = sampler.finish()
        write_report(args.output / "record.json", record)


def read_population(directory, continuations=()):
    """Combine immutable evidence in memory and keep each execution source explicit."""
    directory = Path(directory)
    original = read_original(directory)
    digest = file_sha256(directory / "record.json")
    record = {**original, "sequences": list(original["sequences"])}
    origins = {(s["rank"], s["repetition"]): directory for s in record["sequences"]}
    provenance = []
    for folder in continuations:
        folder = Path(folder)
        path = folder / "record.json"
        addition = json.loads(path.read_text())
        if addition.get("schema") != SCHEMA or addition.get("original_record_sha256") != digest:
            raise ValueError("Continuation does not belong to this exact original record")
        verify_environment(original["environment"], addition["environment"])
        for key in (
            "alpha",
            "configuration",
            "baseline_sha256",
            "trace_sha256",
            "initial_trajectory_sha256",
            "state_dofs",
        ):
            if addition.get(key) != original.get(key):
                raise ValueError(f"Continuation changed {key}")
        if addition.get("reference_sha256") != original["reference"]["sha256"]:
            raise ValueError("Continuation changed the retained reference")
        key = (addition["rank"], addition["repetition"])
        if key not in missing_sequences(original):
            raise ValueError("A continuation cannot replace a recorded original outcome")
        if any((p["rank"], p["repetition"]) == key for p in provenance):
            raise ValueError("Duplicate continuation attempts must be reviewed separately")
        sequences = addition["sequences"]
        if addition["status"] == "complete":
            if len(sequences) != 1 or (sequences[0]["rank"], sequences[0]["repetition"]) != key:
                raise ValueError(
                    "A completed continuation must contain exactly its declared repetition"
                )
            if (
                not isinstance(addition.get("restart_preparation_seconds"), (int, float))
                or not np.isfinite(addition["restart_preparation_seconds"])
                or addition["restart_preparation_seconds"] < 0
            ):
                raise ValueError("Retain the measured restart preparation cost")
            origins[key] = folder
            record["sequences"].append(sequences[0])
        elif sequences:
            raise ValueError("Incomplete continuations must not supply completed timing rows")
        provenance.append(
            {
                "record_sha256": file_sha256(path),
                "source": addition["environment"]["git_head"],
                "status": addition["status"],
                "rank": key[0],
                "repetition": key[1],
                "restart_preparation_seconds": addition.get("restart_preparation_seconds"),
                "error_type": addition.get("error_type"),
                "error_message": addition.get("error_message"),
            }
        )
    if provenance:
        record["status"] = (
            "complete" if not missing_sequences(record) else "continuation_incomplete"
        )
    return record, origins, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "baseline", "trace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=RANKS, required=True)
    parser.add_argument("--repetition", type=int, choices=range(3), required=True)
    args = parser.parse_args()
    with threadpool_limits(8):
        run(args)


if __name__ == "__main__":
    main()
