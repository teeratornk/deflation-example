"""Audit declared outcomes and plot complete comparisons, including failures.

Run on a compute node. This postprocessing does not alter the numerical source,
selection, controls or records. A missing record stays missing, and a failed
solve never becomes a completed-solve timing ratio.
"""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np

from deflation_example.reporting import atomic_output, file_sha256, write_report

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def declared_records(protocol, selection=None):
    for family, case, policy in itertools.product(
        protocol["families"], protocol["cases"], protocol["policies"]
    ):
        yield (
            f"screen/{family}/{case['name']}/{policy}/record.json",
            {"phase": "screen", "group": f"{family}/{case['name']}", "policy": policy},
        )
    if selection is None:
        return
    forward = selection["families"]["forward"]["selected"]
    momentum = selection["families"]["momentum"]["selected"]
    if forward is not None:
        for grid, policy, rep in itertools.product(
            protocol["trajectory_grids"],
            ("newton", forward),
            range(protocol["trajectory_repetitions"]),
        ):
            mesh = grid["mesh"]
            yield (
                f"trajectory/{mesh}/{policy}/rep-{rep}/record.json",
                {"phase": "trajectory", "group": mesh, "policy": policy, "repetition": rep},
            )
    if momentum is not None:
        for policy in ("newton", momentum):
            yield (
                f"derivatives/{policy}/derivatives/record.json",
                {"phase": "derivatives", "group": "derivatives", "policy": policy},
            )
        followup = protocol["optimizer_followup"]
        for method, policy, rep in itertools.product(
            followup["methods"], ("newton", momentum), range(followup["repetitions"])
        ):
            yield (
                f"optimize/{policy}/{method}/rep-{rep}/record.json",
                {"phase": "optimize", "group": method, "policy": policy, "repetition": rep},
            )


def duration(value):
    if value is None:
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("A measured duration must be finite and nonnegative")
    return float(value)


def bounded_metric(value, limit, name):
    if (
        not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or value > limit
    ):
        raise ValueError(f"Verified record violates {name}: {value!r}")
    return float(value)


def equation_metrics(checks, tolerance, *, momentum_only=False):
    """Check stored independent evaluations; this does not rerun the PDE."""
    names = ["momentum_relative_residual", "continuity_relative_residual"]
    if not momentum_only:
        names.append("thermal_relative_residual")
    maximum = max(bounded_metric(checks.get(k), tolerance, k) for k in names)
    if not momentum_only:
        bounded_metric(checks.get("mass_relative_imbalance"), 1e-6, "mass balance")
        energy = checks.get(
            "energy_relative_defect", checks.get("energy", {}).get("relative_defect")
        )
        bounded_metric(energy, 1e-6, "energy balance")
    return maximum


def verified_metrics(record, phase):
    """Refuse verified labels inconsistent with their recorded numerical values."""
    if phase in {"screen", "trajectory"}:
        steps = [record["row"]] if phase == "screen" else record["steps"]
        maximum = max(
            equation_metrics(
                step["checks"],
                1e-12,
                momentum_only=phase == "screen" and record["family"] == "momentum",
            )
            for step in steps
        )
        return {"maximum_equation_relative_residual": maximum}
    if phase == "derivatives":
        report = record["derivatives"]
        bounded_metric(report.get("relative_dot_product_error"), 1e-9, "adjoint dot product")
        orders = report.get("taylor_orders", [])
        if not orders or any(not math.isfinite(v) or v <= 1.9 for v in orders):
            raise ValueError("Verified record violates the Taylor-order criterion")
        equations = report["equations"]
    else:
        cfg = record["configuration"]
        if cfg["nonlinear_tolerance"] != 1e-8 or cfg["inner_tolerance"] != 1e-10:
            raise ValueError("Optimization tolerances differ from the declared protocol")
        equations = []
        components = (
            "primal_absolute",
            "stationarity",
            "dual_feasibility",
            "lower_complementarity",
            "upper_complementarity",
        )
        kkt_maximum = 0.0
        for case in record["cases"]:
            if len(case["equations"]) != cfg["slabs"]:
                raise ValueError("Verified optimization omits a declared time slab")
            equations.extend(case["equations"])
            for name in components:
                kkt_maximum = max(kkt_maximum, bounded_metric(case["kkt"].get(name), 1e-8, name))
            bounded_metric(
                case["adjoint"].get("maximum_momentum_adjoint_relative_residual"),
                1e-8,
                "momentum adjoint",
            )
    if not equations:
        raise ValueError("Verified record has no independently evaluated equations")
    result = {
        "maximum_equation_relative_residual": max(equation_metrics(r, 1e-8) for r in equations)
    }
    if phase == "optimize":
        result["maximum_kkt_component"] = kkt_maximum
    return result


def outcome(path, spec):
    row = {**spec, "status": "missing", "verified": False, "seconds": None}
    if not path.exists():
        return row, None
    record = read(path)
    row.update(record_sha256=file_sha256(path), status=record["status"])
    phase = spec["phase"]
    if phase == "screen":
        local = record.get("row", {})
        row.update(
            status=local.get("status", record["status"]),
            verified=record["status"] == "complete" and local.get("verified", False),
            seconds=duration(local.get("seconds")),
        )
    elif phase == "derivatives":
        row["verified"] = record["status"] == "verified"
    elif phase == "trajectory":
        steps = record.get("steps", [])
        row.update(
            verified=record["status"] == "converged"
            and len(steps) == record["slabs"]
            and all(s["verified"] for s in steps),
            seconds=duration(record.get("all_attempt_process_seconds")),
            sum_step_seconds=duration(record.get("sum_step_seconds")),
            completed_steps=sum(s["verified"] for s in steps),
            declared_steps=record["slabs"],
            quantities=record.get("quantities", {}),
        )
        if "sum_step_seconds" in record and not math.isclose(
            sum(duration(s["seconds"]) for s in steps),
            record["sum_step_seconds"],
            rel_tol=1e-10,
            abs_tol=1e-8,
        ):
            raise ValueError("Trajectory step timers do not sum to the reported interval")
    elif phase == "optimize":
        cases = record.get("cases", [])
        cfg = record["configuration"]
        row.update(
            verified=record["status"] == "complete"
            and record.get("all_problems_verified", False)
            and len(cases) == len(cfg["queries"])
            and all(c["verified"] for c in cases),
            seconds=duration(record.get("sequence_seconds")),
            preparation_inclusive_seconds=duration(record.get("preparation_inclusive_seconds")),
            case_statuses=[c["status"] for c in cases],
            objectives=[c.get("objective") for c in cases],
            kkt=[c.get("kkt") for c in cases],
            inner_iterations=sum(c.get("inner_iterations", 0) for c in cases),
            nonlinear_iterations=sum(c.get("nonlinear_iterations", 0) for c in cases),
            memory=record.get("memory", {}),
        )
        if row["seconds"] is not None:
            parts = record["components_seconds"]
            if any(v < -max(1e-8, 1e-10 * row["seconds"]) for v in parts.values()):
                raise ValueError("Optimizer component timers contain a negative interval")
            if any(not math.isfinite(v) for v in parts.values()) or not math.isclose(
                sum(parts.values()), row["seconds"], rel_tol=1e-10, abs_tol=1e-8
            ):
                raise ValueError("Optimizer component timers do not sum to the reported interval")
        # Checkpoint work is known, but an interrupted attempt may have additional
        # unmeasured work. Do not report its stage interval as an all-attempt total.
        row["resumed"] = record.get("stage", {}).get("resume") is not None
        if row["resumed"]:
            row["seconds"] = None
            row["timing_status"] = "requires_complete_attempt_accounting"
    if row["verified"]:
        row.update(verified_metrics(record, phase))
    return row, record


def groups(rows):
    result = []
    keys = sorted({(r["phase"], r["group"], r["policy"]) for r in rows})
    for phase, group, policy in keys:
        subset = [
            r for r in rows if (r["phase"], r["group"], r["policy"]) == (phase, group, policy)
        ]
        times = [r["seconds"] for r in subset if r["verified"] and r["seconds"] is not None]
        complete = len(times) == len(subset)
        result.append(
            {
                "phase": phase,
                "group": group,
                "policy": policy,
                "declared": len(subset),
                "verified": sum(r["verified"] for r in subset),
                "statuses": [r["status"] for r in subset],
                "verified_seconds": times,
                "all_repetitions_verified_and_timed": complete,
                "median_seconds": statistics.median(times) if complete else None,
                "minimum_seconds": min(times) if complete else None,
                "maximum_seconds": max(times) if complete else None,
            }
        )
    return result


def checked_fields(path, expected):
    if file_sha256(path) != expected:
        raise ValueError("Field checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def root_agreement(root, reference_policy="newton"):
    """Compare only verified local roots with identical inputs and sources."""
    if reference_policy not in read(HERE / "protocol.json")["policies"]:
        raise ValueError("Choose a declared root-comparison policy")
    rows = []
    for reference_path in sorted(root.glob(f"screen/forward/*/{reference_policy}/record.json")):
        reference = read(reference_path)
        if not reference.get("row", {}).get("verified", False):
            continue
        base = checked_fields(reference_path.parent / "fields.npz", reference["field_sha256"])
        for path in sorted(reference_path.parent.parent.glob("*/record.json")):
            record = read(path)
            if record["policy"] == reference_policy or not record.get("row", {}).get(
                "verified", False
            ):
                continue
            for key in (
                "optimization_field_sha256",
                "baseline_sha256",
                "fine_baseline_sha256",
                "replay_fields_sha256",
                "time_s",
                "initial",
                "time_scheme",
                "temperature_scale_K",
                "temperature_offset_K",
            ):
                if reference[key] != record[key]:
                    raise ValueError("Root comparison has different matched inputs: " + key)
            if reference["environment"]["source_sha256"] != record["environment"]["source_sha256"]:
                raise ValueError("Root comparison has different numerical sources")
            data = checked_fields(path.parent / "fields.npz", record["field_sha256"])
            for key in ("state", "velocity", "pressure"):
                if (
                    data[key].shape != base[key].shape
                    or not np.isfinite(data[key]).all()
                    or not np.isfinite(base[key]).all()
                ):
                    raise ValueError("Root comparison requires matching finite fields")
            rows.append(
                {
                    "case": path.parent.parent.name,
                    "policy": record["policy"],
                    "reference_policy": reference_policy,
                    "maximum_temperature_difference_K": float(
                        record["temperature_scale_K"]
                        * np.max(np.abs(data["state"] - base["state"]))
                    ),
                    "maximum_velocity_component_difference_m_s": float(
                        np.max(np.abs(data["velocity"] - base["velocity"]))
                    ),
                    "maximum_kinematic_pressure_difference_m2_s2": float(
                        np.max(np.abs(data["pressure"] - base["pressure"]))
                    ),
                    "record_sha256": file_sha256(path),
                    "reference_record_sha256": file_sha256(reference_path),
                }
            )
    return rows


def plot_comparisons(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for phase, ylabel in (
        ("trajectory", "Complete forward-process time (s)"),
        ("optimize", "Complete optimization-sequence time (s)"),
    ):
        rows = [r for r in summary["groups"] if r["phase"] == phase]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(max(6, 0.85 * len(rows)), 4.2))
        for index, row in enumerate(rows):
            times = row["verified_seconds"]
            color = "#222222" if row["policy"] == "newton" else "#D55E00"
            if times:
                offsets = np.linspace(-0.08, 0.08, len(times)) if len(times) > 1 else [0]
                ax.scatter(index + np.asarray(offsets), times, color=color, s=28)
            if row["median_seconds"] is not None:
                ax.plot([index - 0.18, index + 0.18], [row["median_seconds"]] * 2, color=color)
            ax.text(
                index,
                1.02,
                f"{row['verified']}/{row['declared']} verified",
                ha="center",
                fontsize=8,
                transform=ax.get_xaxis_transform(),
            )
        ax.set_xticks(
            range(len(rows)),
            [f"{r['group']}\n{r['policy']}" for r in rows],
            rotation=25,
            ha="right",
        )
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(output / f"{phase}-complete-times.pdf")
        fig.savefig(output / f"{phase}-complete-times.png", dpi=160)
        plt.close(fig)


def audit(root, output, plots=True, fields=True, root_policy="newton"):
    protocol = read(HERE / "protocol.json")
    selection = read(root / "selection.json") if (root / "selection.json").exists() else None
    if selection is not None and selection["protocol_sha256"] != file_sha256(
        HERE / "protocol.json"
    ):
        raise ValueError("Selection belongs to a different protocol")
    rows, sources = [], set()
    for relative, spec in declared_records(protocol, selection):
        row, record = outcome(root / relative, spec)
        rows.append({"record": relative, **row})
        if record is not None:
            sources.add(json.dumps(record["environment"]["source_sha256"], sort_keys=True))
    if len(sources) > 1:
        raise ValueError("Declared study records contain different numerical sources")
    summary = {
        "schema": "fixed-point-complete-audit-v1",
        "protocol_sha256": file_sha256(HERE / "protocol.json"),
        "selection_complete": selection is not None,
        "rows": rows,
        "groups": groups(rows),
        "root_agreement": root_agreement(root, root_policy) if fields else [],
        "scope": "The local screen, fixed-control trajectories and nonlinear optimization are distinct populations. Medians require every declared repetition to be verified and timed. Sampled process allocations are lower bounds on instantaneous peak memory. Root agreement is distinct from physical resolution.",
        "timing_boundaries": {
            "screen": "One equation solve including verification.",
            "trajectory": "All measured forward-process attempts, including input/model preparation and step serialization; interpreter startup excluded. Unknown interrupted costs stay unavailable.",
            "optimize": "Reference, assembly, nonlinear and inner solves, transfers, verification and cleanup. Common calibration and process preparation are separate. Resumed results require a separate complete-attempt audit.",
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    write_report(output / "audit.json", summary)
    columns = sorted(set().union(*(r.keys() for r in rows)))
    with atomic_output(output / "outcomes.csv") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                for k, v in row.items()
            }
            for row in rows
        )
    if plots:
        plot_comparisons(summary, output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--root-policy",
        choices=read(HERE / "protocol.json")["policies"],
        default="newton",
        help="Verified local root used only for field agreement; this does not select a numerical policy",
    )
    args = parser.parse_args()
    audit(args.root, args.output, root_policy=args.root_policy)


if __name__ == "__main__":
    main()
