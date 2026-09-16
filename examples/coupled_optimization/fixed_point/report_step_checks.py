"""Summarize every declared local correction, including failures and missing work."""

import argparse
import json
from pathlib import Path

import numpy as np

from deflation_example.coupled_saved import file_digest
from deflation_example.reporting import write_report

LABELS = {
    "equation_max": "Newton: equation-max",
    "fixed_scaled": "Newton: scaled residual",
    "temperature": "Anderson: temperature, ω=0.5",
    "coupled": "Anderson: all fields, ω=0.5",
    "relaxed": "Relaxation: ω=0.25",
    "anderson": "Anderson: temperature, ω=0.25",
}


def equation_residual(checks):
    values = [
        checks.get(k)
        for k in (
            "momentum_relative_residual",
            "continuity_relative_residual",
            "thermal_relative_residual",
        )
    ]
    if any(v is None or not np.isfinite(v) or v < 0 for v in values):
        return None
    return float(max(values))


def load_comparison(directories, allow_incomplete=False):
    rows, signature, fields, seen = [], None, [], set()
    for directory in directories:
        path = directory / "record.json"
        record = json.loads(path.read_text())
        key = record["policy"]
        if key not in LABELS or key in seen:
            raise ValueError("Expected one record for each declared policy")
        seen.add(key)
        identity = [
            record.get(k)
            for k in (
                "trajectory_record_sha256",
                "initial_field_sha256",
                "slab_zero_based",
                "time_s",
                "accuracy",
            )
        ]
        if any(v is None for v in identity):
            raise ValueError("Record lacks the matched-system identity")
        if signature is not None and identity != signature:
            raise ValueError("Local comparisons use different inputs or accuracy")
        signature = identity
        if record["accuracy"] != {"equation": 1e-12, "mass": 1e-6, "energy": 1e-6}:
            raise ValueError("Unexpected final accuracy criteria")
        if record["status"] not in {"complete", "execution_failure"} and not allow_incomplete:
            raise ValueError("Local comparison remains incomplete")
        checks = record.get("checks", {})
        residual = equation_residual(checks)
        verified = bool(record.get("verified", False))
        if verified and not (
            record["status"] == "complete"
            and record.get("termination") == "converged"
            and residual is not None
            and residual <= 1e-12
            and 0 <= checks.get("mass_relative_imbalance", np.inf) <= 1e-6
            and 0 <= checks.get("energy_relative_defect", np.inf) <= 1e-6
        ):
            raise ValueError("Successful label disagrees with original-equation checks")
        if verified:
            archive = directory / "fields.npz"
            if file_digest(archive) != record["field_sha256"]:
                raise ValueError("Returned field identity differs")
            with np.load(archive, allow_pickle=False) as data:
                arrays = {name: data[name].copy() for name in ("state", "velocity", "pressure")}
            if not all(np.isfinite(a).all() for a in arrays.values()):
                raise ValueError("Verified record has nonfinite fields")
            fields.append((key, arrays))
        curve = []
        for entry in record.get("history", []):
            iteration = entry.get("iteration", entry.get("coupling_iteration"))
            value = equation_residual(entry)
            if iteration is not None and value is not None:
                curve.append([iteration, value])
        rows.append(
            {
                "policy": key,
                "label": LABELS[key],
                "record_sha256": file_digest(path),
                "source": record["environment"].get("git_head"),
                "status": record.get("termination", record["status"]),
                "verified": verified,
                "maximum_equation_residual": residual,
                "checks": checks,
                "solve_seconds": record.get("solve_seconds"),
                "process_seconds": record.get("process_seconds"),
                "residual_history": curve,
            }
        )
    if seen != set(LABELS):
        raise ValueError("Retain all six declared comparison policies")
    differences = []
    if fields:
        name, baseline = fields[0]
        for label, arrays in fields[1:]:
            if any(arrays[k].shape != baseline[k].shape for k in arrays):
                raise ValueError("Compared field dimensions differ")
            differences.append(
                {
                    "baseline": name,
                    "candidate": label,
                    "maximum_dimensionless_temperature_difference": float(
                        np.max(np.abs(arrays["state"] - baseline["state"]))
                    ),
                    "maximum_velocity_vector_difference_m_s": float(
                        np.linalg.norm(arrays["velocity"] - baseline["velocity"], axis=1).max()
                    ),
                }
            )
    return {
        "scope": "Local correction at identical fixed data. Original failed-trajectory costs are excluded. This is not a complete-trajectory or optimization speedup.",
        "matched_identity": signature,
        "rows": rows,
        "verified_field_differences": differences,
        "incomplete": any(row["status"] == "running" for row in rows),
    }


def plot(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    colors = dict(
        zip(LABELS, ("#0072B2", "#009E73", "#CC79A7", "#E69F00", "#D55E00", "#56B4E9"), strict=True)
    )
    for row in report["rows"]:
        axis = axes[0] if row["policy"] in {"equation_max", "fixed_scaled"} else axes[1]
        history = np.asarray(row["residual_history"])
        color = colors[row["policy"]]
        if history.size:
            axis.semilogy(
                history[:, 0],
                np.maximum(history[:, 1], 1e-17),
                label=row["label"],
                color=color,
                linestyle="--" if row["policy"] == "fixed_scaled" else "-",
            )
            axis.plot(
                history[-1, 0],
                max(history[-1, 1], 1e-17),
                "o" if row["verified"] else "x",
                color=color,
            )
        else:
            axis.plot([], [], label=row["label"] + " — " + row["status"], color=color)
    for axis, title in zip(
        axes, ("Monolithic Newton correction", "Fixed-point correction"), strict=True
    ):
        axis.set_yscale("log")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.axhline(
            1e-12, color="black", linestyle=":", linewidth=1, label="Final equation target"
        )
        axis.set(
            title=title,
            xlabel="Outer iteration",
            ylabel="Maximum original-equation residual",
            ylim=(1e-13, 1e-2),
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Partial local comparison"
        if report["incomplete"]
        else f"Matched local correction at {report['matched_identity'][3]:g} s"
    )
    figure.savefig(output / "local-correction.pdf")
    figure.savefig(output / "local-correction.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs=6, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    report = load_comparison(args.records, args.allow_incomplete)
    args.output.mkdir(parents=True, exist_ok=False)
    report["reporter_sha256"] = file_digest(Path(__file__))
    write_report(args.output / "summary.json", report)
    plot(report, args.output)


if __name__ == "__main__":
    main()
