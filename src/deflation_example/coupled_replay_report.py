"""Compare fixed-source replay tolerances without hiding incomplete attempts."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .coupled_saved import file_digest
from .reporting import write_report


def summarize(optimization_fields, replay_directories, temperature_scale):
    """Read a common saved trajectory and retain each declared replay status."""
    scale = float(temperature_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Temperature scale must be finite and positive")
    digest = file_digest(optimization_fields)
    with np.load(optimization_fields, allow_pickle=False) as data:
        original = data["state"].copy()
    if not original.size or not np.isfinite(original).all():
        raise ValueError("Saved temperature must be nonempty and finite")
    rows, traces, configurations = [], [], []
    for directory in replay_directories:
        directory = Path(directory)
        path = directory / "record.json"
        if not path.exists():
            rows.append({"case": directory.name, "status": "missing", "complete": False})
            continue
        record = json.loads(path.read_text())
        if record.get("optimization_field_sha256") != digest or record.get("subdivision") != 1:
            raise ValueError("Every replay must use the same saved source and original time grid")
        config = record["configuration"]
        if configurations and config != configurations[0]:
            raise ValueError("Replay physical and optimization configurations must match")
        configurations.append(config)
        slabs = int(config["slabs"])
        if slabs < 1 or original.size % slabs:
            raise ValueError("Saved trajectory shape differs from the replay grid")
        expected = original.reshape(slabs, -1)
        row = {
            "case": directory.name,
            "status": record["status"],
            "complete": False,
            "record_sha256": file_digest(path),
            "source": record["environment"]["git_head"],
            "forward_solver": record["forward_solver"],
            "seconds": record.get("seconds"),
        }
        fields = directory / "states.npz"
        if fields.exists():
            with np.load(fields, allow_pickle=False) as data:
                state, times = data["state"].copy(), data["times_s"].copy()
            if (
                state.ndim != 2
                or state.shape[1] != expected.shape[1]
                or len(state) > slabs
                or not len(state)
                or times.shape != (len(state),)
                or not np.isfinite(state).all()
                or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)
            ):
                raise ValueError("Replay fields must contain finite ordered original time steps")
            declared_times = np.asarray([step["time_s"] for step in record["steps"]])
            if declared_times.shape != times.shape or not np.allclose(
                times, declared_times, rtol=1e-12, atol=1e-12
            ):
                raise ValueError("Replay field times differ from the recorded steps")
            error = np.max(np.abs(state - expected[: len(state)]), axis=1) * scale
            complete = (
                record["status"] == "converged"
                and len(state) == slabs
                and all(step["status"] == "converged" for step in record["steps"])
            )
            if record["status"] == "converged" and not complete:
                raise ValueError("A converged replay must contain every converged time step")
            row.update(
                complete=complete,
                recorded_slabs=len(state),
                maximum_recorded_difference_K=float(error.max()),
                maximum_endpoint_difference_K=float(error.max()) if complete else None,
                field_sha256=file_digest(fields),
            )
            if complete and not np.isclose(
                row["maximum_endpoint_difference_K"],
                record["maximum_endpoint_difference_K"],
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(
                    "Recomputed temperature difference disagrees with the replay report"
                )
            traces.append(
                {
                    "case": directory.name,
                    "complete": complete,
                    "times_s": times,
                    "maximum_difference_K": error,
                }
            )
        elif record["status"] == "converged":
            raise ValueError("Converged replay is missing its temperature fields")
        rows.append(row)
    if not rows or len({row["case"] for row in rows}) != len(rows):
        raise ValueError("Declare distinct replay directories")
    return {
        "schema": "coupled-replay-tolerance-summary-v1",
        "optimization_field_sha256": digest,
        "temperature_scale_K": scale,
        "scope": "Same-grid fixed-source replay; every supplied attempt is retained. Residual convergence and trajectory agreement are separate quantities.",
        "rows": rows,
        "traces": traces,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimization-fields", type=Path, required=True)
    parser.add_argument("--replays", type=Path, nargs="+", required=True)
    parser.add_argument("--temperature-scale", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    report = summarize(args.optimization_fields, args.replays, args.temperature_scale)
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "summary.json", report)
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "case",
                "status",
                "complete",
                "seconds",
                "recorded_slabs",
                "maximum_endpoint_difference_K",
            ],
        )
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in report["rows"])
    if args.plot:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(6.4, 3.8), layout="constrained")
        for trace in report["traces"]:
            row = next(r for r in report["rows"] if r["case"] == trace["case"])
            label = f"Tolerance {row['forward_solver']['tolerance']:.0e} ({row['status']})"
            axis.plot(
                trace["times_s"],
                trace["maximum_difference_K"],
                label=label,
                linestyle="-" if trace["complete"] else "--",
            )
        axis.set(
            xlabel="Physical time (s)", ylabel="Maximum temperature difference (K)", yscale="symlog"
        )
        axis.set_yscale("symlog", linthresh=1e-10)
        axis.grid(alpha=0.25)
        if report["traces"]:
            axis.legend(fontsize=8)
        for extension in ("pdf", "png"):
            figure.savefig(args.output / f"trajectory_difference.{extension}", dpi=220)
        plt.close(figure)


if __name__ == "__main__":
    main()
