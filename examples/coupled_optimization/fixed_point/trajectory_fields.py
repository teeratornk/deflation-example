"""Plot a verified fixed-control trajectory using the unchanged physical source."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from deflation_example.coupled_targets import desired_temperature
from deflation_example.coupled_figures import plot_fields
from deflation_example.coupled_pilot import transformer_inputs
from deflation_example.fixed_point_study import prepare, verify_restart
from deflation_example.reporting import file_sha256, write_report


def verified_arrays(record, folder):
    if record["status"] != "converged" or len(record["steps"]) != record["slabs"]:
        raise ValueError("A field figure requires a complete verified trajectory")
    arrays = {key: [] for key in ("state", "velocity", "pressure")}
    for index, row in enumerate(record["steps"]):
        if not row["verified"] or row["slab_zero_based"] != index:
            raise ValueError("Every displayed trajectory step must be verified and ordered")
        path = folder / row["fields"]
        if path.resolve().parent != folder.resolve() or file_sha256(path) != row["field_sha256"]:
            raise ValueError("Trajectory field checksum or location differs")
        with np.load(path, allow_pickle=False) as data:
            for key in arrays:
                if not np.isfinite(data[key]).all():
                    raise ValueError("A field figure requires finite arrays")
                arrays[key].append(data[key].copy())
    return {key: np.stack(value) for key, value in arrays.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-indices", type=int, nargs="+")
    parser.add_argument("--format", choices=("pdf", "png"), default="pdf")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads((Path(__file__).parent / "protocol.json").read_text())
    record = json.loads((args.trajectory / "record.json").read_text())
    fields = verified_arrays(record, args.trajectory)
    if record["optimization_field_sha256"] != protocol["optimization_field_sha256"]:
        raise ValueError("The saved control differs from the declared study")
    inputs = SimpleNamespace(
        baseline=args.data_root / protocol["baseline"],
        fine_baseline=None
        if record["fine_baseline_sha256"] == record["baseline_sha256"]
        else args.data_root / protocol["fine_baseline"],
        optimization=args.data_root / protocol["optimization"],
        target_position=None,
        policy=record["policy"],
        family="forward",
        repetition=record["repetition"],
        threads=record["threads"],
    )
    with threadpool_limits(4):
        problem, controls, metadata = prepare(inputs, record["subdivision"])
        metadata["time_scheme"] = "backward_euler"
        verify_restart(record, metadata, args.trajectory)
        cfg = record["configuration"]
        fields["control"] = controls
        fields["desired"] = desired_temperature(
            problem, cfg["query"], cfg["target_count"], cfg.get("target_startup_s", 0)
        )
        # prepare() verifies the original physical inputs and source transfer.
        baseline = json.loads((inputs.baseline / "record.json").read_text())
        physical = transformer_inputs(baseline["configuration"]["level"])[1]["physical"]
        source_scale = (
            physical["oil_conductivity_W_m_K"]
            * physical["temperature_scale_K"]
            / physical["length_scale_m"] ** 2
        )
        args.output.mkdir(parents=True, exist_ok=False)
        description = plot_fields(
            problem,
            fields,
            source_scale,
            cfg["upper_K"],
            args.output / f"trajectory-fields.{args.format}",
            args.time_indices,
        )
        write_report(
            args.output / "figure.json",
            {
                **description,
                "trajectory_record_sha256": file_sha256(args.trajectory / "record.json"),
                "optimization_field_sha256": record["optimization_field_sha256"],
                "policy": record["policy"],
                "scope": "Complete fixed-source forward trajectory at stored discrete time levels. Source fields are unchanged. This figure establishes neither a new optimizer nor spatial or temporal resolution.",
            },
        )


if __name__ == "__main__":
    main()
