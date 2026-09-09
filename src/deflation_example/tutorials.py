"""Run versioned diagnostic tutorials with portable source and input identifiers."""

import argparse
import csv
import hashlib
from importlib.resources import files
import json
import math
from pathlib import Path

from threadpoolctl import threadpool_limits

from .reporting import atomic_output, environment, write_report
from .tutorial_costs import cost_examples
from .tutorial_spectral import spectral_examples
from .tutorial_verification import accuracy_examples, stopping_examples


class Records:
    """Read only within the supplied record directory and retain content hashes."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = {}

    def path(self, name):
        relative = Path(name)
        path = (self.root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or not path.is_relative_to(self.root)
        ):
            raise ValueError("Unsafe record path")
        return path

    def load(self, name):
        path = self.path(name)
        raw = path.read_bytes()
        value = json.loads(raw)
        self.manifest[name] = hashlib.sha256(raw).hexdigest()
        return value


def write_tables(kind, result, output):
    def table(filename, rows, columns):
        with atomic_output(output / filename) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)

    if kind == "spectral":
        rows = [
            {"example": k, **result[k]}
            for k in ("rank_loss", "repeated_cutoff", "complete_cluster", "eigenvalue_crossing")
        ]
        rows += [{"example": "controlled_spd", **r} for r in result["controlled_spd"]["rows"]]
        rows += [{"example": "diffusion", **r} for r in result["diffusion"]]
        table(
            "spectral.csv",
            rows,
            [
                "example",
                "n",
                "corner_width",
                "amplitude",
                "retained_rank",
                "largest_angle_degrees",
                "separation",
                "separation_status",
                "residual_angle_bound",
                "sufficient_angle_bound",
                "coupling_angle_bound",
                "bound_status",
            ],
        )
    elif kind == "verification":
        rows = [
            {"protocol": g["protocol"]["protocol"], **r}
            for g in result["cht_stopping"]
            for r in g["rows"]
        ]
        table(
            "stopping.csv",
            rows,
            [
                "protocol",
                "n",
                "warm_start",
                "method",
                "passed_sequences",
                "attempted_sequences",
                "accepted_median_seconds",
                "failed_target_statuses",
            ],
        )
    else:
        table(
            "resources.csv",
            result["resource_persistence"]["rows"],
            ["n", "target", "method", "attempts", "statuses", "accepted_median_seconds"],
        )
        table(
            "preparation.csv",
            result["preparation_inclusive"]["rows"],
            [
                "n",
                "warm_start",
                "method",
                "passed_sequences",
                "repetitions",
                "median_seconds",
                "with_common_seconds",
            ],
        )
        table(
            "rank_memory.csv",
            result["rank_memory"]["rows"],
            [
                "n",
                "rank",
                "deployed_rank",
                "status",
                "iterations",
                "cpu_original_residual",
                "gpu_original_residual",
                "total_seconds",
                "peak_torch_gib",
                "remaining_host_work_seconds",
            ],
        )
        table(
            "recycling.csv",
            result["last_window_recycling"]["rows"],
            ["family", "n", "method", "repetition", "seconds", "iterations", "statuses"],
        )


def run(kind, records_root, output):
    config = json.loads(
        files("deflation_example").joinpath(f"conf/tutorial_{kind}.json").read_text()
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_report(output / "protocol.json", config)
    records = Records(records_root)
    with threadpool_limits(limits=1):
        if kind == "spectral":
            result = spectral_examples()
            # Keep the measured diffusion and algebraic records as independently checked inputs.
            previous = records.load("extended/controls.json")
            original_spd = records.load("structure.json")["restriction"]["rows"]
            if len(original_spd) != len(result["controlled_spd"]["rows"]):
                raise ValueError("Missing controlled-SPD amplitude")
            for old, new in zip(original_spd, result["controlled_spd"]["rows"], strict=True):
                for key in (
                    "amplitude",
                    "actual_sine",
                    "separation",
                    "residual_angle_bound",
                    "sufficient_angle_bound",
                ):
                    if (old[key] is None) != (new[key] is None) or (
                        old[key] is not None
                        and not math.isclose(old[key], new[key], rel_tol=2e-9, abs_tol=1e-10)
                    ):
                        raise ValueError(
                            "Reconstructed SPD diagnostic differs from recorded formula"
                        )
            for group in previous["pde_separation"]:
                for row in group["rows"]:
                    current = next(
                        r
                        for r in result["diffusion"]
                        if (r["n"], r["corner_width"]) == (group["n"], row["corner_width"])
                    )
                    for key in (
                        "separation",
                        "actual_sine",
                        "minimum_restricted_singular_value",
                        "subspace_residual_frobenius",
                    ):
                        if not math.isclose(row[key], current[key], rel_tol=2e-9, abs_tol=1e-10):
                            raise ValueError(
                                "Reconstructed diffusion diagnostic differs from recorded formula"
                            )
        elif kind == "verification":
            result = {"accuracy": accuracy_examples(), "cht_stopping": stopping_examples(records)}
        else:
            result = cost_examples(records)
        report = {
            "tutorial_version": 1,
            "protocol": config,
            "environment": environment(),
            "input_sha256": records.manifest,
            "results": result,
        }
        write_report(output / "results.json", report)
        write_tables(kind, result, output)
    print(f"{kind}: wrote results.json; checked {len(records.manifest)} input records")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("spectral", "verification", "costs"))
    parser.add_argument("--records", type=Path, default=Path("examples/benchmarks"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.kind, args.records, args.output)


if __name__ == "__main__":
    main()
