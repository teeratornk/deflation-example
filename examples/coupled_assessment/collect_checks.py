"""Collect every supplied verification record without filtering by outcome."""

import argparse
import hashlib
import json
from pathlib import Path

from deflation_example.reporting import write_report


def collect(forward_records, thermal_diagnostics):
    rows, seen = [], set()
    for path in forward_records:
        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen:
            raise ValueError("Duplicate forward record")
        seen.add(digest)
        record = json.loads(raw)
        steps = record["steps"]
        if record["status"] == "running":
            raise ValueError("A completed forward attempt is required")
        rows.append(
            {
                "record_sha256": digest,
                "source": record["environment"]["git_head"],
                "source_tree_clean": record["environment"]["source_tree_clean"],
                "configuration": record["configuration"],
                "population": record["optimization_population"],
                "selection": record["selection"],
                "status": record["status"],
                "mesh": record["mesh"],
                "bound_K": record["bound_K"],
                "recorded_time_steps": len(steps),
                "trajectory_completed": record["status"] == "converged",
                "observed_peak_temperature_K": max(
                    (s["peak_temperature_K"] for s in steps), default=None
                ),
                "observed_bound_violation_K": max(
                    (s["bound_violation_K"] for s in steps), default=None
                ),
                "observed_fluid_temperature_range_K": [
                    min((s["oil_temperature_min_K"] for s in steps), default=None),
                    max((s["oil_temperature_max_K"] for s in steps), default=None),
                ],
                "metrics": record.get("metrics"),
                "source_integral_refinement_difference_W": record.get(
                    "source_integral_refinement_difference_W"
                ),
                "saved_state_max_difference_K": record.get("saved_state_max_difference_K"),
                "resolution_status": record.get("resolution_status"),
                "steps": [
                    {
                        k: s[k]
                        for k in (
                            "time_s",
                            "status",
                            "peak_temperature_K",
                            "minimum_temperature_K",
                            "bound_violation_K",
                            "maximum_location_rz_m",
                            "published_temperature_domain_satisfied",
                            "history",
                        )
                    }
                    for s in steps
                ],
            }
        )
    diagnostics = []
    for path in thermal_diagnostics:
        raw = Path(path).read_bytes()
        diagnostics.append({"record_sha256": hashlib.sha256(raw).hexdigest(), **json.loads(raw)})
    return {
        "schema": "fixed-control-verification-v1",
        "material_reference": "10.1049/hve2.12345",
        "scope": "Prescribed-flow discretization checks performed before coupled-control assessment. Observed temperatures from incomplete attempts describe their computed prefixes. These results do not establish mesh or time convergence, or the accuracy of a physical temperature prediction.",
        "forward_records": rows,
        "thermal_refinement_diagnostics": diagnostics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-records", nargs="+", type=Path, required=True)
    parser.add_argument("--thermal-diagnostics", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write_report(args.output, collect(args.forward_records, args.thermal_diagnostics))
    print(args.output.name)


if __name__ == "__main__":
    main()
