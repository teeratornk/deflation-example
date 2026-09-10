"""Generate verification and conditional-analysis tables from checked records."""

import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .benchmark_cht_conditioning import controlled_cht
from .benchmark_cht_report import load_study, summarize
from .reporting import atomic_output, environment, write_report
from .tutorials import Records


FAMILIES = {
    "steady": "Steady",
    "uniform_trajectory": "Uniform trajectory",
    "nonuniform_trajectory": "Unequal-step trajectory",
}


def scientific(value):
    return f"{value:.2e}"


def verify_conditioning(record):
    with threadpool_limits(limits=1):
        rebuilt = controlled_cht(record["n"], [row["removed_nodes"] for row in record["rows"]])
    keys = (
        "minimum_restricted_singular_value",
        "subspace_residual_frobenius",
        "separation",
        "sufficient_angle_bound",
        "actual_sine",
        "original_preconditioned_condition",
        "deflated_condition_upper_bound",
        "measured_effective_condition",
    )
    for stored, fresh in zip(record["rows"], rebuilt["rows"], strict=True):
        for key in keys:
            if stored[key] is None or fresh[key] is None:
                if stored[key] is not fresh[key]:
                    raise ValueError("Conditional-bound availability differs from reconstruction")
            elif not np.isclose(stored[key], fresh[key], rtol=1e-7, atol=1e-10):
                raise ValueError(f"Reconstructed conditional quantity differs: {key}")
        for key in (
            "retained_rank",
            "sufficient_bound_status",
            "certifies_conditioning_improvement",
        ):
            if stored[key] != fresh[key]:
                raise ValueError(f"Conditional status differs from reconstruction: {key}")
    return rebuilt


def validation_rows(record):
    if not record["success"]:
        raise ValueError("Validation table requires the accepted independent comparison")
    rows = []
    for group in record["groups"]:
        for case in group["cases"]:
            methods = list(case["methods"].values())
            if not case["independent_success"] or any(m["status"] != "converged" for m in methods):
                raise ValueError("Validation contains an unsuccessful optimization")
            independent = max(case["independent_kkt"].values())
            kkt = max(v for m in methods for v in m["kkt"].values())
            if max(independent, kkt) > record["kkt_tolerance"]:
                raise ValueError("Validation fails its stated KKT tolerance")
            if any(
                inner["status"] != "converged"
                or inner["original_residual"] > record["final_inner_rtol"]
                for method in methods
                for inner in method["inner"]
            ):
                raise ValueError("Validation contains a failed original residual")
            rows.append(
                {
                    "family": group["family"],
                    "query": case["query"],
                    "independent_kkt": independent,
                    "pdas_kkt": kkt,
                    "relative_state_difference": max(
                        m["state_relative_difference"] for m in methods
                    ),
                    "relative_objective_difference": max(
                        m["relative_objective_difference"] for m in methods
                    ),
                }
            )
    return rows


def pilot_tables(roots, output):
    records = []
    with (
        atomic_output(output / "pilot_protocol_rows.tex") as settings,
        atomic_output(output / "pilot_outcome_rows.tex") as outcomes,
    ):
        for index, root in enumerate(roots, 1):
            protocol, sequences, hashes = load_study(root)
            c, identifier = protocol["controls"], f"P{index}"
            rows = summarize(protocol, sequences)
            grid = rf"$ {c['n']}^3$"
            if c["problem"] == "transient":
                grid += rf"$\times {c['slabs']}$"
            settings.write(
                f"{identifier} & {grid} & {c['targets']} & {c['bound']:.3e} & "
                f"{c.get('initial_active', 'empty')} & {c.get('residual_refresh', 1000)} & "
                f"{c['rtol']:.0e} & {c.get('cg_factor', 1):g} & "
                f"{'yes' if c.get('cache_operator_product', False) else 'no'}" + r" \\" + "\n"
            )
            entries = []
            for method in ("jacobi", "reference", "recycling", "amgx"):
                selected = [r for r in rows if r["method"] == method]
                if not selected:
                    entries.append("--")
                    continue
                if len(selected) != 1:
                    raise ValueError("A compact pilot row requires one declared start policy")
                row = selected[0]
                value = "--" if row["median_seconds"] is None else f"{row['median_seconds']:.3f}"
                entries.append(f"{row['accepted_sequences']}/{row['declared_sequences']}; {value}")
            outcomes.write(identifier + " & " + " & ".join(entries) + r" \\" + "\n")
            records.append(
                {
                    "identifier": identifier,
                    "study": Path(root).name,
                    "protocol": protocol,
                    "input_sha256": hashes,
                    "rows": rows,
                }
            )
    return records


def run(validation_root, conditioning_root, output, pilot_roots=()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    validation, conditioning = Records(validation_root), Records(conditioning_root)
    validation_record = validation.load("results.json")
    conditioning_record = conditioning.load("results.json")
    rows = validation_rows(validation_record)
    rebuilt = verify_conditioning(conditioning_record)
    with atomic_output(output / "validation_rows.tex") as stream:
        for row in rows:
            values = [
                row["independent_kkt"],
                row["pdas_kkt"],
                row["relative_state_difference"],
                row["relative_objective_difference"],
            ]
            stream.write(
                f"{FAMILIES[row['family']]} & {row['query']} & "
                + " & ".join(scientific(value) for value in values)
                + r" \\"
                + "\n"
            )
    with atomic_output(output / "conditioning_rows.tex") as stream:
        for row in conditioning_record["rows"]:
            stream.write(
                f"{row['removed_nodes']} & {row['minimum_restricted_singular_value']:.5f} & "
                f"{row['subspace_residual_frobenius']:.5f} & {row['separation']:.5f} & "
                f"{row['actual_sine']:.4f} & {row['sufficient_angle_bound']:.4f} & "
                f"{row['original_preconditioned_condition']:.2f} & "
                f"{row['deflated_condition_upper_bound']:.2f} & "
                f"{row['measured_effective_condition']:.2f}" + r" \\" + "\n"
            )
    pilots = pilot_tables(pilot_roots, output)
    report = {
        "protocol": "cht-supporting-tables-v1",
        "generator_environment": environment(),
        "input_sha256": {
            "validation": validation.manifest,
            "conditioning": conditioning.manifest,
            "pilots": {pilot["study"]: pilot["input_sha256"] for pilot in pilots},
        },
        "validation": rows,
        "reconstructed_conditioning": rebuilt,
        "pilots": pilots,
    }
    write_report(output / "support.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", required=True)
    parser.add_argument("--conditioning-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot-roots", nargs="*", default=[])
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
