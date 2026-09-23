"""Assemble the corrected coupled model's evidence into manuscript artifacts.

The campaign's original evidence was a population of complete sequences on an
operator whose stabilisation is now known to be inconsistent and first order, so
this study reports a different shape: declared comparisons, each read against
thresholds fixed before it ran, and each traceable to one outcome record.

Three things reach the manuscript from here.

  the correction   what the stabilisation defect was and what fixing it did to the
                   operator, including the conditioning it removed
  the regime map   where a reference coarse space pays and where it does not, which
                   on the corrected operator is mostly "does not"
  the solver       the velocity-frozen preconditioner that made a converged,
                   verified optimum on the corrected model reachable at all

Every number the prose cites is written here as a macro, so the manuscript holds
no hand-typed value and the gate can recompute each one from the records.
"""

import json
from pathlib import Path

from .reporting import atomic_output


def _ratio(value, digits=2):
    return "---" if value is None else f"{value:.{digits}f}"


# The protocols read out in their own vocabulary; the table says what it means.
VERDICTS = {
    "qualifies": "pays",
    "harder without reduction": "does not pay",
    "not harder": "does not pay",
    "misalignment not the cause; advantage not restored": "does not pay",
    "refuted: the correction removes the advantage": "does not pay",
}

LABELS = {
    "uncorrected operator": "uncorrected operator",
    "corrected, nominal regularization": r"corrected, $\alpha = 10^{-14}$",
    "corrected, alpha 1e-13": r"corrected, $\alpha = 10^{-13}$",
    "corrected, alpha 1e-12": r"corrected, $\alpha = 10^{-12}$",
    "corrected, alpha 1e-11": r"corrected, $\alpha = 10^{-11}$",
    "corrected, refined mesh": "corrected, refined mesh",
}


def load(directory):
    """Read the declared outcome records this study produced."""
    directory = Path(directory)
    wanted = {
        "headline": "coupled-smooth-preconditioner-v1-outcome.json",
        "ablation": "coupled-preconditioner-ablation-v1-outcome.json",
        "regime_pilot": "coupled-hard-regime-pilot-v1-outcome.json",
        "corrected_screen": "coupled-matched-reference-v1-outcome.json",
        "hard_regime_confirmation": "coupled-hard-regime-confirmation-v1-outcome.json",
    }
    records = {}
    for name, filename in wanted.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"The evidence needs {filename}")
        records[name] = json.loads(path.read_text())
    return records


def deflation_arm(directory):
    """The paper's own method on the same model, read from its record.

    The parallel study ran it: rank-200 reference deflation, same smoothed model,
    same warm start, converged and verified. It is the incumbent the preconditioner
    has to be compared against, because it is what this paper advocates everywhere
    else, and a comparison against an unpreconditioned solver alone would not
    answer the question a reader actually has.
    """
    record = Path(directory) / "coupled-smooth-optimization-v6-195465-1" / "record.json"
    if not record.is_file():
        return None
    data = json.loads(record.read_text())
    case = data["cases"][0]
    applications = sum(
        step["linear_iterations"]
        for row in case["history"]
        for attempt in row["attempts"]
        for step in attempt["qp_history"]
    )
    # The same verification quantities the headline arms carry, so the gate can hold
    # this arm to the same standard rather than trusting its "verified" flag alone.
    equations = case["equations"]
    return {
        "source": "coupled-smooth-optimization-v6-195465-1, the parallel study's run",
        "status": case["status"],
        "verified": case["verified"],
        "iterations": case["nonlinear_iterations"],
        "expensive_applications": applications,
        "hours": data["sequence_seconds"] / 3600,
        "optimality": case["kkt"]["stationarity"],
        "worst_thermal_residual": max(s["thermal_relative_residual"] for s in equations),
        "worst_mass_imbalance": max(s["mass_relative_imbalance"] for s in equations),
        "max_momentum_adjoint_residual": max(
            s["momentum_adjoint_relative_residual"] for s in case["adjoint"]["steps"]
        ),
    }


def iteration_counts(directory, pattern):
    """Median inner iteration counts from the public comparison records."""
    found = []
    for path in sorted(Path(directory).glob(pattern)):
        stack = [json.loads(path.read_text())]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                value = item.get("iterations")
                if isinstance(value, int):
                    found.append(value)
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
    return sorted(found)


def coarse_spectrum(directory):
    """What the deployed coarse space sees of the operator, measured two ways.

    The condition of the coarse operator is recorded by the solver in every run that
    deploys a space, so the comparison across regimes costs nothing to assemble. The
    controlled pair is the operator-cost screen, whose two rank-200 arms differ only
    in whether the stabilisation is the corrected one; the other entries differ in
    more than that and are labelled with what else they change.

    The scaled Rayleigh placement comes from the declared placement probe, which
    refuted the reading it was declared to test: both spaces lie far below the
    spectral bulk, so the loss of advantage is not the coarse space pointing along
    ordinary directions. Only what was measured is carried here.
    """
    directory = Path(directory)
    rows, values = [], {}

    def conditions(pattern):
        found = []
        for path in sorted(directory.glob(pattern)):
            record = json.loads(path.read_text())
            stack = [record]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    value = item.get("coarse_condition")
                    if isinstance(value, (int, float)) and value and value != 1.0:
                        found.append(float(value))
                    stack.extend(item.values())
                elif isinstance(item, list):
                    stack.extend(item)
        return sorted(found)

    screen = {
        "uncorrected": conditions(
            "coupled-consistent-scaling-v1-runs/shipped-reference-r200/stage-*/record.json"
        ),
        "corrected": conditions(
            "coupled-consistent-scaling-v1-runs/consistent-reference-r200/stage-*/record.json"
        ),
    }
    if not all(screen.values()):
        return None, {}, []
    median = lambda v: v[len(v) // 2]  # noqa: E731
    values["coupledUncorrectedCoarseCondition"] = _ratio(median(screen["uncorrected"]))
    values["coupledCorrectedCoarseCondition"] = _ratio(median(screen["corrected"]))
    matched = conditions(
        "coupled-frozen-screen-v1-runs/nominal-reference-r200-frozen/stage-*/record.json"
    )
    if matched:
        values["coupledMatchedCoarseCondition"] = _ratio(median(matched))
    hard = conditions(
        "coupled-hard-regime-pilot-v1-runs/alpha-1e-11-reference-r200-matched/stage-*/record.json"
    )
    if hard:
        values["coupledHardRegimeCoarseCondition"] = _ratio(max(hard), 0)
    transformer = conditions(
        "mesh-public-data-v1/final/transformer-transient8/reference-*/record.json"
    )
    if transformer:
        values["coupledTransformerCoarseCondition"] = _ratio(median(transformer), 1)
        for method in ("reference", "jacobi"):
            counts = iteration_counts(
                directory,
                f"mesh-public-data-v1/final/transformer-transient8/{method}-*/record.json",
            )
            if counts:
                values[f"coupledTransformer{method.capitalize()}Iterations"] = str(
                    int(median(counts))
                )

    placement = directory / "coupled-coarse-placement-v2.json"
    if placement.is_file():
        for row in json.loads(placement.read_text())["rows"]:
            name = "Uncorrected" if row["model"] == "uncorrected" else "Corrected"
            quotients = row["scaled_rayleigh_quotients"]
            low, high = quotients["0.0"], quotients["1.0"]
            values[f"coupled{name}PlacementRange"] = f"{low:.2g} to {high:.2g}"
            values[f"coupled{name}PlacementSpread"] = _ratio(high / low)
            values[f"coupled{name}ScaledMax"] = _ratio(
                row["largest_scaled_eigenvalue_power_estimate"]
            )
    return screen, values, rows


def regime_map(records):
    """Where the coarse space pays, in the order the manuscript reads them."""
    pilot, corrected = records["regime_pilot"], records["corrected_screen"]
    uncorrected = corrected["reference_numbers"]["shipped_operator"]
    nominal = corrected["ratios"]
    rows = [
        {
            "where": "uncorrected operator",
            "cg_ratio": uncorrected["C"],
            "wall_ratio": uncorrected["R"],
            "reading": "pays, on a discretisation that does not converge",
            "budget": "ten quadratics",
            "conditions": "hybrid device",
            "source": "the retained operator-cost scaling screen",
        },
        {
            "where": "corrected, nominal regularization",
            "cg_ratio": nominal["C"],
            "wall_ratio": nominal["R"],
            "reading": corrected["reading"],
            "budget": "ten quadratics",
            "conditions": "hybrid device",
            "source": "coupled-matched-reference-v1",
        },
    ]
    for regime in ("alpha-1e-13", "alpha-1e-12", "alpha-1e-11", "refined"):
        row = pilot["regimes"].get(regime) or {}
        ratios = row.get("ratios") or {}
        if regime == "refined":
            row, ratios = pilot["refined_v3"], pilot["refined_v3"]["ratios"]
        rows.append(
            {
                "where": regime.replace("alpha-", "corrected, alpha ").replace(
                    "refined", "corrected, refined mesh"
                ),
                "cg_ratio": ratios.get("C2"),
                "wall_ratio": ratios.get("R"),
                "reading": row.get("reading"),
                "budget": "two quadratics",
                "conditions": (
                    "host device, flow tolerance 1e-11, the third attempt at this regime"
                    if regime == "refined"
                    else "hybrid device"
                ),
                "source": "coupled-hard-regime-refined-v3"
                if regime == "refined"
                else "coupled-hard-regime-pilot-v1",
            }
        )
    return rows


def macros(records, regime, deflation=None):
    head = records["headline"]
    diagonal, frozen = head["arms"]["diagonal"], head["arms"]["frozen"]
    ablation = records["ablation"]
    coarse = ablation["readings"]["coarse_space"]["measured"]
    values = {
        # The solver result, on completed and verified solves.
        "coupledFrozenApplicationRatio": _ratio(head["E"]),
        "coupledFrozenWallRatio": _ratio(head["R"]),
        "coupledFrozenApplications": str(frozen["expensive_applications"]),
        "coupledDiagonalApplications": str(diagonal["expensive_applications"]),
        "coupledFrozenHours": _ratio(frozen["hours"]),
        "coupledDiagonalHours": _ratio(diagonal["hours"]),
        "coupledFrozenIterations": str(frozen["iterations"]),
        "coupledFrozenOptimality": f"{frozen['optimality']:.1e}",
        "coupledFrozenAgreement": f"{head['checks']['the_answer_did_not_move']['relative_objective_difference']:.0e}",
        "coupledFrozenReading": head["reading"],
        # What the coarse space does beside it, and how sensitive the setting is.
        "coupledCoarseBesideFrozenApplications": str(coarse["applications"]),
        "coupledCoarseBesideFrozenHours": _ratio(coarse["hours"]),
        "coupledSweepSpreadPercent": _ratio(
            100 * ablation["readings"]["sweeps"]["measured_spread"], 1
        ),
        # The regime map's two ends.
        "coupledUncorrectedCgRatio": _ratio(regime[0]["cg_ratio"]),
        "coupledUncorrectedWallRatio": _ratio(regime[0]["wall_ratio"]),
        "coupledCorrectedCgRatio": _ratio(regime[1]["cg_ratio"]),
        "coupledCorrectedWallRatio": _ratio(regime[1]["wall_ratio"]),
    }
    if deflation is not None:
        values.update(
            {
                "coupledDeflationApplications": str(deflation["expensive_applications"]),
                "coupledDeflationHours": _ratio(deflation["hours"]),
                "coupledFrozenAgainstDeflationApplications": _ratio(
                    deflation["expensive_applications"] / frozen["expensive_applications"]
                ),
                "coupledFrozenAgainstDeflationWall": _ratio(deflation["hours"] / frozen["hours"]),
            }
        )
    hard = next((r for r in regime if r["where"].endswith("1e-11")), None)
    if hard:
        values["coupledHardRegimeCgRatio"] = _ratio(hard["cg_ratio"])
        values["coupledHardRegimeWallRatio"] = _ratio(hard["wall_ratio"])
    worst = next((r for r in regime if r["where"].endswith("1e-12")), None)
    if worst:
        values["coupledWorstRegimeCgRatio"] = _ratio(worst["cg_ratio"])
    return values


def mechanism_rows(values, regime, controlled):
    """One row per operator: what the coarse space resolves, and what deflation did.

    The first two rows are the controlled pair and are the only comparison in this
    table where a single thing differs. The rest change more than the stabilisation
    and say so, so the table cannot be read as a sweep of one variable.
    """
    rows = []
    transformer = values.get("coupledTransformerCoarseCondition")
    deflated = values.get("coupledTransformerReferenceIterations")
    plain = values.get("coupledTransformerJacobiIterations")
    if transformer and deflated and plain:
        rows.append(
            [
                "prescribed-flow transformer",
                transformer,
                f"{deflated} against {plain} CG iterations",
            ]
        )
    uncorrected = values.get("coupledUncorrectedCoarseCondition")
    corrected = values.get("coupledCorrectedCoarseCondition")
    if uncorrected and corrected:
        rows.append(
            [
                "coupled, uncorrected stabilisation",
                uncorrected,
                f"{_ratio(regime[0]['cg_ratio'])} fewer CG iterations",
            ]
        )
        rows.append(
            [
                "coupled, corrected stabilisation",
                corrected,
                f"{_ratio(controlled['C'])}, no reduction",
            ]
        )
    matched = values.get("coupledMatchedCoarseCondition")
    if matched:
        rows.append(
            [
                "coupled, corrected, reference rebuilt to match",
                matched,
                f"{values.get('coupledCorrectedCgRatio', '')}, no reduction",
            ]
        )
    hard = values.get("coupledHardRegimeCoarseCondition")
    if hard:
        rows.append(
            [
                r"coupled, corrected, $\alpha = 10^{-11}$",
                hard,
                f"{values.get('coupledHardRegimeCgRatio', '')} fewer CG iterations",
            ]
        )
    return rows


def build(evidence_directory, output):
    """Write the macros, the tables and the summary the gate recomputes."""
    records = load(evidence_directory)
    regime = regime_map(records)
    deflation = deflation_arm(evidence_directory)
    screen, spectrum, _ = coarse_spectrum(evidence_directory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    values = macros(records, regime, deflation)
    values.update(spectrum)
    controlled = records["corrected_screen"]["reference_numbers"][
        "corrected_operator_shipped_reference"
    ]
    values["coupledScalingCorrectedCgRatio"] = _ratio(controlled["C"])
    mechanism = mechanism_rows(values, regime, controlled)
    with atomic_output(output / "macros.tex") as stream:
        stream.write(
            "\n".join(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items())
            + "\n"
        )
    end = r" \\"
    with atomic_output(output / "regime_rows.tex") as stream:
        stream.write(
            "\n".join(
                " & ".join(
                    [
                        LABELS.get(row["where"], row["where"]),
                        _ratio(row["cg_ratio"]),
                        _ratio(row["wall_ratio"]),
                        VERDICTS.get(row["reading"], row["reading"]),
                        row["budget"],
                    ]
                )
                + end
                for row in regime
            )
            + "\n"
        )
    if mechanism:
        with atomic_output(output / "mechanism_rows.tex") as stream:
            stream.write("\n".join(" & ".join(row) + end for row in mechanism) + "\n")
    head = records["headline"]
    with atomic_output(output / "preconditioner_rows.tex") as stream:
        stream.write(
            "\n".join(
                " & ".join(
                    [
                        label,
                        str(arm["iterations"]),
                        str(arm["expensive_applications"]),
                        f"{arm['hours']:.2f}",
                        f"{arm['optimality']:.1e}",
                        "yes" if arm["verified"] else "no",
                    ]
                )
                + end
                for label, arm in (
                    [("Jacobi diagonal", head["arms"]["diagonal"])]
                    + ([("rank-200 deflation", deflation)] if deflation else [])
                    + [("velocity-frozen", head["arms"]["frozen"])]
                )
            )
            + "\n"
        )
    summary = {
        "schema": "coupled-corrected-evidence-v1",
        "headline": {
            "population": ["diagonal", "frozen"],
            "arms": head["arms"],
            "application_ratio": head["E"],
            "wall_ratio": head["R"],
            "reading": head["reading"],
            "checks": head["checks"],
        },
        "regime": regime,
        "coarse_spectrum": {
            "controlled_pair": {
                "source": "coupled-consistent-scaling-v1-runs, the two rank-200 arms; identical "
                "in rank, regularisation, slab count, device and inner policy, differing only in "
                "consistent_stabilization",
                "uncorrected_coarse_conditions": screen["uncorrected"] if screen else [],
                "corrected_coarse_conditions": screen["corrected"] if screen else [],
                "cg_ratios": {"uncorrected": regime[0]["cg_ratio"], "corrected": controlled["C"]},
            },
            "placement": {
                "probe": "coupled-coarse-placement-v2",
                "declared_reading": "refuted: the corrected space did not move into the spectral "
                "bulk, so the loss of advantage is not the space pointing along ordinary "
                "directions",
                "what_stands": "both spaces lie far below the bulk; what collapses is the spread "
                "of the coarse space's spectral view",
            },
            "claim": "association only; no mechanism is asserted",
        },
        "deflation_reference": deflation,
        "ablation": records["ablation"]["readings"],
        "hard_regime_confirmation": {
            "reading": records["hard_regime_confirmation"]["reading"],
            "why": records["hard_regime_confirmation"]["why"],
        },
        "macros": values,
    }
    with atomic_output(output / "summary.json") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True, help="the study-runs directory")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    summary = build(arguments.evidence, arguments.output)
    print(json.dumps({k: summary[k] for k in ("schema",)}, indent=1))
    print(f"wrote {arguments.output}/macros.tex and its tables")


if __name__ == "__main__":
    main()
