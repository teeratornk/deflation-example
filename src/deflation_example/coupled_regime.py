"""When the coarse space pays: the two declared regime ablations, read by their rules.

The headline campaign measures one point. These two ablations move the cost of a
single operator application in both directions from it and ask whether the sign of
the comparison follows. The velocity-frozen Gauss--Newton model makes one
application roughly sixty times cheaper; doubling the time slabs makes it roughly
twice as expensive while leaving a fixed-rank coarse correction alone.

Every verdict here is the one written in the corresponding design file before any of
its runs were read. Nothing in this module chooses a rule; it applies them and says
which clause failed when one does.
"""

import argparse
import json
from pathlib import Path

from .coupled_convergence import maximum_kkt
from .coupled_presentation import _fmt, linear_split
from .reporting import atomic_output, environment, write_report

SCHEMA = "coupled-regime-analysis-v1"
TANGENT_SCHEMA = "coupled-tangent-regime-design-v1"
SCALING_SCHEMA = "coupled-operator-cost-scaling-v1"
KKT_TOLERANCE = 1e-8
# The tangent design's thresholds, restated here only so a failure names its clause.
FROZEN_PRODUCTION_INFLATION = 4.85
REFUTED_SPEEDUP = 1.2
COUPLED_MINIMUM_SPEEDUP = 1.2
FROZEN_MAXIMUM_SHARE = 0.25
STRADDLE = (0.95, 1.05)
SCALING_MINIMUM_CHANGE = 0.05
OBJECTIVE_TOLERANCE = 1e-6


def load_arm(directory):
    """One measured arm: its split, its verification and the state it stopped in."""
    path = Path(directory) / "record.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    cases = record.get("cases") or []
    if not cases:
        return {"status": record.get("status", "unknown"), "measured": False}
    split = linear_split(record)
    case = cases[0]
    kkt = maximum_kkt(case["kkt"]) if case.get("kkt") else None
    return {
        "directory": str(directory),
        "status": case.get("status"),
        "sequence_status": record.get("status"),
        "measured": record.get("status") == "complete",
        "verified": bool(case.get("verified")),
        "maximum_kkt": kkt,
        "objective": case.get("objective"),
        "tangent": record["configuration"].get("tangent", "coupled"),
        "method": record["configuration"]["method"],
        "rank": record["configuration"]["rank"],
        "slabs": record["configuration"].get("slabs"),
        **split,
    }


def pair(zero, deflated):
    """The declared quantities of one rank-zero and one deflated arm."""
    if not zero or not deflated or "sequence_seconds" not in zero:
        return None
    if "sequence_seconds" not in deflated:
        return None
    outer = zero["outer_iterations"], deflated["outer_iterations"]
    if not all(outer):
        return None
    per_iteration = (
        zero["sequence_seconds"] / outer[0],
        deflated["sequence_seconds"] / outer[1],
    )
    cg_per_iteration = (zero["cg_iterations"] / outer[0], deflated["cg_iterations"] / outer[1])
    per_cg = (
        zero["linear_seconds"] / max(zero["cg_iterations"], 1),
        deflated["linear_seconds"] / max(deflated["cg_iterations"], 1),
    )
    return {
        "S": zero["sequence_seconds"] / deflated["sequence_seconds"],
        "S_per_iteration": per_iteration[0] / per_iteration[1],
        "S_inner": zero["linear_seconds"] / deflated["linear_seconds"]
        if deflated["linear_seconds"]
        else None,
        "C": cg_per_iteration[0] / cg_per_iteration[1],
        "B": per_cg[1] / per_cg[0] if per_cg[0] else None,
        "share": {"zero": zero["linear_fraction"], "deflated": deflated["linear_fraction"]},
        "outer_iterations": {"zero": outer[0], "deflated": outer[1]},
        "equal_outer_iterations": outer[0] == outer[1],
    }


def optima_agree(first, second):
    if first is None or second is None:
        return None
    if first.get("objective") is None or second.get("objective") is None:
        return None
    scale = max(abs(first["objective"]), 1e-300)
    return abs(first["objective"] - second["objective"]) / scale <= OBJECTIVE_TOLERANCE


def row_directory(root, row, repetition):
    """Where a row's record is, allowing for the matched-cap fallback.

    When a pre-check kill switch fires, the design's fallback runs every row with
    a matched outer-iteration cap, and the runner names that directory with a
    ``-cap`` suffix so it sits beside the uncapped one instead of replacing it.
    The uncapped record is preferred when both exist.
    """
    parent = root / f"{row['tangent']}-{row['method']}-r{row['rank']}"
    plain = parent / f"rep-{repetition}"
    if (plain / "record.json").exists():
        return plain, None
    capped = sorted(parent.glob(f"rep-{repetition}-cap*"))
    for candidate in capped:
        if (candidate / "record.json").exists():
            return candidate, int(candidate.name.rsplit("cap", 1)[1])
    return plain, None


def tangent_analysis(design, root):
    """Apply the tangent design's rules to whatever of its rows have been measured."""
    root = Path(root)
    arms, rows = {}, []
    caps = set()
    for row in design["rows"]:
        repetition = row.get("repetition", 0)
        directory, cap = row_directory(root, row, repetition)
        arm = load_arm(directory)
        if arm is not None and cap is not None:
            arm["matched_outer_iteration_cap"] = cap
            caps.add(cap)
        key = (row["tangent"], row["method"], repetition)
        arms[key] = arm
        rows.append({**row, "repetition": repetition, "arm": arm})
    pairs, inflation = {}, {}
    for tangent in design["tangents"]:
        zero, deflated = arms.get((tangent, "jacobi", 0)), arms.get((tangent, "reference", 0))
        pairs[tangent] = pair(zero, deflated)
    for method in ("jacobi", "reference"):
        coupled, frozen = arms.get(("coupled", method, 0)), arms.get(("frozen", method, 0))
        if (
            coupled
            and frozen
            and coupled.get("outer_iterations")
            and frozen.get("outer_iterations")
        ):
            inflation[method] = frozen["outer_iterations"] / coupled["outer_iterations"]
    repetitions = {}
    for tangent in design["tangents"]:
        speedups = []
        for repetition in (0, 1):
            measured = pair(
                arms.get((tangent, "jacobi", repetition)),
                arms.get((tangent, "reference", repetition)),
            )
            if measured:
                speedups.append(measured["S"])
        repetitions[tangent] = speedups
    verdict = tangent_verdict(design, arms, pairs, inflation, repetitions)
    return {
        "design_schema": design["schema"],
        "rows": rows,
        "pairs": pairs,
        "outer_iteration_inflation": inflation,
        "repeated_speedups": repetitions,
        "matched_outer_iteration_caps": sorted(caps),
        "verdict": verdict,
    }


def tangent_verdict(design, arms, pairs, inflation, repetitions):
    """The design's own clauses, in its own order, naming the first that fails."""
    frozen, coupled = pairs.get("frozen"), pairs.get("coupled")
    if frozen is None or coupled is None:
        return {"status": "pending", "reason": "Not every arm of both models has been measured"}
    capped = sorted(
        {
            arm["matched_outer_iteration_cap"]
            for arm in arms.values()
            if arm and "matched_outer_iteration_cap" in arm
        }
    )
    if capped:
        # A pre-check kill switch fired, so the design directs the comparison to be
        # made at equal outer-iteration count. That comparison does not depend on
        # the inflation, and a capped arm is unverified by construction.
        return {
            "status": "inconclusive",
            "reason": (
                "A pre-check kill switch fired, so these rows are the declared "
                f"comparison at a matched cap of {capped[0]} outer iterations, which "
                "does not depend on the outer-iteration inflation"
            ),
            "matched_outer_iteration_caps": capped,
            "equal_outer_iteration_comparison": {
                name: None
                if measured is None
                else {
                    "per_iteration_wall": measured["S_per_iteration"],
                    "C": measured["C"],
                    "B": measured["B"],
                    "equal_outer_iterations": measured["equal_outer_iterations"],
                }
                for name, measured in pairs.items()
            },
        }
    unverified = [
        f"{key[0]} {key[1]}"
        for key, arm in arms.items()
        if arm
        and arm.get("measured")
        and (not arm.get("verified") or (arm.get("maximum_kkt") or 1) > KKT_TOLERANCE)
    ]
    if unverified:
        return {
            "status": "inconclusive",
            "reason": f"An arm did not verify at the nonlinear tolerance: {', '.join(sorted(unverified))}",
        }
    agreement = {
        method: optima_agree(arms.get(("coupled", method, 0)), arms.get(("frozen", method, 0)))
        for method in ("jacobi", "reference")
    }
    if any(value is False for value in agreement.values()):
        return {
            "status": "inconclusive",
            "reason": "The two models reached different optima beyond the declared tolerance",
            "agreement": agreement,
        }
    straddles = [s for s in repetitions.get("frozen", []) if STRADDLE[0] <= s <= STRADDLE[1]]
    if STRADDLE[0] <= frozen["S"] <= STRADDLE[1] and len(straddles) >= 2:
        return {
            "status": "inconclusive",
            "reason": "The frozen speedup sits within a twentieth of one with repetitions straddling it",
        }
    if frozen["S"] >= REFUTED_SPEEDUP:
        return {
            "status": "refuted",
            "reason": "Deflation still pays on the cheap operator, so the stated mechanism needs revising",
        }
    clauses = {
        "S_inner(frozen) < 1": frozen["S_inner"] is not None and frozen["S_inner"] < 1,
        "S(frozen) < 1": frozen["S"] < 1,
        "S(coupled) > 1.2": coupled["S"] > COUPLED_MINIMUM_SPEEDUP,
        "share(frozen) < 0.25": frozen["share"]["deflated"] is not None
        and frozen["share"]["deflated"] < FROZEN_MAXIMUM_SHARE,
        "C(frozen) < B(frozen)": frozen["B"] is not None and frozen["C"] < frozen["B"],
    }
    failed = [name for name, held in clauses.items() if not held]
    if failed:
        return {
            "status": "inconclusive",
            "reason": f"Clauses not met: {', '.join(failed)}",
            "clauses": clauses,
        }
    production = {
        method: value < FROZEN_PRODUCTION_INFLATION for method, value in inflation.items()
    }
    return {
        "status": "supported",
        "clauses": clauses,
        "frozen_is_the_better_production_choice": all(production.values()) if production else None,
        "outer_iteration_inflation": inflation,
    }


def scaling_analysis(design, root):
    """Apply the scaling design's rules to its matched fixed-budget cases."""
    root = Path(root)
    arms, rows = {}, []
    for case in design["cases"]:
        directory = root / f"slabs{case['slabs']}-{case['method']}-r{case['rank']}" / "stage-1"
        arm = load_arm(directory)
        arms[(case["slabs"], case["method"])] = arm
        rows.append({**case, "arm": arm})
    budget = design["common"]["nonlinear_cap"]
    ratios = {}
    for slabs in sorted({case["slabs"] for case in design["cases"]}):
        measured = pair(arms.get((slabs, "jacobi")), arms.get((slabs, "reference")))
        if measured is not None:
            measured["R"] = measured["S_per_iteration"]
            ratios[slabs] = measured
    return {
        "design_schema": design["schema"],
        "budget": budget,
        "rows": rows,
        "ratios": ratios,
        "verdict": scaling_verdict(arms, ratios, budget),
    }


def scaling_verdict(arms, ratios, budget):
    if len(ratios) < 2:
        return {"status": "pending", "reason": "Both slab counts need a measured matched pair"}
    short = [
        f"{slabs} slabs {method}"
        for (slabs, method), arm in arms.items()
        if arm and arm.get("measured") and arm.get("outer_iterations", 0) < budget
    ]
    if short:
        return {
            "status": "inconclusive",
            "reason": f"An arm stopped before the matched budget: {', '.join(sorted(short))}",
        }
    disagreeing = [
        slabs
        for slabs in ratios
        if optima_agree(arms.get((slabs, "jacobi")), arms.get((slabs, "reference"))) is False
    ]
    if disagreeing:
        return {
            "status": "inconclusive",
            "reason": f"The two arms diverged at {disagreeing} slabs, so the budgets are not matched",
        }
    coarse, fine = sorted(ratios)
    change = (ratios[fine]["R"] - ratios[coarse]["R"]) / ratios[coarse]["R"]
    if ratios[fine]["R"] < ratios[coarse]["R"]:
        return {
            "status": "refuted",
            "reason": "The coarse correction grows at least as fast as the operator",
            "relative_change": change,
        }
    if abs(change) < SCALING_MINIMUM_CHANGE:
        return {
            "status": "inconclusive",
            "reason": "The two cost ratios differ by less than a twentieth",
            "relative_change": change,
        }
    falling = ratios[fine]["B"] is not None and ratios[fine]["B"] < ratios[coarse]["B"]
    if not falling:
        return {
            "status": "inconclusive",
            "reason": "The cost ratio grew but the per-iteration coarse cost did not fall",
            "relative_change": change,
        }
    return {"status": "supported", "relative_change": change}


def build(tangent_design, tangent_root, scaling_design, scaling_root, output):
    report = {
        "schema": SCHEMA,
        "generator_environment": environment(),
        "tangent": None,
        "scaling": None,
    }
    if tangent_design is not None:
        design = json.loads(Path(tangent_design).read_text())
        if design["schema"] != TANGENT_SCHEMA:
            raise ValueError("The tangent design has an unexpected schema")
        report["tangent"] = tangent_analysis(design, tangent_root)
    if scaling_design is not None:
        design = json.loads(Path(scaling_design).read_text())
        if design["schema"] != SCALING_SCHEMA:
            raise ValueError("The scaling design has an unexpected schema")
        report["scaling"] = scaling_analysis(design, scaling_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_report(output / "regime.json", report)
    write_tables(report, output)
    return report


def _row(cells):
    return " & ".join(cells) + r" \\"


def write_tables(report, output):
    lines = []
    tangent = report.get("tangent")
    if tangent:
        for name, measured in sorted((tangent["pairs"] or {}).items()):
            if measured is None:
                lines.append(_row([name.capitalize(), "not measured", "---", "---", "---", "---"]))
                continue
            lines.append(
                _row(
                    [
                        name.capitalize(),
                        _fmt(measured["S"], ".2f"),
                        _fmt(measured["S_inner"], ".2f"),
                        _fmt(measured["C"], ".2f"),
                        _fmt(measured["B"], ".2f"),
                        _fmt(measured["share"]["deflated"], ".2f"),
                    ]
                )
            )
    with atomic_output(output / "regime_rows.tex") as handle:
        handle.write("\n".join(lines) + ("\n" if lines else ""))
    lines = []
    scaling = report.get("scaling")
    if scaling:
        for slabs, measured in sorted((scaling["ratios"] or {}).items()):
            lines.append(
                _row(
                    [
                        str(slabs),
                        _fmt(measured["R"], ".2f"),
                        _fmt(measured["C"], ".2f"),
                        _fmt(measured["B"], ".2f"),
                        _fmt(measured["share"]["zero"], ".2f"),
                        _fmt(measured["share"]["deflated"], ".2f"),
                    ]
                )
            )
    with atomic_output(output / "scaling_rows.tex") as handle:
        handle.write("\n".join(lines) + ("\n" if lines else ""))


def macros(report):
    """Prose numbers of the regime analysis; missing measurements stay missing."""
    values = {}
    tangent = report.get("tangent") or {}
    for name, measured in (tangent.get("pairs") or {}).items():
        label = name.capitalize()
        if measured is None:
            continue
        values[f"coupled{label}Speedup"] = _fmt(measured["S"], ".2f")
        values[f"coupled{label}CgReduction"] = _fmt(measured["C"], ".2f")
        values[f"coupled{label}PerCgCost"] = _fmt(measured["B"], ".2f")
    verdict = (tangent.get("verdict") or {}).get("status")
    if verdict:
        values["coupledTangentVerdict"] = verdict
    inflation = tangent.get("outer_iteration_inflation") or {}
    if "jacobi" in inflation:
        values["coupledFrozenOuterInflation"] = _fmt(inflation["jacobi"], ".2f")
    scaling = report.get("scaling") or {}
    for slabs, measured in (scaling.get("ratios") or {}).items():
        values[f"coupledScaling{slabs}Ratio"] = _fmt(measured["R"], ".2f")
        values[f"coupledScaling{slabs}PerCgCost"] = _fmt(measured["B"], ".2f")
    scaling_verdict = (scaling.get("verdict") or {}).get("status")
    if scaling_verdict:
        values["coupledScalingVerdict"] = scaling_verdict
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tangent-design", type=Path)
    parser.add_argument("--tangent-runs", type=Path)
    parser.add_argument("--scaling-design", type=Path)
    parser.add_argument("--scaling-runs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print", action="store_true", help="Print a readable summary")
    args = parser.parse_args()
    report = build(
        args.tangent_design,
        args.tangent_runs,
        args.scaling_design,
        args.scaling_runs,
        args.output,
    )
    if args.print:
        print(json.dumps({"macros": macros(report)}, indent=1))
        for name in ("tangent", "scaling"):
            section = report.get(name)
            if section:
                print(name, json.dumps(section["verdict"], indent=1))


if __name__ == "__main__":
    main()
