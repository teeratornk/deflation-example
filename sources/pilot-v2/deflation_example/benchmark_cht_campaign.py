"""Frozen complete-optimization comparisons for the reference-policy study."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from .benchmark_cht import run as run_study, specification
from .reporting import environment, write_report


BASE = {
    "phase": "final",
    "device": "cuda",
    "targets": 16,
    "rank": 100,
    "window": 100,
    "repeats": 5,
    "initial_active": "all",
    "residual_refresh": 10000,
    "cg_factor": 0.1,
    "amgx_factor": 0.1,
    "rtol": 1e-10,
    "outer_tolerance": 1e-8,
    "cache_operator_product": True,
    "warm_starts": ["outer_inner"],
    "threads": 4,
}


def studies():
    """Each entry is an independent population of five complete repetitions."""
    result = {}
    for n in (24, 32, 48, 64):
        result[f"steady{n}"] = {"problem": "steady", "n": n}
    for n in (12, 16, 24, 32):
        result[f"transient{n}"] = {"problem": "transient", "n": n, "slabs": 8}
    for slabs in (4, 16, 32):
        result[f"time{slabs}"] = {"problem": "transient", "n": 16, "slabs": slabs}
    for problem, n in (("steady", 32), ("transient", 24)):
        for rank in (20, 200):
            result[f"{problem}-rank{rank}"] = {
                "problem": problem,
                "n": n,
                "rank": rank,
                "window": rank,
                "methods": ["reference", "recycling"],
            }
        result[f"{problem}-cold"] = {
            "problem": problem,
            "n": n,
            "warm_starts": ["cold"],
        }
    return result


def campaign_specification():
    return {
        "protocol": "complete-reference-policy-campaign-v1",
        "base": BASE,
        "studies": studies(),
        "decision": "settings fixed after correctness, construction, stopping, rank and spatial pilots; mode-dependent temporal construction selected for its exact homogeneous reference eigensystem; tensor and mode-dependent pilot timings were similar",
        "rank_zero": "the Jacobi method in each primary population is the efficient rank-zero control; rank-20 and rank-200 comparisons reuse the same target functions, bound and numerical criteria",
        "memory_comparison": {
            "gpu_budgets_bytes": [8 * 2**30, 16 * 2**30, 64 * 2**30],
            "host_budget_bytes": 16 * 2**30,
            "rule": "report all requested ranks first; separately screen the rank-20, rank-100 and rank-200 records against common sampled process-memory budgets and compare the measured cost of feasible ranks; these finite-rank cost envelopes describe the measured choices, without a held-out optimal-rank claim",
            "sampling": "10 ms target interval with observed maximum gap; sampled peaks are lower bounds, not hard allocation limits",
        },
        "failure_rule": "retain every declared attempt; exclude numerical failures and capped runs from accepted-solution speedups; keep memory screening separate from numerical acceptance",
        "calibration": "one fixed physical bound per family from the supplied calibration record; import its measured cost once in preparation-inclusive totals; all spatial and temporal refinements preserve that bound",
    }


def run(output, selected, steady_calibration, transient_calibration):
    available = studies()
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(available):
        raise ValueError("Select distinct declared studies")
    observed = environment()
    if observed["source_tree_clean"] is False:
        raise ValueError("Freeze a clean source revision before the final campaign")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = campaign_specification()
    report.update(environment=observed, selected=selected, outcomes=[], complete=False)
    configs = {}
    for name in selected:
        settings = available[name]
        calibration = (
            steady_calibration if settings["problem"] == "steady" else transient_calibration
        )
        config = OmegaConf.create(
            BASE | settings | {"calibration_report": calibration, "output": str(output / name)}
        )
        configs[name] = config
    report["declared_studies"] = {name: specification(c) for name, c in configs.items()}
    write_report(output / "campaign.json", report)
    for name, config in configs.items():
        result = run_study(config)
        report["outcomes"].append(
            {"study": name, "complete": result["complete"], "success": result["success"]}
        )
        write_report(output / "campaign.json", report)
    report["complete"] = len(report["outcomes"]) == len(selected)
    write_report(output / "campaign.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selected", nargs="+", choices=list(studies()), required=True)
    parser.add_argument("--steady-calibration", required=True)
    parser.add_argument("--transient-calibration", required=True)
    report = run(**vars(parser.parse_args()))
    if not all(row["success"] for row in report["outcomes"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
