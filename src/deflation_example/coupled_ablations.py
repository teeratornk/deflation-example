"""Run and summarize the declared one-factor coupled optimization ablations."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf

from .coupled_optimize import derivative_report, equations_verified, load_problem
from .coupled_report import validate_record
from .coupled_saved import file_digest
from .coupled_sequence import run as run_sequence
from .coupled_targets import desired_temperature
from .reporting import environment, write_report
from .validation import integer

METHODS = ("jacobi", "reference", "recycling")
DEFAULT_PROTOCOL = Path("examples/coupled_optimization/ablations/protocol.json")


def protocol_digest(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def cases(protocol, device="cpu"):
    if protocol.get("schema") != "coupled-optimization-ablations-v1":
        raise ValueError("Unknown ablation protocol")
    integer(protocol["repetitions"], "Declared repetitions", 1)
    if device not in {"cpu", "cuda"}:
        raise ValueError("Choose cpu or cuda")
    selected = protocol["deployments"][device]
    if selected != "all" and (
        not isinstance(selected, list)
        or not selected
        or len(set(selected)) != len(selected)
        or set(selected) - {g["id"] for g in protocol["groups"]}
    ):
        raise ValueError("Declare distinct existing groups for each deployment")
    result, names, configurations = [], set(), set()
    for group in protocol["groups"]:
        if selected != "all" and group["id"] not in selected:
            continue
        if not re.fullmatch(r"[a-z][a-z0-9_]*", group["id"]):
            raise ValueError("Ablation identifiers must be lowercase words and digits")
        for method in group["methods"]:
            if method not in METHODS:
                raise ValueError("Unknown ablation solver")
            values = {**protocol["common"], **group["overrides"], "method": method}
            if method == "jacobi":
                values["rank"] = 0
            identity = json.dumps(values, sort_keys=True)
            name = group["id"] + "-" + method
            if name in names or identity in configurations:
                raise ValueError("The design must not count duplicate configurations")
            names.add(name)
            configurations.add(identity)
            result.append(
                {"id": name, "group": group["id"], "label": group["label"], "settings": values}
            )
    if not result:
        raise ValueError("Declare at least one ablation case")
    return result


def configuration(case, baseline, output, device="cpu", repetition=0):
    if device not in {"cpu", "cuda"}:
        raise ValueError("Choose cpu or cuda")
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).parent / "conf")):
        config = compose(config_name="coupled_sequence")
    return OmegaConf.merge(
        config,
        case["settings"],
        {
            "baseline_directory": str(baseline),
            "output": str(output),
            "device": device,
            "repetition": integer(repetition, "Repetition", 0),
        },
    )


def derivative_identity(config):
    # The target changes the linear tracking term, not the state/source Jacobian.
    # Bounds, reference construction and linear-solver choices likewise leave it
    # unchanged. Each temporal grid, physical model and momentum target is checked.
    return {
        key: config[key]
        for key in (
            "transient",
            "slabs",
            "horizon_s",
            "alpha",
            "flow_tolerance",
            "flow_cap",
            "flow_continuation",
        )
    }


def check_gate(gate, config, baseline_digest):
    if (
        gate.get("status") != "verified"
        or gate.get("baseline_sha256") != baseline_digest
        or gate.get("derivative_identity") != derivative_identity(config)
    ):
        raise ValueError("A matching verified derivative gate is required")
    report = gate["derivatives"]
    if (
        not np.isfinite(report["taylor_orders"]).all()
        or len(report["taylor_orders"]) < 3
        or min(report["taylor_orders"]) <= 1.9
        or not np.isfinite(report["relative_dot_product_error"])
        or report["relative_dot_product_error"] > 1e-9
        or not equations_verified(report["equations"])
    ):
        raise ValueError("Derivative gate disagrees with its numerical checks")


def run_gate(protocol, baseline, slabs, output):
    from threadpoolctl import threadpool_limits

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    case = cases(protocol)[0]
    config = OmegaConf.to_container(configuration(case, baseline, output), resolve=True)
    config["slabs"] = integer(slabs, "Time slabs", 1)
    with threadpool_limits(config["threads"]):
        problem, record = load_problem(config)
        query = config["queries"][0]
        desired = desired_temperature(
            problem, query["target"], config["target_count"], config["target_startup_s"]
        )
        metadata = {
            "schema": "coupled-ablation-derivative-gate-v1",
            "environment": environment(),
            "protocol_sha256": protocol_digest(protocol),
            "baseline_sha256": record["baseline_sha256"],
            "derivative_identity": derivative_identity(config),
        }
        write_report(output / "record.json", {**metadata, "status": "running"})
        problem.evaluation_callback = lambda row: write_report(output / "progress.json", row)
        report = {
            **metadata,
            "status": "verified",
            "derivatives": derivative_report(problem, desired),
        }
        try:
            check_gate(report, config, record["baseline_sha256"])
        except ValueError:
            report["status"] = "derivative_check_failed"
        write_report(output / "record.json", report)
        if report["status"] != "verified":
            raise ValueError("The derivative checks failed")


def run_case(protocol, index, baseline, gate_directory, root, device="cpu", repetition=0):
    design = cases(protocol, device)
    index = integer(index, "Case index", 0)
    repetition = integer(repetition, "Repetition", 0)
    if index >= len(design) or repetition >= protocol["repetitions"]:
        raise ValueError("Case or repetition lies outside the declared design")
    case = design[index]
    output = Path(root) / case["id"] / f"rep-{repetition}"
    config = configuration(case, baseline, output / "sequence", device, repetition)
    baseline_record = json.loads((Path(baseline) / "record.json").read_text())
    gate = json.loads((Path(gate_directory) / "record.json").read_text())
    check_gate(gate, config, baseline_record["baseline_sha256"])
    # Require the same numerical source for derivative verification and the run.
    env = environment()
    if gate["environment"]["source_sha256"] != env["source_sha256"]:
        raise ValueError("Derivative gate and optimizer must use the same source")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "coupled-ablation-case-v1",
        "protocol_sha256": protocol_digest(protocol),
        "case_index": index,
        "case": case,
        "device": device,
        "repetition": repetition,
        "derivative_gate_sha256": file_digest(Path(gate_directory) / "record.json"),
        "environment": env,
    }
    write_report(output / "case.json", {**manifest, "status": "running"})
    try:
        result = run_sequence(config)
    except Exception as failure:
        write_report(
            output / "case.json",
            {**manifest, "status": "run_error", "error_type": type(failure).__name__},
        )
        raise
    write_report(output / "case.json", {**manifest, "status": result["status"]})
    return result


def summarize(protocol, root, device="cpu"):
    rows, identities = [], set()
    for case in cases(protocol, device):
        for repetition in range(protocol["repetitions"]):
            directory = Path(root) / case["id"] / f"rep-{repetition}"
            row = {
                "case": case["id"],
                "group": case["group"],
                "label": case["label"],
                "method": case["settings"]["method"],
                "repetition": repetition,
                "status": "missing",
                "verified": False,
                "requested_rank": case["settings"]["rank"],
            }
            path, manifest_path = directory / "sequence" / "record.json", directory / "case.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if (
                    manifest["protocol_sha256"] != protocol_digest(protocol)
                    or manifest["case"] != case
                ):
                    raise ValueError("Ablation manifest differs from the declared design")
                row["status"] = manifest["status"]
            if path.exists():
                if not manifest_path.exists():
                    raise ValueError("Ablation records require their declared case manifest")
                report = json.loads(path.read_text())
                row["record_sha256"] = file_digest(path)
                cfg = report["configuration"]
                if any(cfg.get(k) != v for k, v in case["settings"].items()):
                    raise ValueError("Ablation output differs from its declared settings")
                if cfg["device"] != device:
                    raise ValueError("The output belongs to a different deployment")
                if cfg["repetition"] != repetition or manifest["repetition"] != repetition:
                    raise ValueError("Ablation repetition differs from its declared location")
                env = report["environment"]
                if env["source_sha256"] != manifest["environment"]["source_sha256"]:
                    raise ValueError("Ablation manifest and measured source differ")
                identities.add(
                    json.dumps(
                        {
                            "source": env["source_sha256"],
                            "cpu": env["cpu_model"],
                            "numpy": env["numpy"],
                            "scipy": env["scipy"],
                            "device": cfg["device"],
                            "device_details": report.get("device"),
                            "numerical_policy": report.get("numerical_policy"),
                            "baseline": report.get("baseline_sha256"),
                        },
                        sort_keys=True,
                    )
                )
                row["record_status"] = report["status"]
                if report["status"] != "running" or manifest["status"] != "run_error":
                    row["status"] = report["status"]
                else:
                    row["error_type"] = manifest.get("error_type")
                if "sequence_seconds" in report:
                    total = report["sequence_seconds"]
                    components = list(report["components_seconds"].values())
                    if (
                        not np.isfinite([total, *components]).all()
                        or total <= 0
                        or any(value < -1e-8 for value in components)
                        or not np.isclose(sum(components), total, rtol=1e-10, atol=1e-8)
                    ):
                        raise ValueError(
                            "Ablation timing components must match the measured interval"
                        )
                    if report["status"] == "sequence_error" and not report["cases"]:
                        if report["all_problems_verified"] or report["verified_problems"]:
                            raise ValueError("An empty failed sequence cannot be verified")
                        row["error_type"] = report.get("error_type")
                    else:
                        row["verified"] = validate_record(report)
                    row["sequence_seconds"] = report["sequence_seconds"]
                    row["preparation_inclusive_seconds"] = report["preparation_inclusive_seconds"]
                    row["target_statuses"] = [c["status"] for c in report["cases"]]
                    row["inner_iterations"] = sum(
                        c.get("inner_iterations", 0) for c in report["cases"]
                    )
                    row["nonlinear_iterations"] = sum(
                        c.get("nonlinear_iterations", 0) for c in report["cases"]
                    )
                    row["maximum_kkt"] = max(
                        (max(c["kkt"].values()) for c in report["cases"] if "kkt" in c),
                        default=None,
                    )
                    steps = [
                        s
                        for c in report["cases"]
                        for h in c.get("history", [])
                        for a in h["attempts"]
                        for s in a["qp_history"]
                    ]
                    ranks = [s["deployed_rank"] for s in steps if "deployed_rank" in s]
                    row["deployed_rank_min"] = min(ranks, default=None)
                    row["deployed_rank_max"] = max(ranks, default=None)
                    row["fallback_count"] = sum(s.get("fallback") is not None for s in steps)
                    row["maximum_original_inner_residual"] = max(
                        (s["linear_residual"] for s in steps if "linear_residual" in s),
                        default=None,
                    )
                    row["newly_inactive"] = sum(
                        s.get("timing", {}).get("newly_inactive", 0) for s in steps
                    )
                    row["newly_active"] = sum(
                        s.get("timing", {}).get("newly_active", 0) for s in steps
                    )
                    memory = report["memory"]
                    for key in ("peak_host_rss_bytes", "peak_gpu_process_bytes"):
                        row["sampled_" + key] = memory.get(key) if memory.get("complete") else None
            rows.append(row)
    # Some pre-assembly records do not yet have their baseline hash. Check the
    # complete configuration in every case; final reports require one source and deployment.
    complete_identities = {i for i in identities if json.loads(i)["baseline"] is not None}
    if len(complete_identities) > 1:
        raise ValueError("Keep source, baseline and deployment populations separate")
    return {
        "schema": "coupled-ablation-summary-v1",
        "protocol_sha256": protocol_digest(protocol),
        "repetitions": protocol["repetitions"],
        "device": device,
        "rows": rows,
        "scope": protocol["timing"],
        "outcome_policy": protocol["outcomes"],
    }


def plot_summary(report, output):
    """Show all measured attempts; failed attempts use crosses, not speedups."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from .coupled_convergence import COLORS, LABELS

    groups = list(dict.fromkeys(row["group"] for row in report["rows"]))
    labels = {row["group"]: row["label"] for row in report["rows"]}
    measured = [row for row in report["rows"] if "sequence_seconds" in row]
    if not measured:
        return False
    with plt.rc_context({"font.size": 9}):
        fig, axes = plt.subplots(
            1, 2, figsize=(10, max(4, len(groups) * 0.43)), sharey=True, layout="constrained"
        )
        for row in measured:
            offset = (-0.18, 0, 0.18)[METHODS.index(row["method"])]
            y = groups.index(row["group"]) + offset
            for ax, key, scale in zip(
                axes, ("sequence_seconds", "inner_iterations"), (3600, 1), strict=True
            ):
                if row.get(key) is not None:
                    ax.scatter(
                        row[key] / scale,
                        y,
                        color=COLORS[row["method"]],
                        marker="o" if row["verified"] else "x",
                        s=25,
                    )
        axes[0].set_yticks(range(len(groups)), [labels[g] for g in groups])
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Complete one-target time (h)")
        axes[1].set_xlabel("Inner CG iterations")
        for ax in axes:
            ax.grid(axis="x", alpha=0.2)
            ax.set_xlim(left=0)
        handles = [
            Line2D([], [], marker="o", ls="", color=COLORS[m], label=LABELS[m]) for m in METHODS
        ]
        handles.append(
            Line2D([], [], marker="x", ls="", color="black", label="Failed or capped attempt")
        )
        axes[0].legend(handles=handles, loc="best", fontsize=8)
        fig.suptitle("Single-run ablations; all measured attempts retained", fontsize=10)
        fig.savefig(Path(output) / "ablations.pdf")
        fig.savefig(Path(output) / "ablations.png", dpi=220)
        plt.close(fig)
    return True


def write_summary(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_report(output / "summary.json", report)
    keys = list(dict.fromkeys(key for row in report["rows"] for key in row))
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(report["rows"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "gate", "run", "report"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--slabs", type=int, default=64)
    parser.add_argument("--case", type=int)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    required = {
        "list": (),
        "gate": ("baseline", "output"),
        "run": ("baseline", "gate", "root", "case"),
        "report": ("root", "output"),
    }
    if any(getattr(args, key) is None for key in required[args.action]):
        parser.error("Missing arguments for " + args.action)
    if args.action == "list":
        for index, case in enumerate(cases(protocol, args.device)):
            print(index, case["id"], json.dumps(case["settings"], sort_keys=True))
    elif args.action == "gate":
        run_gate(protocol, args.baseline, args.slabs, args.output)
    elif args.action == "run":
        run_case(
            protocol, args.case, args.baseline, args.gate, args.root, args.device, args.repetition
        )
    else:
        report = summarize(protocol, args.root, args.device)
        write_summary(report, args.output)
        if args.plot:
            plot_summary(report, args.output)


if __name__ == "__main__":
    main()
