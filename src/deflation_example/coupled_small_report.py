"""Retain every small-study outcome and compare only verified matched runs."""

import argparse
import json
from pathlib import Path

from .coupled_projected_report import audit
from .coupled_small_study import verification_identity
from .reporting import file_sha256, write_report


def summarize(records):
    rows = []
    for record in records:
        result = audit(record)
        if record is None:
            rows.append({**result, "speedup": None})
            continue
        cfg = record["configuration"]
        rows.append(
            {
                **result,
                "alpha": cfg["alpha"],
                "rank": cfg["rank"],
                "device": cfg["device"],
                "slabs": cfg["slabs"],
                "targets": cfg["queries"],
                "components_seconds": record.get("components_seconds", {}),
                "memory": record.get("memory"),
                "match": [
                    verification_identity(cfg),
                    cfg["queries"],
                    cfg["device"],
                    record.get("environment", {}).get("source_sha256"),
                    record.get("baseline_sha256"),
                    record.get("initial_state"),
                ],
                "speedup": None,
            }
        )
    for row in rows:
        if not row.get("verified") or row.get("rank", 0) == 0:
            continue
        baselines = [
            r
            for r in rows
            if r.get("rank") == 0 and r.get("verified") and r.get("match") == row["match"]
        ]
        if len(baselines) == 1 and row["cumulative_attempt_seconds"] > 0:
            row["speedup"] = (
                baselines[0]["cumulative_attempt_seconds"] / row["cumulative_attempt_seconds"]
            )
    for row in rows:
        row.pop("match", None)
    return {
        "schema": "coupled-small-study-summary-v1",
        "runs": rows,
        "scope": "Development single-run ratios include reference construction. Repeated complete sequences are required for a performance claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    records = [json.loads(p.read_text()) if p.exists() else None for p in args.records]
    report = summarize(records)
    for row, path in zip(report["runs"], args.records, strict=True):
        row.update(
            input_name=path.parent.name, record_sha256=file_sha256(path) if path.exists() else None
        )
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "summary.json", report)
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        alphas = sorted({r["alpha"] for r in report["runs"] if "alpha" in r})
        if not alphas:
            return
        fig, axes = plt.subplots(
            1, len(alphas), figsize=(5 * len(alphas), 4), squeeze=False, layout="constrained"
        )
        for alpha, axis in zip(alphas, axes[0], strict=True):
            for record, row in zip(records, report["runs"], strict=True):
                if record is None or row["alpha"] != alpha:
                    continue
                values = [
                    max(h["kkt"].values())
                    for c in record.get("cases", [])
                    for h in c.get("history", [])
                ]
                if record.get("cases") and record["cases"][-1].get("kkt"):
                    values.append(max(record["cases"][-1]["kkt"].values()))
                axis.semilogy(
                    range(len(values)),
                    values,
                    marker=".",
                    label=f"rank {row['rank']}: {row['status']}",
                )
            axis.axhline(1e-8, color="black", linestyle=":", linewidth=1)
            axis.set(
                title=f"Regularization {alpha:g}",
                xlabel="Recorded nonlinear step",
                ylabel="Maximum KKT component",
            )
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.legend(fontsize=8)
        fig.savefig(args.output / "convergence.pdf")
        fig.savefig(args.output / "convergence.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
