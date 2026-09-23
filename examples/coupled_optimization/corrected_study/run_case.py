"""Run one declared corrected-formulation pilot without overwriting records."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, choices=range(4), required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Full frozen source commit")
    parser.add_argument("--device", choices=("cpu", "hybrid"), default="hybrid")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if len(args.source) != 40 or source != args.source:
        raise ValueError("The checkout must match the full declared source commit")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "src", "tests", "examples"],
        cwd=root,
        check=True,
    )
    design = json.loads(Path(__file__).with_name("protocol.json").read_text())
    baseline = json.loads((args.baseline / "record.json").read_text())
    for name in ("baseline_sha256", "thermal_nodes", "flow_unknowns"):
        if baseline[name] != design[name]:
            raise ValueError(f"Baseline {name} differs from the declared pilot")
    if args.output.exists():
        raise FileExistsError(args.output)
    case = design["pilots"][args.case]
    command = [
        sys.executable,
        "-m",
        "deflation_example.coupled_sequence",
        "--config-name",
        "coupled_corrected",
        f"baseline_directory={args.baseline.resolve()}",
        f"output={args.output.resolve()}",
        f"method={case['method']}",
        f"alpha={case['alpha']}",
        f"rank={case['rank']}",
        f"recycle_window={max(1, case['rank'])}",
        f"device={args.device}",
    ]
    if args.dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
