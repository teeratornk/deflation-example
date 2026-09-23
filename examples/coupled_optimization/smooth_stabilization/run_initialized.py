"""Run one fixed-policy smooth-model optimization from an assessed initial state."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("jacobi", "reference"), required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Full frozen source commit")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if len(args.source) != 40 or source != args.source:
        raise ValueError("The checkout must match the full declared source commit")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "tests", "examples"], cwd=root, text=True
    ).strip():
        raise ValueError("The numerical source and example must be clean")
    if args.output.exists():
        raise FileExistsError(args.output)
    rank = 0 if args.method == "jacobi" else 200
    command = [
        sys.executable,
        "-m",
        "deflation_example.coupled_sequence",
        "--config-name",
        "coupled_smooth",
        f"baseline_directory={args.baseline.resolve()}",
        f"initial_state_snapshot={args.snapshot.resolve()}",
        f"initial_state_assessment={args.assessment.resolve()}",
        f"output={args.output.resolve()}",
        f"method={args.method}",
        f"rank={rank}",
        f"recycle_window={max(1, rank)}",
    ]
    if args.dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
