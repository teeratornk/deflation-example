"""Regenerate all replay summaries from the bundled measurement extracts."""

import argparse
import json
from pathlib import Path
import re

from deflation_example.coupled_retention_report import plot, summarize_by_deployment
from deflation_example.reporting import file_sha256, write_report


def summarize_extract(path):
    bundle = json.loads(path.read_text())
    if bundle["schema"] != "coupled-reference-screen-extract-v1":
        raise ValueError("Unsupported measurement extract")
    results, digests, names = [], set(), set()
    for group in bundle["groups"]:
        name = group["name"]
        if not re.fullmatch(r"[a-z][a-z0-9-]*", name) or name in names:
            raise ValueError("Unique portable comparison names are required")
        names.add(name)
        reports = []
        for record in group["records"]:
            digest = record["original_record_sha256"]
            if not re.fullmatch(r"[0-9a-f]{64}", digest) or digest in digests:
                raise ValueError("Original record identifiers must be unique SHA-256 digests")
            digests.add(digest)
            index = record["environment_index"]
            if type(index) is not int or not 0 <= index < len(bundle["environments"]):
                raise ValueError("Invalid environment index")
            reports.append({**record, "environment": bundle["environments"][index]})
        results.append(
            {
                "name": name,
                "original_record_sha256": [r["original_record_sha256"] for r in reports],
                "solve_count": sum(len(r["rows"]) for r in reports),
                "summary": summarize_by_deployment(bundle["manifest"], reports),
            }
        )
    return {
        "schema": "coupled-reference-screen-reproduction-v1",
        "measurement_extract_sha256": file_sha256(path),
        "scope": bundle["scope"],
        "groups": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("measurements.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_extract(args.input)
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "summary.json", result)
    for comparison in result["groups"]:
        for index, group in enumerate(comparison["summary"]["groups"]):
            folder = args.output / comparison["name"] / f"deployment-{index + 1:02d}"
            folder.mkdir(parents=True)
            plot(group["summary"], folder)
    print("Reproduced all measurement extracts; no PDE solve or pooled deployment comparison.")


if __name__ == "__main__":
    main()
