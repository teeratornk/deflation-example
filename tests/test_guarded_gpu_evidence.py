"""Keep the guarded follow-up distinct and check every released repetition."""

import hashlib
import json
from pathlib import Path

from deflation_example.mesh_report import load_study


def test_guarded_gpu_archive_checks_every_sequence_and_source():
    root = Path(__file__).parents[1] / "examples/guarded_refinement"
    manifest = json.loads((root / "manifest.json").read_text())
    for name, checksum in manifest["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == checksum
    total = 0
    for form in ("steady", "transient"):
        study, records = load_study(root / "records" / form)
        assert study["controls"]["targets"] == 4
        assert study["controls"]["repeats"] == 5
        assert study["controls"]["residual_policy"] == "refine"
        for method in study["methods"]:
            assert method["accepted"] == method["requested"] == 5
            assert method["max_original_residual"] <= 1e-10
            assert method["max_kkt"] <= 1e-8
            total += method["accepted"]
        for group in records.values():
            for record in group:
                assert record["environment"]["git_head"] == manifest["numerical_source"]
                assert len(record["cases"]) == 4
                for case in record["cases"]:
                    for solve in case["inner"]:
                        assert (
                            solve["refinement_policy"]
                            == "initial-guard-original-residual-error-equations-v2"
                        )
    assert total == 40
