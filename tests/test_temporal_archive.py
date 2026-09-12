"""Archive validation retains unsuccessful cases and rejects mixed evidence."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "temporal_archive", Path(__file__).parents[1] / "tools/archive_temporal_study.py"
)
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def optimization():
    environment = {
        "git_head": "source",
        "source_tree_clean": True,
        "source_sha256": {"code": "hash"},
    }
    design = {"protocol_sha256": "protocol", "environment": environment}
    case = {"query": 7, "slabs": 4}
    record = {**case, **deepcopy(design), "status": "verification_failed", "verified": False}
    return record, design, case


def test_failed_optimization_remains_in_the_archive():
    record, design, case = optimization()
    archive.verify_optimization(record, design, case, "source")


@pytest.mark.parametrize("status", ["running", "missing"])
def test_incomplete_optimization_is_not_a_final_record(status):
    record, design, case = optimization()
    record["status"] = status
    with pytest.raises(ValueError, match="terminated attempt"):
        archive.verify_optimization(record, design, case, "source")


def test_archive_rejects_changed_source():
    record, design, case = optimization()
    record["environment"]["source_sha256"]["code"] = "changed"
    with pytest.raises(ValueError, match="target selection"):
        archive.verify_optimization(record, design, case, "source")


def test_unavailable_git_status_still_requires_matching_file_hashes():
    record, design, case = optimization()
    record["environment"]["source_tree_clean"] = None
    archive.verify_optimization(record, design, case, "source")
    record["environment"]["source_sha256"]["code"] = "changed"
    with pytest.raises(ValueError, match="target selection"):
        archive.verify_optimization(record, design, case, "source")


def test_capped_replay_remains_in_the_archive():
    _, design, _ = optimization()
    record = {
        "case": 0,
        "replay_slabs": [256, 512],
        "selected_cases": [0, 4, 5, 9],
        "protocol_sha256": design["protocol_sha256"],
        "optimization_source": design["environment"],
        "status": "resolution_cap",
    }
    archive.verify_replay(record, design, 0, [256, 512])
    record["selected_cases"] = [0, 4]
    with pytest.raises(ValueError, match="selection changed"):
        archive.verify_replay(record, design, 0, [256, 512])
