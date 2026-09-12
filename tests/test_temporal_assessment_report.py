"""The presentation keeps paired inputs and every preselected case."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "temporal_report", Path(__file__).parents[1] / "tools/report_temporal_assessment.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def pair():
    a = {
        "design": {
            "protocol": {"transport_form": "advective", "bound": 0.8},
            "input_sha256": {"mesh": "frozen"},
            "queries": [7, 15],
            "cases": [{"query": 7, "slabs": 4}, {"query": 15, "slabs": 4}],
            "desired_exceedance_scores": [1.0, 2.0],
            "physical": {"horizon_s": 600.0},
        },
        "rows": [
            {"query": 7, "slabs": 4, "verified": False},
            {"query": 15, "slabs": 4, "verified": True},
        ],
    }
    b = deepcopy(a)
    b["design"]["protocol"]["transport_form"] = "skew"
    return a, b


def test_paired_report_retains_unsuccessful_rows():
    a, b = pair()
    report.validate_pair(a, b)
    b["rows"].pop(0)
    with pytest.raises(ValueError, match="Every declared optimization"):
        report.validate_pair(a, b)


@pytest.mark.parametrize(
    "key", ["queries", "input_sha256", "physical", "cases", "desired_exceedance_scores"]
)
def test_report_rejects_changed_physical_inputs_or_selection(key):
    a, b = pair()
    b["design"][key] = None
    with pytest.raises(ValueError, match="Paired designs differ"):
        report.validate_pair(a, b)


def test_report_rejects_other_protocol_changes():
    a, b = pair()
    b["design"]["protocol"]["bound"] = 0.9
    with pytest.raises(ValueError, match="only changed"):
        report.validate_pair(a, b)
