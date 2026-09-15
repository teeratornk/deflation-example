"""Frozen selection and Hydra commands for the externally runnable study."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_module
import pytest
from deflation_example.reporting import environment, write_report


def workflow():
    path = Path(__file__).parents[1] / "examples/coupled_optimization/fixed_point/workflow.py"
    spec = importlib.util.spec_from_file_location("fixed_point_workflow", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_requires_all_records(tmp_path):
    module = workflow()
    protocol = module.read(module.HERE / "protocol.json")
    with pytest.raises(ValueError, match="Missing declared record"):
        module.selection(protocol, tmp_path)
    assert not (tmp_path / "selection.json").exists()


def test_selection_uses_coverage_before_time_and_preserves_failures(tmp_path):
    module = workflow()
    protocol = module.read(module.HERE / "protocol.json")
    source = environment()
    for family, case, policy in module.screen_tasks(protocol):
        output = tmp_path / "screen" / family / case["name"] / policy
        output.mkdir(parents=True)
        ok = policy == "anderson3" or case["name"] == "ordinary"
        write_report(
            output / "record.json",
            {
                "status": "complete",
                "environment": source,
                "optimization_field_sha256": protocol["optimization_field_sha256"],
                "row": {
                    "verified": ok,
                    "status": "converged" if ok else "iteration_cap",
                    "seconds": 100 if policy == "anderson3" else 1,
                },
            },
        )
    selected = module.selection(protocol, tmp_path)
    for family in protocol["families"]:
        assert selected["families"][family]["selected"] == "anderson3"
        assert len(selected["families"][family]["records"]) == 30
        assert selected["families"][family]["converged_counts"]["newton"] == 1
    with pytest.raises(FileExistsError):
        module.selection(protocol, tmp_path)


@pytest.mark.parametrize("method", [None, "jacobi", "reference", "recycling"])
def test_optimizer_commands_compose_without_hidden_defaults(tmp_path, method):
    module = workflow()
    protocol = module.read(module.HERE / "protocol.json")
    args = SimpleNamespace(
        output=tmp_path,
        data_root=tmp_path,
        threads=8,
        phase="derivatives" if method is None else "optimize",
    )
    cmd, _ = module.optimization_command(args, protocol, "anderson3", method=method)
    with initialize_config_module(config_module="deflation_example.conf", version_base=None):
        config = compose(
            config_name="coupled_optimize" if method is None else "coupled_sequence",
            overrides=cmd[3:],
        )
    assert config.flow_method == "anderson"
    assert config.flow_cap == 100
    assert config.slabs == 64 and config.target_startup_s == 60
    assert config.flow_tolerance == 1e-12 and config.inner_tolerance == 1e-10
    if method is not None:
        assert len(config.queries) == 1 and config.queries[0].target == 7
