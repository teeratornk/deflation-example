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
    assert config.reference_baseline_directory is None
    if method is not None:
        assert len(config.queries) == 1 and config.queries[0].target == 7


def test_separate_gate_root_requires_both_matching_derivatives(tmp_path):
    module = workflow()
    protocol = module.read(module.HERE / "protocol.json")
    gate_root = tmp_path / "gates"
    gate_root.mkdir()
    source = environment()
    write_report(
        gate_root / "selection.json",
        {
            "protocol_sha256": module.file_sha256(module.HERE / "protocol.json"),
            "environment": source,
            "families": {"momentum": {"selected": "anderson5"}},
        },
    )
    args = SimpleNamespace(
        phase="optimize",
        task=2,
        threads=8,
        data_root=tmp_path,
        output=tmp_path / "new-attempts",
        gate_root=gate_root,
    )
    for policy in ("newton", "anderson5"):
        folder = gate_root / "derivatives" / policy / "derivatives"
        folder.mkdir(parents=True)
        write_report(
            folder / "record.json",
            {
                "status": "verified",
                "environment": source,
            },
        )
    cmd, output = module.command(args, protocol)
    assert output == args.output / "optimize/newton/reference/rep-0"
    assert "reference_baseline_directory=null" in cmd
    failed = gate_root / "derivatives/anderson5/derivatives/record.json"
    write_report(failed, {"status": "failed", "environment": source})
    with pytest.raises(ValueError, match="Both matching numerical derivative gates"):
        module.command(args, protocol)
    write_report(failed, {"status": "verified", "environment": {"source_sha256": {}}})
    with pytest.raises(ValueError, match="Both matching numerical derivative gates"):
        module.command(args, protocol)


@pytest.mark.parametrize("steps", [[], [0.2, 0.3]])
def test_same_mesh_configured_reference_uses_identity_transfer(monkeypatch, steps):
    import numpy as np
    from deflation_example.coupled_reference import configured_reference
    from test_coupled_reference import problem
    from test_axisymmetric_flow import annular_rectangle

    def forbidden_reload(*args, **kwargs):
        raise AssertionError("The same-mesh reference must reuse the loaded problem")

    monkeypatch.setattr("deflation_example.coupled_optimize.load_problem", forbidden_reload)
    model = problem(annular_rectangle(3), steps)
    basis = configured_reference(
        model,
        {
            "rank": 2,
            "reference_baseline_directory": None,
            "reference_spatial_policy": "scaled_schur",
            "reference_temporal_metric": "jacobi",
        },
        {},
    )
    size = model.spatial_size * max(1, len(steps))
    values = basis.restrict(np.arange(size))
    assert values.shape == (size, 2)
    assert np.isfinite(values).all()
    assert np.linalg.matrix_rank(values) == 2
