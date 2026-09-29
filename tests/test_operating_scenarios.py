"""Operating scenarios: slab load profiles, per-query flows and chained windows."""

import numpy as np
import pytest
from omegaconf import OmegaConf
from scipy import sparse
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits

from deflation_example import benchmark_mesh as study
from deflation_example.mesh_presentation import summarize
from deflation_example.mesh_showcases import build_showcase, reassemble


@pytest.fixture(autouse=True)
def bounded_blas_threads():
    with threadpool_limits(2):
        yield


def controls(**overrides):
    return study.controls(OmegaConf.create({"methods": ["jacobi"], **overrides}))


@pytest.mark.parametrize("transient", [False, True])
def test_nominal_rows_and_initial_states_reproduce_the_forcing_bitwise(transient):
    c = controls(transient=transient, slabs=3)
    _, problem = study.build_model(c)
    assert study.operating_problem(problem) is problem
    width = 3 if transient else 1
    same = study.operating_problem(problem, [1.0] * width, problem.initial)
    assert np.array_equal(same.forcing, problem.forcing)
    assert same.H is problem.H


def test_load_rows_scale_only_the_slab_sources():
    c = controls(transient=True, slabs=3, initial_temperature=0.05)
    _, problem = study.build_model(c)
    row = [0.0, 1.0, 2.5]
    scaled = study.operating_problem(problem, row)
    n = problem.spatial_size
    first = problem.nodal_capacity * problem.initial / problem.steps[0]
    for k, factor in enumerate(row):
        expected = factor * problem.spatial_forcing + (first if k == 0 else 0)
        np.testing.assert_allclose(scaled.forcing[k * n : (k + 1) * n], expected, rtol=1e-15)
    assert scaled.H is problem.H


@pytest.mark.parametrize(
    "overrides",
    [
        {"targets": 2, "load_profile": [1.0, 1.0, 1.0]},
        {"targets": 2, "load_profile": [1.0, -0.5]},
        {"targets": 2, "query_flows": [1.0]},
        {"targets": 2, "query_flows": [1.0, 0.0]},
        {"targets": 2, "chain_initial": True},
        {"targets": 1, "query_targets": [0, 0], "query_bounds": [0.2, 0.2]},
        {
            "targets": 1,
            "query_targets": [0, 0],
            "query_bounds": [0.2, 0.2],
            "load_profile": [1.0, 1.0],
        },
    ],
)
def test_inconsistent_operating_scenarios_are_refused(overrides):
    with pytest.raises(ValueError):
        controls(**overrides)


def test_a_repeated_target_and_bound_is_a_new_query_under_another_load():
    c = controls(
        targets=1,
        query_targets=[0, 0],
        query_bounds=[0.2, 0.2],
        load_profile=[1.0, 1.5],
        save_fields=True,
    )
    record, fields, _ = study.sequence(c, "jacobi")
    assert record["success"], record.get("failure")
    cases = record["cases"]
    assert [case["load_row"] for case in cases] == [[1.0], [1.5]]
    assert cases[0]["target_sha256"] == cases[1]["target_sha256"]
    assert cases[0]["operation_sha256"] != cases[1]["operation_sha256"]
    _, problem = study.build_model(c)
    for case, field in zip(cases, fields):
        # Independent forward solve with the scaled source.
        state = spsolve(
            problem.spatial_A, field["control"] + case["load_row"][0] * problem.spatial_forcing
        )
        np.testing.assert_allclose(state, field["state"], rtol=1e-9, atol=1e-12)


def test_chained_windows_start_from_the_previous_final_state():
    c = controls(
        transient=True,
        slabs=2,
        targets=3,
        chain_initial=True,
        load_profile=[0.5, 1.0, 1.5, 2.0, 1.0, 0.5],
        save_fields=True,
    )
    record, fields, _ = study.sequence(c, "jacobi")
    assert record["success"], record.get("failure")
    _, problem = study.build_model(c)
    n = problem.spatial_size
    initial = problem.initial
    for case, field in zip(record["cases"], fields):
        assert case["initial_sha256"] == study._hash(initial)
        # Independent backward-Euler substitution from the chained initial state.
        previous, states = initial, []
        for k, dt in enumerate(problem.steps):
            C = problem.nodal_capacity / dt
            rhs = (
                field["control"][k * n : (k + 1) * n]
                + case["load_row"][k] * problem.spatial_forcing
                + C * previous
            )
            previous = spsolve(problem.spatial_A + sparse.diags(C), rhs)
            states.append(previous)
        np.testing.assert_allclose(np.concatenate(states), field["state"], rtol=1e-9, atol=1e-12)
        initial = field["state"][-n:]


@pytest.mark.parametrize("geometry", ["engine_3d", "transformer_2d"])
def test_flow_reassembly_changes_only_the_transport(geometry):
    showcase = build_showcase(geometry)
    same = reassemble(showcase, 1.0)
    for name in ("stiffness", "mass", "capacity"):
        a, b = getattr(same, name), getattr(showcase.assembly, name)
        if hasattr(a, "toarray"):
            assert (a != b).nnz == 0
        else:
            assert np.array_equal(a, b)
    assert np.array_equal(same.load, showcase.assembly.load)
    faster = reassemble(showcase, 2.0)
    assert (faster.stiffness != showcase.assembly.stiffness).nnz > 0
    assert np.array_equal(faster.mass, showcase.assembly.mass)
    assert np.array_equal(faster.load, showcase.assembly.load)
    with pytest.raises(ValueError):
        reassemble(showcase, 0.0)


@pytest.mark.parametrize("method", ["jacobi", "reference"])
def test_flow_queries_reassemble_at_each_change_and_keep_the_reference(method):
    c = controls(targets=3, query_flows=[1.0, 1.6, 1.6], methods=[method], rank=6)
    record = study.sequence(c, method)[0]
    assert record["success"], record.get("failure")
    cases = record["cases"]
    assert [case["flow_scale"] for case in cases] == [1.0, 1.6, 1.6]
    assert cases[0]["reassembly_seconds"] == 0.0
    assert cases[1]["reassembly_seconds"] > 0.0
    assert cases[2]["reassembly_seconds"] == 0.0


def test_default_records_carry_no_operating_fields():
    record = study.sequence(controls(targets=2), "jacobi")[0]
    assert record["success"], record.get("failure")
    assert all("operation_sha256" not in case for case in record["cases"])


def test_summaries_accept_repeated_targets_under_distinct_loads(tmp_path):
    config = OmegaConf.create(
        {
            "methods": ["jacobi"],
            "targets": 1,
            "query_targets": [0, 0],
            "query_bounds": [0.2, 0.2],
            "load_profile": [1.0, 1.5],
            "save_fields": False,
            "output": str(tmp_path / "run"),
        }
    )
    assert study.run(config)
    summary = summarize(tmp_path / "run")
    assert summary["methods"][0]["accepted"] == 1


def test_the_skew_transport_form_is_declared_for_the_transformer_only():
    with pytest.raises(ValueError):
        controls(transport_form="skew")
    with pytest.raises(ValueError):
        controls(geometry="transformer_2d", transport_form="upwind")
    advective = controls(geometry="transformer_2d")
    skew = controls(geometry="transformer_2d", transport_form="skew")
    _, a = study.build_model(advective)
    _, b = study.build_model(skew)
    assert (a.A != b.A).nnz > 0
    assert np.array_equal(a.spatial_forcing, b.spatial_forcing)
