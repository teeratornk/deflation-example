"""Cost accounting, resource ownership, and optional real-device checks."""

from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve
from deflation_example.amgx import AmgxSession, amgx_cg, destroy_all
from deflation_example.benchmark_gpu import problem_instances, representative
from deflation_example.spectral import analytical_reference, lowest_modes
from deflation_example.timing import PhaseTimer, PHASES


def test_timing_partitions_elapsed_interval():
    ticks = iter([2., 3., 5., 8., 10.])
    timer = PhaseTimer(clock=lambda: next(ticks))
    timer.mark("upload")
    timer.synchronize(lambda: None)
    result = timer.finish()
    assert result["total_seconds"] == 8
    assert sum(result["components_seconds"].values()) == 8
    assert result["components_seconds"]["upload"] == 1
    assert result["components_seconds"]["synchronization"] == 3
    assert set(result["components_seconds"]) == set(PHASES)
    with pytest.raises(ValueError):
        timer.finish()


@pytest.mark.parametrize("next_time", [-1., float("nan")])
def test_invalid_clock_rejected(next_time):
    ticks = iter([0., next_time])
    timer = PhaseTimer(clock=lambda: next(ticks))
    with pytest.raises(ValueError):
        timer.mark("iteration")


def fake_amgx(fail_setup=False):
    created, destroyed, setups = Counter(), Counter(), []

    class Object:
        def __init__(self, kind):
            self.kind = kind
            self.status, self.iterations_number = "success", 1

        def create(self, *args):
            created[self.kind] += 1
            return self

        create_simple = create
        create_from_dict = create

        def destroy(self):
            destroyed[self.kind] += 1

        def upload_CSR(self, value):
            self.value = value.copy()

        def upload(self, value):
            self.value = value.copy()

        def setup(self, matrix):
            setups.append(matrix.value.shape)
            if fail_setup:
                raise RuntimeError("simulated setup error")
            self.matrix = matrix.value

        def solve(self, rhs, x):
            x.value = spsolve(self.matrix, rhs.value)

        def download(self, target):
            target[:] = self.value

    api = SimpleNamespace(**{kind: (lambda k=kind: Object(k))
                             for kind in ("Config", "Resources", "Matrix", "Vector", "Solver")})
    return api, created, destroyed, setups


@pytest.mark.parametrize("persistent", [False, True])
def test_resources_not_hierarchies_are_reused(persistent):
    api, created, destroyed, setups = fake_amgx()
    session = AmgxSession(api).open() if persistent else None
    for n in (3, 5):
        A, b = sparse.eye(n), np.ones(n)
        result, metrics = amgx_cg(A, b, api=api, synchronize=lambda: None, session=session)
        assert result.status == "converged"
        assert metrics["resources_reused"] == persistent
        assert metrics["hierarchy_reused"] is False
        assert sum(metrics["components_seconds"].values()) == pytest.approx(metrics["total_seconds"])
        np.testing.assert_array_equal(b, np.ones(n))
    if persistent:
        session.close()
    assert created == destroyed
    assert created["Resources"] == (1 if persistent else 2)
    assert created["Solver"] == created["Matrix"] == 2
    assert setups == [(3, 3), (5, 5)]


def test_cleanup_on_setup_failure():
    api, created, destroyed, _ = fake_amgx(fail_setup=True)
    with pytest.raises(RuntimeError, match="setup error"):
        amgx_cg(sparse.eye(3), np.ones(3), api=api, synchronize=lambda: None)
    assert created == destroyed


def test_cleanup_attempts_all_objects():
    calls = []

    class Failing:
        def destroy(self):
            calls.append(1)
            raise RuntimeError("simulated cleanup error")

    with pytest.raises(RuntimeError, match="cleanup failed"):
        destroy_all([Failing(), Failing()])
    assert calls == [1, 1]


def test_median_is_one_record_and_failure_not_hidden():
    records = [{"total_seconds": t, "residual": 1e-11, "status": "converged", "marker": i}
               for i, t in enumerate((3., 1., 2.))]
    assert representative(records)["marker"] == 2
    records[0]["status"] = "maxiter"
    assert representative(records)["status"] == "repetition_failed"


def test_reference_eigenpairs_and_mode_order():
    from deflation_example.problems import laplacian
    n, rank = 5, 12
    basis, modes = analytical_reference(n, 2, rank)
    L = laplacian(n)
    values = 4*(n+1)**2*np.sin(np.pi*np.arange(1, n+1)/(2*(n+1)))**2
    eigenvalues = np.array([sum(values[i-1] for i in mode) for mode in modes])
    np.testing.assert_allclose(basis.T@basis, np.eye(rank), atol=2e-14)
    np.testing.assert_allclose(L@basis, basis*eigenvalues, atol=2e-13)
    assert np.all(np.diff(eigenvalues) >= -1e-12)
    assert modes == lowest_modes(n, 2, rank)


def test_problem_generation_uses_distinct_targets_and_verified_pdas():
    cases = list(problem_instances(8))
    assert len(cases) == 3
    assert len({c[3]["rhs_sha256"] for c in cases}) == 3
    assert len({c[3]["bound"] for c in cases}) == 1
    assert all(max(c[3]["outer_kkt"].values()) <= 1e-8 for c in cases)


def timing_fixture():
    from deflation_example.benchmark_report import METHODS
    runs = []
    for repeat in range(3):
        parts = dict.fromkeys(PHASES, 0.)
        parts["iteration"] = repeat + 1.
        runs.append({"repetition": repeat, "status": "converged", "residual": 1e-11,
                     "total_seconds": repeat + 1., "components_seconds": parts,
                     "hierarchy_reused": False})
    return {"protocol": "itemized-gpu-resource-control-v1", "success": True,
            "controls": {"repeats": 3}, "grids": [{"n": 8, "reference_seconds": .5,
                "warmups": {m: {"total_seconds": .1} for m in METHODS},
                "persistent_session_startup": {"total_seconds": .2},
                "persistent_session_cleanup": {"total_seconds": .3}}],
            "cases": [{"n": 8, "index": i, "restriction_seconds": .01,
                       "basis_restriction_seconds": .02,
                       "repetitions": {m: deepcopy(runs) for m in METHODS},
                       "solvers": {m: deepcopy(runs[1]) for m in METHODS}} for i in range(3)]}


def test_cumulative_resource_cost_charged_once():
    from deflation_example.benchmark_report import checked_series
    methods = checked_series(timing_fixture())[8]["methods"]
    assert methods["deflated_gpu_qr"]["cumulative_seconds"][-1] == pytest.approx(.6+3*2.03)
    assert methods["amgx_fresh_resources"]["cumulative_seconds"][-1] == pytest.approx(.1+3*2.01)
    assert methods["amgx_persistent_resources"]["cumulative_seconds"][-1] == pytest.approx(.6+3*2.01)


@pytest.mark.parametrize("fault", ["failed", "double_count", "hierarchy", "median", "missing"])
def test_report_rejects_inconsistent_measurements(fault):
    from deflation_example.benchmark_report import checked_series
    report = timing_fixture()
    run = report["cases"][0]["repetitions"]["amgx_fresh_resources"][0]
    if fault == "failed":
        run["status"] = "maxiter"
    elif fault == "double_count":
        run["components_seconds"]["cleanup"] = 1.
    elif fault == "hierarchy":
        run["hierarchy_reused"] = True
    elif fault == "median":
        report["cases"][0]["solvers"]["amgx_fresh_resources"]["total_seconds"] = 1.
    else:
        report["cases"].pop()
    with pytest.raises(ValueError):
        checked_series(report)


def test_complete_sequence_rebuilds_all_inner_systems(monkeypatch):
    from deflation_example.benchmark_sequence import complete_sequence
    api, created, destroyed, setups = fake_amgx()
    torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    sequence = complete_sequence(4, 2, .0001, "amgx_persistent_resources", torch, api, 1e-10, 1e-8)
    assert sequence["success"]
    assert created == destroyed
    assert created["Resources"] == 1
    assert created["Matrix"] == len(setups) == sequence["outer_iterations"]
    assert sequence["total_seconds"] >= sum(c["seconds"] for c in sequence["cases"])
    assert all(max(c["kkt"].values()) <= 1e-8 for c in sequence["cases"])


@pytest.mark.gpu
def test_gpu_qr_parity_and_timing():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from deflation_example.gpu import gpu_deflated_cg
    rng = np.random.default_rng(18)
    K = rng.normal(size=(30, 30))
    A = sparse.csr_matrix(K.T@K + np.eye(30))
    b = rng.normal(size=30)
    Z = np.linalg.eigh(A.toarray())[1][:, :5]
    for basis in (Z, np.column_stack([Z, Z]), np.zeros((30, 5))):
        for backend in ("gpu_qr", "cpu_svd"):
            result, timing = gpu_deflated_cg(A, b, basis, A.diagonal(), basis_backend=backend)
            assert result.status == "converged" and result.residual <= 1e-10
            np.testing.assert_allclose(result.x, spsolve(A, b), atol=1e-9)
            assert sum(timing["components_seconds"].values()) == pytest.approx(timing["total_seconds"])
            assert timing["orthogonalized_rank"] == (0 if not basis.any() else 5)
