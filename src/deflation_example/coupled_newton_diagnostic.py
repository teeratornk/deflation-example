"""Check the Jacobian and Newton direction at a saved forward-replay state.

This read-only diagnostic does not continue the trajectory or change a control.
It distinguishes a derivative or linear-solve defect from a large Newton step
and tests descent using both the solver's merit and fixed residual scaling.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

from .axisymmetric_flow import FlowResult
from .coupled_newton_replay import equation_merit, step_equations
from .coupled_optimize import load_problem
from .coupled_resolution import forward_model
from .coupled_saved import load_saved_solution, require_matching_baseline
from .coupled_spatial_resolution import transfer_source
from .coupled_step_spectrum import step_linearization
from .reporting import environment, write_report
from .validation import integer


def inspect_step(problem, source, previous_state, previous_flow, state, flow, slab):
    """Compare the analytic derivative with centered differences at fixed data."""
    model = forward_model(problem)
    residual, metrics = step_equations(
        problem, model, state, flow, source, previous_state, previous_flow, slab
    )
    H = step_linearization(problem, state[problem.free], flow.velocity, slab)[0]
    scaling = 1 / np.maximum(abs(H).max(axis=1).toarray().ravel(), np.finfo(float).tiny)
    factor = splu((sparse.diags(scaling) @ H).tocsc())
    update = factor.solve(-scaling * residual)
    denominator = max(np.linalg.norm(scaling * residual), np.finfo(float).tiny)
    nf = len(problem.flow_free)

    def candidate(direction, step):
        dx = np.zeros(problem.flow.size)
        dx[problem.flow_free] = direction[:nf]
        temperature = state.copy()
        temperature[problem.free] += step * direction[nf:]
        velocity = flow.velocity + step * np.column_stack(
            (dx[: problem.flow.nv], dx[problem.flow.nv : 2 * problem.flow.nv])
        )
        pressure = flow.pressure + step * dx[2 * problem.flow.nv :]
        return temperature, FlowResult(velocity, pressure, "diagnostic", [])

    def evaluate(direction, step):
        temperature, current_flow = candidate(direction, step)
        return step_equations(
            problem, model, temperature, current_flow, source, previous_state, previous_flow, slab
        )

    # Normalize each direction so the finite-difference displacement is explicit.
    rng = np.random.default_rng(827)
    directions = {"random": rng.normal(size=H.shape[0]), "newton": update.copy()}
    derivatives = []
    for name, direction in directions.items():
        direction /= max(np.max(np.abs(direction)), np.finfo(float).tiny)
        exact = H @ direction
        for epsilon in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
            plus = evaluate(direction, epsilon)[0]
            minus = evaluate(direction, -epsilon)[0]
            numerical = (plus - minus) / (2 * epsilon)
            derivatives.append(
                {
                    "direction": name,
                    "maximum_component_displacement": epsilon,
                    "relative_error": float(
                        np.linalg.norm(numerical - exact)
                        / max(np.linalg.norm(exact), np.finfo(float).tiny)
                    ),
                    "row_scaled_relative_error": float(
                        np.linalg.norm(scaling * (numerical - exact))
                        / max(np.linalg.norm(scaling * exact), np.finfo(float).tiny)
                    ),
                }
            )
    trials = []
    for step in (1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
        value, checks = evaluate(update, step)
        trials.append(
            {
                "step": step,
                "merit_ratio": float(equation_merit(checks) / max(equation_merit(metrics), 1e-300)),
                "fixed_row_scaled_residual_ratio": float(
                    np.linalg.norm(scaling * value) / denominator
                ),
                **checks,
            }
        )
    _, full_update = candidate(update, 1.0)
    return {
        "status": "complete",
        "slab_zero_based": slab,
        "state_degrees_of_freedom": problem.spatial_size,
        "jacobian_dimension": H.shape[0],
        "current_metrics": metrics,
        "linear_relative_residual": float(
            np.linalg.norm(scaling * (residual + H @ update)) / denominator
        ),
        "maximum_temperature_update_K": float(
            np.max(np.abs(update[nf:])) * problem.temperature_scale
        ),
        "maximum_velocity_update_m_s": float(np.max(np.abs(full_update.velocity - flow.velocity))),
        "derivative_checks": derivatives,
        "line_search_checks": trials,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fine-baseline", type=Path, required=True)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("reference", "jacobi", "recycling"), default="reference"
    )
    parser.add_argument("--target-position", type=int)
    parser.add_argument("--step", type=int, help="Zero-based step; defaults to the last saved step")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import json

    with threadpool_limits(integer(args.threads, "Threads", 1)):
        record, cfg, fields, digest = load_saved_solution(
            args.optimization, args.method, args.target_position
        )
        replay = json.loads((args.replay / "record.json").read_text())
        subdivision = integer(replay.get("subdivision", 1), "Time subdivision", 1)
        comparison_cfg = {**cfg, "slabs": cfg["slabs"] * subdivision}
        coarse, baseline = load_problem(
            {**comparison_cfg, "baseline_directory": str(args.baseline)}
        )
        fine, fine_baseline = load_problem(
            {**comparison_cfg, "baseline_directory": str(args.fine_baseline)}
        )
        require_matching_baseline(record, baseline)
        if (
            replay["optimization_field_sha256"] != digest
            or replay["fine_baseline_sha256"] != fine_baseline["baseline_sha256"]
            or replay["baseline_sha256"] != baseline["baseline_sha256"]
        ):
            raise ValueError("The saved replay must use the specified control and baselines")
        repeated = np.repeat(fields["control"].reshape(cfg["slabs"], -1), subdivision, axis=0)
        controls, _ = transfer_source(coarse, fine, repeated)
        with np.load(args.replay / "states.npz", allow_pickle=False) as data:
            n = len(data["state"]) - 1 if args.step is None else integer(args.step, "Step", 0)
            if n >= len(data["state"]) or n >= fine.slabs:
                raise ValueError("Select a saved time step")
            if not np.isclose(data["times_s"][n], fine.physical_steps[: n + 1].sum()):
                raise ValueError("The saved time does not match the problem")
            state = fine.full_temperature(data["state"][n])
            flow = FlowResult(data["velocity"][n].copy(), data["pressure"][n].copy(), "saved", [])
            previous_state = fine.full_temperature(data["state"][n - 1] if n else fine.initial)
            previous_flow = (
                FlowResult(
                    data["velocity"][n - 1].copy(), data["pressure"][n - 1].copy(), "saved", []
                )
                if n
                else fine.initial_flow
            )
        source = np.zeros(len(fine.mesh.nodes))
        source[fine.free] = controls[n]
        result = inspect_step(fine, source, previous_state, previous_flow, state, flow, n)
        args.output.mkdir(parents=True, exist_ok=False)
        write_report(
            args.output / "record.json",
            {
                "schema": "coupled-newton-step-diagnostic-v1",
                "environment": environment(),
                "optimization_field_sha256": digest,
                "fine_baseline_sha256": fine_baseline["baseline_sha256"],
                "replay_status": replay["status"],
                "scope": "Derivative and direction checks at unchanged saved fields; no continuation or reoptimization.",
                **result,
            },
        )


if __name__ == "__main__":
    main()
