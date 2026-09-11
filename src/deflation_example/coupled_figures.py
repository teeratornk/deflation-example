"""Plot saved coupled temperatures, signed sources and fluid velocities."""

import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_optimize import load_problem
from .coupled_pilot import transformer_inputs
from .coupled_saved import load_saved_solution, require_matching_baseline
from .reporting import write_report


def plot_fields(problem, fields, source_scale, upper_K, output, indices=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not np.isfinite(upper_K):
        raise ValueError("The displayed temperature bound must be finite")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import TwoSlopeNorm

    if not problem.mesh.axisymmetric or problem.mesh.dimension != 2:
        raise ValueError("This figure requires an axisymmetric thermal-flow model")
    if not np.isfinite(source_scale) or source_scale <= 0:
        raise ValueError("A positive physical source scale is required")
    indices = (
        np.unique(np.linspace(0, problem.slabs - 1, min(3, problem.slabs), dtype=int))
        if indices is None
        else np.asarray(indices)
    )
    if (
        indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or not len(indices)
        or len(np.unique(indices)) != len(indices)
        or indices.min() < 0
        or indices.max() >= problem.slabs
    ):
        raise ValueError("Choose distinct stored time levels within the trajectory")
    nodes, flow = problem.mesh.nodes, problem.flow
    shape = (problem.slabs, problem.spatial_size)
    state, target, source = (
        np.asarray(fields[name]).reshape(shape) for name in ("state", "desired", "control")
    )
    velocity = np.asarray(fields["velocity"])
    if velocity.shape != (problem.slabs, flow.nv, 2) or any(
        not np.isfinite(a).all() for a in (state, target, source, velocity)
    ):
        raise ValueError("Finite temperature, source and P2 velocity fields must match the model")
    tri = mtri.Triangulation(nodes[:, 1], nodes[:, 0], problem.mesh.cells[:, [0, 2, 1]])
    template = np.array([[0, 3, 4], [3, 1, 5], [4, 5, 2], [3, 5, 4]])
    fluid_tri = mtri.Triangulation(
        flow.points[:, 1], flow.points[:, 0], flow.p2[:, template].reshape(-1, 3)[:, [0, 2, 1]]
    )
    temperatures = problem.temperature_offset + problem.temperature_scale * np.concatenate(
        (state[indices], target[indices])
    )
    tmin = min(problem.temperature_offset, temperatures.min())
    tmax = max(float(upper_K), temperatures.max())
    qmax = max(float(np.max(np.abs(source[indices])) * source_scale / 1e6), np.finfo(float).eps)
    vmax = max(float(np.max(np.linalg.norm(velocity[indices], axis=-1))), np.finfo(float).eps)
    times = np.cumsum(problem.physical_steps) if len(problem.physical_steps) else np.array([0.0])
    fig, axes = plt.subplots(
        len(indices),
        4,
        figsize=(11, 2.5 * len(indices)),
        squeeze=False,
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    artists = []
    for row, index in enumerate(indices):
        fields_at_time = []
        for values in (target[index], state[index]):
            full = problem.full_temperature(values)
            fields_at_time.append(problem.temperature_offset + problem.temperature_scale * full)
        full_source = np.zeros(len(nodes))
        full_source[problem.free] = source[index] * source_scale / 1e6
        fields_at_time.append(full_source)
        for col, values in enumerate(fields_at_time):
            options = (
                {"cmap": "inferno", "vmin": tmin, "vmax": tmax}
                if col < 2
                else {"cmap": "RdBu_r", "norm": TwoSlopeNorm(vmin=-qmax, vcenter=0, vmax=qmax)}
            )
            artist = axes[row, col].tripcolor(
                tri, values, shading="gouraud", rasterized=True, **options
            )
            if row == 0:
                artists.append(artist)
        axes[row, 3].tripcolor(
            tri,
            facecolors=np.zeros(len(problem.mesh.cells)),
            cmap="Greys",
            vmin=-1,
            vmax=1,
            rasterized=True,
        )
        artist = axes[row, 3].tripcolor(
            fluid_tri,
            np.linalg.norm(velocity[index], axis=1),
            shading="gouraud",
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            rasterized=True,
        )
        if row == 0:
            artists.append(artist)
        axes[row, 0].set_ylabel(
            f"$t={times[index]:g}$ s\n$r$ (m)" if len(problem.physical_steps) else "$r$ (m)"
        )
    titles = ("Desired temperature", "Temperature", "Distributed source", "Fluid speed")
    units = ("K", "K", r"MW m$^{-3}$", r"m s$^{-1}$")
    for col, (title, unit) in enumerate(zip(titles, units, strict=True)):
        axes[0, col].set_title(title)
        axes[-1, col].set_xlabel("$z$ (m)")
        fig.colorbar(
            artists[col], ax=axes[:, col], orientation="horizontal", label=unit, shrink=0.9
        )
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return {
        "time_indices": indices.tolist(),
        "times_s": times[indices].tolist(),
        "temperature_bound_K": float(upper_K),
        "source_scale_W_m3": float(source_scale),
        "thermal_rendering": "P1 temperature and source fields; source is zero at prescribed-temperature nodes.",
        "velocity_rendering": "P2 nodal speed on four linear subtriangles per fluid element; solid regions are gray.",
        "coordinates": "Physical axial z and radial r coordinates in metres; plotting scales are unequal.",
        "scope": "Stored discrete time levels; this figure does not establish temporal resolution between them.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument(
        "--method", default="reference", choices=("jacobi", "reference", "recycling")
    )
    parser.add_argument("--target-position", type=int)
    parser.add_argument("--time-indices", nargs="+", type=int)
    parser.add_argument("--format", choices=("pdf", "png"), default="pdf")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(4):
        record, cfg, fields, digest = load_saved_solution(
            args.optimization, args.method, args.target_position
        )
        problem, baseline = load_problem({**cfg, "baseline_directory": str(args.baseline)})
        require_matching_baseline(record, baseline)
        field = args.optimization / (
            args.method + "-fields.npz"
            if args.target_position is None
            else f"target-{args.target_position:02d}.npz"
        )
        with np.load(field, allow_pickle=False) as data:
            fields["velocity"] = data["velocity"].copy()
        parameters = transformer_inputs(baseline["configuration"]["level"])[1]["physical"]
        source_scale = (
            parameters["oil_conductivity_W_m_K"]
            * parameters["temperature_scale_K"]
            / parameters["length_scale_m"] ** 2
        )
        args.output.mkdir(parents=True, exist_ok=False)
        description = plot_fields(
            problem,
            fields,
            source_scale,
            cfg["upper_K"],
            args.output / ("coupled_fields." + args.format),
            args.time_indices,
        )
        write_report(
            args.output / "figure.json",
            {
                **description,
                "optimization_field_sha256": digest,
                "baseline_sha256": baseline["baseline_sha256"],
                "method": args.method,
                "target": cfg["query"],
                "target_position": args.target_position,
            },
        )


if __name__ == "__main__":
    main()
