"""Plot the body-fitted geometries and accepted changing temperature constraints."""

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .mesh_presentation import tetrahedral_slice
from .benchmark_mesh import build_model
from .benchmark_cht import unpack_mask
from .mesh_showcases import desired_temperature
from .meshes import ThermalMesh


def plot_showcases(transformer, engine, output, queries=(0, 7, 15), slab=1):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if not queries or len(set(queries)) != len(queries):
        raise ValueError("Choose distinct target indices")
    fig = plt.figure(figsize=(1.9 * (len(queries) + 1), 6.2), constrained_layout=True)
    grid = fig.add_gridspec(4, len(queries) + 1, width_ratios=[1.35] + [1] * len(queries))
    materials_cmap = ListedColormap(["#56b4e9", "#e69f00", "#332288"])
    for row, directory in enumerate((transformer, engine)):
        directory = Path(directory)
        record = json.loads((directory / "record.json").read_text())
        c = record["controls"]
        expected = "transformer_2d" if row == 0 else "engine_3d"
        if c["geometry"] != expected or not record["success"]:
            raise ValueError("Showcase figures require accepted transformer and engine sequences")
        showcase, model = build_model(c)
        if showcase.preparation["input_sha256"] != record["mesh"]["input_sha256"]:
            raise ValueError("Figure reconstruction requires the recorded mesh input bundle")
        mesh = showcase.assembly.mesh
        nodes, cells, free, materials = mesh.nodes, mesh.cells, mesh.free, mesh.materials
        geometry = fig.add_subplot(grid[2 * row : 2 * row + 2, 0], projection="3d" if row else None)
        if row:
            coarse = ThermalMesh.load(Path(__file__).parent / "data" / expected / "mesh.npz")
            gas = coarse.cells[coarse.materials == 0]
            faces = np.concatenate([gas[:, f] for f in combinations(range(4), 3)])
            unique, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
            shell = Poly3DCollection(
                coarse.nodes[unique[counts == 1]],
                facecolor="#66c2a5",
                edgecolor="#447765",
                linewidth=0.2,
                alpha=0.85,
            )
            geometry.add_collection3d(shell)
            cube = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
            for a, b in combinations(cube, 2):
                if np.sum(np.abs(a - b)) == 1:
                    geometry.plot(*np.array([a, b]).T, color=".35", linewidth=0.7)
            geometry.set(
                xlim=(0, 1),
                ylim=(0, 1),
                zlim=(0, 1),
                xlabel="$x_1/L$",
                ylabel="$x_2/L$",
                zlabel="$x_3/L$",
                title="Bore-in-block model",
                box_aspect=(1, 1, 1),
            )
            for spatial_axis in (geometry.xaxis, geometry.yaxis, geometry.zaxis):
                spatial_axis.labelpad = -2
                spatial_axis.label.set_fontsize(8)
            geometry.view_init(elev=22, azim=-52)
            geometry.tick_params(labelsize=6, pad=0)
        else:
            tri = mtri.Triangulation(nodes[:, 1], nodes[:, 0], cells[:, [0, 2, 1]])
            im = geometry.tripcolor(
                tri,
                facecolors=materials,
                cmap=materials_cmap,
                norm=BoundaryNorm(np.arange(4) - 0.5, 3),
                rasterized=True,
            )
            geometry.set(xlabel="$z$ (m)", ylabel="$r$ (m)", title="Oil, winding, and baffles")
            colorbar = fig.colorbar(
                im, ax=geometry, orientation="horizontal", ticks=[0, 1, 2], pad=0.12
            )
            colorbar.ax.set_xticklabels(["Oil", "Winding", "Baffle"], fontsize=9)
        for col, query in enumerate(queries, 1):
            if query not in range(c["targets"]):
                raise ValueError("A requested target lies outside the recorded sequence")
            values = np.zeros((len(nodes), 2))
            selected_slab = slab if c["transient"] else 0
            desired = desired_temperature(model, query, c["targets"])
            if (
                hashlib.sha256(np.ascontiguousarray(desired).tobytes()).hexdigest()
                != record["cases"][query]["target_sha256"]
            ):
                raise ValueError("Figure target differs from its recorded optimization query")
            active = unpack_mask(record["cases"][query]["active_mask_bits"], model.size)
            for k, field in enumerate((desired, active)):
                values[free, k] = field.reshape(-1, len(free))[selected_slab]
            axis = fig.add_subplot(grid[2 * row, col])
            active_axis = fig.add_subplot(grid[2 * row + 1, col])
            if row:
                xy, triangles, sampled, owners = tetrahedral_slice(
                    nodes, cells, values, coordinate=0.5, axis=2
                )
                tri = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles)
                axis.set(xlabel="$x_1/L$", ylabel="$x_2/L$", aspect="equal")
                active_axis.set(xlabel="$x_1/L$", ylabel="$x_2/L$", aspect="equal")
            else:
                tri = mtri.Triangulation(nodes[:, 1], nodes[:, 0], cells[:, [0, 2, 1]])
                sampled = values
                axis.set(xlabel="$z$ (m)", ylabel="$r$ (m)")
                active_axis.set(xlabel="$z$ (m)", ylabel="$r$ (m)")
            artist = axis.tripcolor(
                tri,
                sampled[:, 0],
                shading="gouraud",
                cmap="inferno",
                vmin=0,
                vmax=1.25,
                rasterized=True,
            )
            active_artist = active_axis.tricontourf(
                tri,
                sampled[:, 1],
                levels=[-0.01, 0.5, 1.01],
                colors=["#f2f2f2", "#0072b2"],
            )
            active_axis.set_title(f"Active set, target {query + 1}", fontsize=9)
            title = f"Target {query + 1}"
            if c["transient"]:
                time = (selected_slab + 1) * c["horizon"] / c["slabs"]
                title += (
                    f", $t={time * record['parameters']['physical']['time_scale_s']:.0f}$ s"
                    if not row
                    else f", $t={time:.3f}$"
                )
            axis.set_title(title, fontsize=9)
            if col == len(queries):
                fig.colorbar(artist, ax=axis, label="Normalized desired temperature", shrink=0.8)
                bar = fig.colorbar(active_artist, ax=active_axis, ticks=[0.25, 0.75], shrink=0.8)
                bar.ax.set_yticklabels(["Inactive", "Active"], fontsize=8)
    metadata = (
        {"CreationDate": None, "ModDate": None} if Path(output).suffix.lower() == ".pdf" else None
    )
    fig.savefig(output, dpi=200, metadata=metadata)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, nargs="+", default=[0, 7, 15])
    parser.add_argument("--slab", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new figure output")
    plot_showcases(args.transformer, args.engine, args.output, args.queries, args.slab)


if __name__ == "__main__":
    main()
