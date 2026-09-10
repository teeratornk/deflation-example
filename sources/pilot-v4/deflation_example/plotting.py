"""Optional plots; values and solver status remain available without Matplotlib."""

import numpy as np


def plot_results(output, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(output / "fields-1.npz") as fields:
        n, dim = int(fields["n"]), int(fields["dimension"])
        fig, axes = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
        for ax, key in zip(axes, ["desired", "state", "multiplier"]):
            values = fields[key].reshape((n,) * dim)
            if dim == 3:
                values = values[:, :, n // 2]
            half_step = 0.5 / (n + 1)
            im = ax.imshow(values.T, origin="lower", extent=[half_step, 1 - half_step] * 2)
            ax.set(title=key + (" (central slice)" if dim == 3 else ""), xlabel="x1", ylabel="x2")
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(
            f"{report['configuration']['problem']}, angle=pi/4; direct PDAS: {report['cases'][1]['direct']['status']}"
        )
        fig.savefig(output / "fields.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
    methods = sorted({m for r in report["cases"] for m in r["kernels"]})
    for method in methods:
        rows = [r for r in report["cases"] if method in r["kernels"]]
        ax.plot(
            [r["theta"] for r in rows],
            [r["kernels"][method]["iterations"] for r in rows],
            "o-",
            label=method,
        )
    ax.set(
        xlabel="Target angle (radians)",
        ylabel="Recorded iterations",
        title="Counts are not wall-time speedups",
    )
    if methods:
        ax.legend()
    if not report["success"]:
        ax.text(
            0.5,
            0.5,
            "FAILED CHECKS — inspect results.json",
            transform=ax.transAxes,
            ha="center",
            color="red",
        )
    fig.savefig(output / "iterations.png", dpi=160)
    plt.close(fig)
