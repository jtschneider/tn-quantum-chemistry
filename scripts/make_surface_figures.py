"""Inspect the Experiment 4 ``(R, delta)`` surface before it is handed to a model.

Reads the cached surface store and renders, from cached data only:

* ``figures/h10_surface.{svg,pdf,png}`` -- the FCI energy surface on the left and
  a per-geometry status map on the right.

The status panel is the point of the exercise. A dataset is not ready for a
learning study because it has the right number of rows; it is ready when the
rows that failed, and the rows whose method did not converge, are known and
located. Plotting them beside the energy shows immediately whether failures
cluster in one corner of the surface -- which would bias any split drawn
through it -- or scatter harmlessly.

Run with::

    uv run python scripts/make_surface_figures.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tn_quantum_chemistry.schema import RecordStore, Status

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

#: Sequential ramp for energy: one hue, light to dark, never a rainbow.
ENERGY_CMAP = "Blues_r"

#: Status colours, reserved and never reused as series colours.
STATUS_COLOURS = {
    Status.CONVERGED: "#1baf7a",
    Status.NOT_CONVERGED: "#eda100",
    Status.FAILED: "#e34948",
}
STATUS_LABELS = {
    Status.CONVERGED: "all methods converged",
    Status.NOT_CONVERGED: "some method did not converge",
    Status.FAILED: "some method failed",
}


def load_surface(path: Path):
    by_point: dict[tuple[float, float], dict] = defaultdict(dict)
    for record in RecordStore(path):
        key = (
            float(record.geometry_parameters["R"]),
            float(record.geometry_parameters["delta"]),
        )
        by_point[key][record.method] = record
    return by_point


def _grid(by_point):
    spacings = sorted({R for R, _ in by_point})
    dimerisations = sorted({d for _, d in by_point})
    return spacings, dimerisations


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=3, width=0.8)


def worst_status(records: dict) -> str:
    statuses = {r.status for r in records.values()}
    if Status.FAILED in statuses:
        return Status.FAILED
    if Status.NOT_CONVERGED in statuses:
        return Status.NOT_CONVERGED
    return Status.CONVERGED


def make_figure(by_point, stem: Path) -> list[Path]:
    spacings, dimerisations = _grid(by_point)
    energy = np.full((len(dimerisations), len(spacings)), np.nan)
    status = np.empty_like(energy, dtype=object)

    for j, R in enumerate(spacings):
        for i, d in enumerate(dimerisations):
            records = by_point.get((R, d))
            if not records:
                continue
            fci = records.get("FCI")
            if fci is not None and fci.converged and fci.e_total is not None:
                energy[i, j] = fci.e_total
            status[i, j] = worst_status(records)

    fig, (ax_e, ax_s) = plt.subplots(
        1, 2, figsize=(12.6, 4.9), facecolor=SURFACE,
        gridspec_kw={"wspace": 0.26},
    )

    # ---- energy surface --------------------------------------------------- #
    _style(ax_e)
    mesh = ax_e.pcolormesh(
        spacings, dimerisations, energy, cmap=ENERGY_CMAP, shading="nearest",
    )
    contours = ax_e.contour(
        spacings, dimerisations, energy, levels=10, colors=INK_SECONDARY,
        linewidths=0.5, alpha=0.55,
        # Every energy here is negative, and matplotlib dashes negative levels
        # by default -- which would dash every contour on the plot and read as
        # noise rather than as iso-energy lines.
        linestyles="solid",
    )
    ax_e.clabel(contours, inline=True, fontsize=6.5, fmt="%.2f")
    bar = fig.colorbar(mesh, ax=ax_e, pad=0.02)
    bar.set_label("E$_{FCI}$  (Hartree)", color=INK_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    bar.outline.set_visible(False)
    ax_e.axhline(0.0, color=INK, linewidth=0.9, alpha=0.5)
    ax_e.set_title("FCI energy surface", color=INK, fontsize=12, loc="left", pad=10)

    # ---- status map ------------------------------------------------------- #
    _style(ax_s)
    for j, R in enumerate(spacings):
        for i, d in enumerate(dimerisations):
            state = status[i, j]
            if state is None:
                continue
            ax_s.plot(
                R, d, "s", markersize=13, color=STATUS_COLOURS[state],
                markeredgecolor=SURFACE, markeredgewidth=1.6,
            )
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=8,
                   color=STATUS_COLOURS[s], label=STATUS_LABELS[s])
        for s in (Status.CONVERGED, Status.NOT_CONVERGED, Status.FAILED)
    ]
    legend = ax_s.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
        ncol=3, frameon=False, fontsize=8.5,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    ax_s.axhline(0.0, color=INK, linewidth=0.9, alpha=0.5)
    ax_s.set_title("Per-geometry status", color=INK, fontsize=12, loc="left", pad=10)

    for ax in (ax_e, ax_s):
        ax.set_xlabel("spacing R  (Å)", color=INK_SECONDARY, fontsize=10)
        ax.set_ylabel("dimerisation δ  (Å)", color=INK_SECONDARY, fontsize=10)
        ax.set_xlim(min(spacings) - 0.12, max(spacings) + 0.12)
        ax.set_ylim(min(dimerisations) - 0.1, max(dimerisations) + 0.1)

    fig.suptitle(
        "H$_{10}$ (R, δ) surface, STO-3G — bonds alternate R ± δ/2",
        color=INK, fontsize=13, x=0.055, ha="left", y=0.99,
    )
    fig.text(
        0.055, 0.93,
        "δ < 0 gives five short bonds (five H₂); δ > 0 gives four short bonds "
        "(four H₂ and two unpaired atoms). The chain has nine bonds, an odd\n"
        "number, so the two signs are different systems and the surface is not "
        "symmetric about δ = 0. FCI is exact only for this basis, electron "
        "number and symmetry sector.",
        color=INK_SECONDARY, fontsize=8.5, ha="left", va="top", linespacing=1.5,
    )
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.06, right=0.97)

    written = []
    for suffix in ("svg", "pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight",
                    format=suffix)
        written.append(path)
    plt.close(fig)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path,
        default=Path("data/raw/h10_surface/h10_sto-3g_surface.jsonl"),
    )
    parser.add_argument("--figdir", type=Path, default=Path("figures"))
    args = parser.parse_args(argv)

    if not args.records.exists():
        parser.error(
            f"{args.records} not found. Run scripts/generate_h10_surface.py first; "
            "this script never launches calculations of its own."
        )
    args.figdir.mkdir(parents=True, exist_ok=True)

    by_point = load_surface(args.records)
    spacings, dimerisations = _grid(by_point)
    written = make_figure(by_point, args.figdir / "h10_surface")

    counts = defaultdict(int)
    for records in by_point.values():
        counts[worst_status(records)] += 1
    print(f"{len(by_point)} geometries ({len(spacings)} R x {len(dimerisations)} δ)")
    for state in (Status.CONVERGED, Status.NOT_CONVERGED, Status.FAILED):
        print(f"  {STATUS_LABELS[state]:<32} {counts[state]:>3}")
    for path in written:
        print(f"  {path}  ({path.stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
