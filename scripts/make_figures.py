"""Experiment 2 figures, built only from cached data.

This script performs no electronic-structure calculation. It reads the record
store written by ``generate_h10_curve.py`` and renders:

* ``figures/h10_dissociation.{svg,pdf,png}`` -- energy (left) and signed error
  against FCI (right), side by side;
* ``figures/h10_walltime.{svg,pdf,png}`` -- cost, kept on its own axes so it is
  never conflated with accuracy;

Vector formats are the deliverable; the PNG exists so the figure can be eyeballed
without a viewer.
* ``figures/h10_curve.csv`` -- the table view, which is also what makes the
  low-contrast series legible for readers who cannot separate them by hue.

Run with::

    uv run python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tn_quantum_chemistry.methods import (
    reference_convergence_floor,
)
from tn_quantum_chemistry.schema import RecordStore, Status

# Validated categorical slots (light surface). Colour follows the method, and
# the two DMRG entries share one hue because they are one method at two bond
# dimensions, not two methods.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = {
    "RHF": "#2a78d6",
    "MP2": "#eb6834",
    "CCSD": "#1baf7a",
    "CCSD(T)": "#eda100",
    "DMRG": "#e87ba4",
}
FCI_INK = "#0b0b0b"

#: Below this the signed-error axis is linear. Set at the FCI reference's own
#: convergence floor: nothing smaller than this is resolvable against FCI, so
#: compressing it to the zero line is honest rather than merely tidy.
#:
#: Lowered from 1e-9 when the Davidson settings on both solvers were tightened
#: (FCI conv_tol 1e-14 with max_space 30, DMRG thresholds 1e-14). The floor
#: itself fell from 2e-9 to 3e-12, and the band is drawn at
#: ``max(floor, LINTHRESH)`` -- so leaving this at 1e-9 would have drawn a band
#: three orders of magnitude wider than the caption's stated floor.
LINTHRESH = 1e-12

#: Sparse decade ticks. The default symlog locator emits a label per decade on
#: both branches, which here is twenty labels across mostly empty axis.
ERROR_TICKS = (1e0, 1e-2, 1e-4, 1e-6, 1e-9, 0.0, -1e-9, -1e-6, -1e-4, -1e-2, -1e0)


def load_curve(path: Path) -> dict[float, dict[str, object]]:
    by_geometry: dict[float, dict[str, object]] = defaultdict(dict)
    for record in RecordStore(path):
        spacing = float(record.geometry_parameters["R"])
        by_geometry[spacing][record.method] = record
    return dict(sorted(by_geometry.items()))


def series_points(
    curve: dict, method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spacings, energies and a converged mask for one method."""
    spacings, energies, converged = [], [], []
    for spacing, records in curve.items():
        record = records.get(method)
        if record is None or record.e_total is None:
            continue
        spacings.append(spacing)
        energies.append(record.e_total)
        converged.append(record.status == Status.CONVERGED)
    return np.array(spacings), np.array(energies), np.array(converged, dtype=bool)


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=3, width=0.8)


def _plot(ax, x, y, mask, color, label, *, dashed=False, linewidth=2.0):
    """One series, with unconverged points drawn as open markers."""
    ax.plot(x, y, color=color, linewidth=linewidth, label=label,
            linestyle="--" if dashed else "-", zorder=3, solid_capstyle="round")
    if mask.any():
        ax.plot(x[mask], y[mask], "o", color=color, markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
    if (~mask).any():
        ax.plot(x[~mask], y[~mask], "o", markerfacecolor=SURFACE, markersize=5.5,
                markeredgecolor=color, markeredgewidth=1.6, zorder=5)


def _direct_labels(ax, entries, x_at, *, dx=0.06, gap=0.055) -> None:
    """Right-edge labels, nudged apart so none collide.

    Required rather than decorative: three of the series colours sit below 3:1
    contrast on this surface, so identity must not rest on hue alone.

    Spacing is done in axes fraction. Doing it in data units breaks on a symlog
    axis, where equal data gaps are wildly unequal on screen and the collision
    pushes cascade into a pile.
    """
    from matplotlib.transforms import blended_transform_factory

    to_axes = ax.transAxes.inverted()
    positioned = []
    for name, y, color, text in entries:
        frac = float(to_axes.transform(ax.transData.transform((x_at, y)))[1])
        positioned.append((frac, name, y, color, text))
    positioned.sort(key=lambda item: -item[0])

    placed: list[float] = []
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for frac, _, y, color, text in positioned:
        target = min(max(frac, 0.0), 1.0)
        for previous in placed:
            if abs(target - previous) < gap:
                target = previous - gap
        placed.append(target)
        ax.annotate(text, xy=(x_at, y), xycoords="data",
                    xytext=(x_at + dx, target), textcoords=trans,
                    color=color, fontsize=8.5, va="center", ha="left",
                    annotation_clip=False)


def make_curve_figure(curve: dict, outpath: Path, dmrg_max: int, dmrg_low: int) -> None:
    fig, (ax_energy, ax_error) = plt.subplots(
        1, 2, figsize=(12.4, 5.8), facecolor=SURFACE,
        gridspec_kw={"wspace": 0.30, "width_ratios": [1, 1]},
    )

    best_dmrg, low_dmrg = f"DMRG(chi={dmrg_max})", f"DMRG(chi={dmrg_low})"
    ladder = ["RHF", "MP2", "CCSD", "CCSD(T)"]

    # ---------------- left: total energy -------------------------------- #
    _style_axes(ax_energy)
    labels = []
    for method in ladder:
        x, y, mask = series_points(curve, method)
        _plot(ax_energy, x, y, mask, SERIES[method], method)
        labels.append((method, y[-1], SERIES[method], method))

    x_fci, y_fci, mask_fci = series_points(curve, "FCI")
    _plot(ax_energy, x_fci, y_fci, mask_fci, FCI_INK, "FCI (reference)", linewidth=2.4)
    labels.append(("FCI", y_fci[-1], FCI_INK, "FCI"))

    x_d, y_d, mask_d = series_points(curve, best_dmrg)
    _plot(ax_energy, x_d, y_d, mask_d, SERIES["DMRG"], f"DMRG χ={dmrg_max}",
          linewidth=2.0)
    labels.append(("DMRG", y_d[-1], SERIES["DMRG"], f"DMRG χ={dmrg_max}"))

    ax_energy.set_xlabel("nearest-neighbour spacing R  (Å)", color=INK_SECONDARY,
                         fontsize=10)
    ax_energy.set_ylabel("total energy  (Hartree)", color=INK_SECONDARY, fontsize=10)
    ax_energy.set_title("Total energy", color=INK, fontsize=12,
                        loc="left", pad=10)
    ax_energy.set_xlim(min(curve) - 0.1, max(curve) + 0.75)
    _direct_labels(ax_energy, labels, max(curve))

    # ---------------- right: signed error against FCI --------------------- #
    _style_axes(ax_error)
    fci_by_spacing = dict(zip(x_fci, y_fci))

    floor = max(
        reference_convergence_floor(list(records.values()))
        for records in curve.values()
    )
    ax_error.axhspan(-max(floor, LINTHRESH), max(floor, LINTHRESH),
                     color=GRID, alpha=0.6, zorder=0)
    ax_error.axhline(0.0, color=FCI_INK, linewidth=1.4, zorder=2)

    error_labels = []
    for method, dashed in [(m, False) for m in ladder] + [
        (low_dmrg, True), (best_dmrg, False)
    ]:
        x, y, mask = series_points(curve, method)
        if x.size == 0:
            continue
        errors = np.array([e - fci_by_spacing[s] for s, e in zip(x, y)])
        color = SERIES["DMRG"] if method.startswith("DMRG") else SERIES[method]
        text = (
            f"DMRG χ={method.split('=')[1].rstrip(')')}"
            if method.startswith("DMRG(chi=")
            else method
        )
        _plot(ax_error, x, errors, mask, color, text, dashed=dashed,
              linewidth=1.7 if dashed else 2.0)
        error_labels.append((method, errors[-1], color, text))

    ax_error.set_yscale("symlog", linthresh=LINTHRESH)
    ax_error.set_yticks(ERROR_TICKS)
    ax_error.set_yticklabels([
        "0" if t == 0 else f"{'−' if t < 0 else '+'}10$^{{{int(np.log10(abs(t)))}}}$"
        for t in ERROR_TICKS
    ])
    ax_error.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_error.set_xlabel("nearest-neighbour spacing R  (Å)", color=INK_SECONDARY,
                        fontsize=10)
    ax_error.set_ylabel("E − E$_{FCI}$  (Hartree, symlog)", color=INK_SECONDARY,
                       fontsize=10)
    ax_error.set_title("Signed error against FCI", color=INK, fontsize=12,
                       loc="left", pad=10)
    ax_error.set_xlim(min(curve) - 0.1, max(curve) + 0.75)
    _direct_labels(ax_error, error_labels, max(curve))

    # Both explanatory notes live in the caption rather than on the axes: the
    # panel has no region that stays empty across the whole range, and the
    # caption is where the limitations belong in any case.

    handles, names = ax_energy.get_legend_handles_labels()
    h2, n2 = ax_error.get_legend_handles_labels()
    for handle, name in zip(h2, n2):
        if name not in names:
            handles.append(handle)
            names.append(name)
    legend = fig.legend(handles, names, loc="lower center", ncol=7, frameon=False,
                        fontsize=9, bbox_to_anchor=(0.5, -0.015))
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.suptitle(
        "H$_{10}$ dissociation, STO-3G, δ = 0 — one finite Hamiltonian",
        color=INK, fontsize=13.5, x=0.055, ha="left", y=0.985,
    )
    fig.text(
        0.055, 0.930,
        "Open markers mark calculations that did not converge; their energies are "
        "plotted but are not converged results.\n"
        "Right: negative values lie below FCI — for a fixed finite Hamiltonian FCI is "
        "exact, so this is a failure, not an improvement. The shaded band is the FCI\n"
        f"reference's own convergence floor ({floor:.0e} Ha), inside which no difference "
        "is resolved. FCI is exact only for this basis, electron number and symmetry\n"
        "sector; these are not chemically converged bond energies. Wall time is "
        "reported separately, in figures/h10_walltime.png.",
        color=INK_SECONDARY, fontsize=8.6, ha="left", va="top", linespacing=1.5,
    )
    fig.subplots_adjust(top=0.745, bottom=0.16, left=0.07, right=0.97)
    _save(fig, outpath)


def _save(fig, stem: Path, formats: tuple[str, ...] = ("svg", "pdf", "png")) -> list[Path]:
    """Write one figure in each format. Vector first; the PNG is for eyeballing."""
    written = []
    for suffix in formats:
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight",
                    format=suffix)
        written.append(path)
    plt.close(fig)
    return written


def make_walltime_figure(curve: dict, outpath: Path, dmrg_max: int) -> None:
    """Cost, on its own axes. Kept apart from accuracy on purpose."""
    fig, ax = plt.subplots(figsize=(6.6, 4.0), facecolor=SURFACE)
    _style_axes(ax)
    labels = []
    for method in ["RHF", "MP2", "CCSD", "CCSD(T)"]:
        spacings = np.array(list(curve))
        times = np.array([
            curve[s][method].walltime_seconds if method in curve[s] else np.nan
            for s in curve
        ], dtype=float)
        ax.plot(spacings, times, color=SERIES[method], linewidth=2.0, marker="o",
                markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.0)
        labels.append((method, times[-1], SERIES[method], method))

    for method, color, text in [("FCI", FCI_INK, "FCI"),
                                (f"DMRG(chi={dmrg_max})", SERIES["DMRG"],
                                 f"DMRG χ={dmrg_max}")]:
        spacings = np.array(list(curve))
        times = np.array([
            curve[s][method].walltime_seconds if method in curve[s] else np.nan
            for s in curve
        ], dtype=float)
        ax.plot(spacings, times, color=color, linewidth=2.2, marker="o",
                markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.0)
        labels.append((method, times[-1], color, text))

    ax.set_yscale("log")
    ax.set_xlabel("nearest-neighbour spacing R  (Å)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("wall time  (s)", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Cost per geometry", color=INK, fontsize=12, fontweight="semibold",
                 loc="left", pad=10)
    ax.set_xlim(min(curve) - 0.1, max(curve) + 0.8)
    _direct_labels(ax, labels, max(curve))
    fig.text(0.01, -0.02,
             "Single machine, 4 threads. Comparable within this figure only.",
             color=INK_SECONDARY, fontsize=8)
    _save(fig, outpath)


def write_table(curve: dict, outpath: Path) -> None:
    """The table view. Also the relief for the low-contrast series colours."""
    methods: list[str] = []
    for records in curve.values():
        for method in records:
            if method not in methods:
                methods.append(method)
    with outpath.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "R_angstrom", "method", "status", "e_total_hartree",
            "error_vs_fci_hartree", "walltime_seconds",
        ])
        for spacing, records in curve.items():
            e_fci = records["FCI"].e_total if "FCI" in records else None
            for method in methods:
                record = records.get(method)
                if record is None:
                    continue
                error = (
                    "" if record.e_total is None or e_fci is None
                    else f"{record.e_total - e_fci:.6e}"
                )
                writer.writerow([
                    f"{spacing:g}", method, record.status,
                    "" if record.e_total is None else f"{record.e_total:.12f}",
                    error,
                    "" if record.walltime_seconds is None
                    else f"{record.walltime_seconds:.2f}",
                ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path,
        default=Path("data/raw/h10_curve/h10_sto-3g_curve.jsonl"),
    )
    parser.add_argument("--figdir", type=Path, default=Path("figures"))
    args = parser.parse_args(argv)

    if not args.records.exists():
        parser.error(
            f"{args.records} not found. Run scripts/generate_h10_curve.py first; "
            "this script never launches calculations of its own."
        )
    args.figdir.mkdir(parents=True, exist_ok=True)

    curve = load_curve(args.records)
    chis = sorted({
        int(m.split("=")[1].rstrip(")"))
        for records in curve.values() for m in records if m.startswith("DMRG(chi=")
    })
    dmrg_low, dmrg_max = min(chis), max(chis)

    figure = args.figdir / "h10_dissociation"
    make_curve_figure(curve, figure, dmrg_max, dmrg_low)
    walltime = args.figdir / "h10_walltime"
    make_walltime_figure(curve, walltime, dmrg_max)
    table = args.figdir / "h10_curve.csv"
    write_table(curve, table)

    print(f"{len(curve)} geometries, DMRG sequence χ = {chis}")
    for stem in (figure, walltime):
        for suffix in ("svg", "pdf", "png"):
            path = stem.with_suffix(f".{suffix}")
            print(f"  {path}  ({path.stat().st_size / 1024:.0f} kB)")
    print(f"  {table}  ({table.stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
