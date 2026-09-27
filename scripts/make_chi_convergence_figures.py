"""Experiment 6 figures: how the DMRG error decays with bond dimension.

Reads the store written by ``chi_convergence.py`` and renders:

* ``figures/h10_chi_convergence.{svg,pdf,png}`` -- small multiples, one panel per
  chain spacing, showing the error against FCI as a function of ``chi`` with the
  seed-to-seed spread as a band;
* ``figures/h10_chi_summary.{svg,pdf,png}`` -- the fitted decay exponent and the
  converged bipartite entanglement, each on its own axes against spacing;
* ``figures/h10_chi_convergence.csv`` -- the table view.

Small multiples rather than six lines on shared axes: spacing is a *continuous*
parameter, so it would want a sequential ramp, and one hue cannot supply six
ordinally distinguishable steps at this surface. Faceting sidesteps the problem
and shows the shape comparison -- whether the decay rate changes with geometry --
more directly than six overlapping curves.

This script performs no electronic-structure calculation.

    uv run python scripts/make_chi_convergence_figures.py
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from tn_quantum_chemistry.schema import RecordStore, Status

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

#: Floor of the FCI reference itself. A DMRG error below this is not resolvable
#: against the reference and is drawn as an upper bound, not a measurement.
REFERENCE_FLOOR = 2.3e-12

#: An error at or below this counts as converged for the purpose of reporting
#: chi*. Set well above the reference floor so chi* measures the MPS rather than
#: the last digits of the FCI comparison.
CONVERGED_TOL = 1e-9


def load(path: Path, cutoff: float) -> dict[float, dict[int, list[dict]]]:
    """{R: {chi: [per-seed rows]}} for one series in the store.

    The store holds runs at the same nominal chi with and without a
    singular-value cutoff. They are different calculations, so a series is
    selected explicitly rather than pooled -- pooling them would silently
    average two settings and present the result as seed spread.
    """
    out: dict[float, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in RecordStore(path):
        if record.status != Status.CONVERGED:
            continue
        if record.dmrg.get("cutoff", 1e-20) != cutoff:
            continue
        R = float(record.geometry_parameters["R"])
        out[R][int(record.dmrg["requested_bond_dim"])].append(
            {
                "seed": record.dmrg["seed"],
                "effective_chi": record.dmrg.get("effective_bond_dim"),
                "error": abs(record.fci_discrepancy),
                "s_max": record.extra["max_bipartite_entanglement"],
                "dw": record.dmrg["final_discarded_weight"],
                "walltime": record.walltime_seconds,
            }
        )
    return {R: dict(sorted(chis.items())) for R, chis in sorted(out.items())}


def converged_chi(chis: list[int], errors: list[float], tol: float) -> int | None:
    """Smallest chi whose error is within ``tol``, or None if the grid never gets there.

    This replaces a fitted power-law exponent, which these curves do not
    support: the error is nearly flat in chi and then falls several orders of
    magnitude between two adjacent grid points. A single slope through that
    shape describes neither regime, and quoting one would imply a smooth
    variational decay that the data does not show.
    """
    for chi, error in sorted(zip(chis, errors)):
        if error <= tol:
            return chi
    return None


def fit_exponent(chis: list[int], errors: list[float]) -> tuple[float, float]:
    """Least-squares slope of log(error) against log(chi), pre-cliff only.

    Restricted to points well above the reference floor, so the fit describes
    the regime where truncation still limits the answer. Returned positive: a
    larger value means a faster decay. Reported as a secondary diagnostic, never
    as the headline -- see :func:`converged_chi`.
    """
    pairs = [(c, e) for c, e in zip(chis, errors) if e > 1e3 * REFERENCE_FLOOR]
    if len(pairs) < 3:
        return float("nan"), float("nan")
    x = np.log(np.array([c for c, _ in pairs], dtype=float))
    y = np.log(np.array([e for _, e in pairs], dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return -float(slope), r2


def seed_spread(by_chi: dict[int, list[dict]]) -> float:
    """Worst max/min error ratio across seeds, above the reference floor only.

    Restricted to bond dimensions whose median error still exceeds
    ``CONVERGED_TOL``. Once a run reaches the floor its remaining error is the
    FCI comparison's own last digits, and ratios there are large but meaningless
    -- 1e-14 against 5.7e-13 is a factor of 57 of pure noise. Including those
    points turned a 2.9x spread into an apparent 57x one.
    """
    ratios = [
        max(r["error"] for r in rows) / min(r["error"] for r in rows)
        for rows in by_chi.values()
        if min(r["error"] for r in rows) > 0
        and float(np.median([r["error"] for r in rows])) > CONVERGED_TOL
    ]
    return max(ratios) if ratios else float("nan")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)


def figure_small_multiples(data: dict, out: Path, stem: str) -> dict[float, tuple]:
    n = len(data)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(9.0, 3.0 * nrows), sharex=True, sharey=True
    )
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes).ravel()
    fits: dict[float, tuple] = {}

    for ax, (R, by_chi) in zip(axes, data.items()):
        _style(ax)
        chis = sorted(by_chi)
        lo = [min(r["error"] for r in by_chi[c]) for c in chis]
        hi = [max(r["error"] for r in by_chi[c]) for c in chis]
        med = [float(np.median([r["error"] for r in by_chi[c]])) for c in chis]

        ax.fill_between(chis, lo, hi, color=BLUE, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(chis, med, color=BLUE, linewidth=2.0, zorder=3,
                marker="o", markersize=4.5, markeredgecolor=SURFACE,
                markeredgewidth=1.2)

        exponent, r2 = fit_exponent(chis, med)
        chi_star = converged_chi(chis, med, CONVERGED_TOL)
        fits[R] = (exponent, r2, chi_star, chis, med)
        ax.axhline(REFERENCE_FLOOR, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([16, 32, 64, 128, 256, 512])
        ax.set_xticklabels(["16", "32", "64", "128", "256", "512"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(f"R = {R:.1f} Å", color=INK, fontsize=10, loc="left", pad=6)
        label = f"χ* = {chi_star}" if chi_star else f"χ* > {max(chis)}"
        # Bottom-left: these curves run from top-left to bottom-right, so it
        # is the one corner that never collides with the data or with the chi*
        # rule drawn below.
        ax.text(0.05, 0.07, label, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9, color=INK)
        if chi_star:
            ax.axvline(chi_star, color=INK_MUTED, linewidth=1.0,
                       linestyle=(0, (2, 3)), zorder=1)

    for ax in axes[n:]:
        ax.set_visible(False)
    for i, ax in enumerate(axes[:n]):
        if i % ncols == 0:
            ax.set_ylabel("|E − E$_{FCI}$|  (Ha)", color=INK_SECONDARY, fontsize=9)
        if i >= n - ncols:
            ax.set_xlabel("bond dimension χ", color=INK_SECONDARY, fontsize=9)

    fig.suptitle(
        "DMRG error against FCI versus bond dimension, H$_{10}$/STO-3G",
        color=INK, fontsize=11.5, x=0.055, ha="left", y=0.985,
    )
    fig.text(
        0.055, 0.945,
        "Line is the median of three MPS seeds, band is their full range. "
        "Dashed line is the two-solver floor (2.3×10⁻¹² Ha).",
        color=INK_SECONDARY, fontsize=8.5, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return fits


def figure_summary(data: dict, fits: dict, out: Path, stem: str) -> None:
    """chi* against spacing, beside the entanglement entropy of the same states.

    Two panels rather than two y-scales on one: they are different quantities
    and overlaying them would invite reading a correlation off the geometry of
    the plot. The point of the pair is that there is no such correlation --
    between the entropy and the bond dimension needed for a fixed *energy*
    accuracy. At fixed *truncation* error the entropy does predict the required
    number of states, as MPS theory requires; the two differ because the energy
    carried per unit of discarded weight varies by 133x across this surface.
    """
    spacings = sorted(data)
    ceiling = max(max(data[R]) for R in spacings)
    chi_star = [fits[R][2] for R in spacings]
    unreached = [c is None for c in chi_star]
    plotted = [ceiling * 1.35 if c is None else c for c in chi_star]
    s_conv = [max(r["s_max"] for r in data[R][max(data[R])]) for R in spacings]

    spread = [seed_spread(data[R]) for R in spacings]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.5, 3.4))
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax1, ax2, ax3):
        _style(ax)
        ax.set_xlabel("spacing R  (Å)", color=INK_SECONDARY, fontsize=9)

    ax1.plot(spacings, plotted, color=BLUE, linewidth=2.0, zorder=3)
    for x, y, miss in zip(spacings, plotted, unreached):
        ax1.plot([x], [y], marker="^" if miss else "o", markersize=7 if miss else 6,
                 color=BLUE, markerfacecolor=SURFACE if miss else BLUE,
                 markeredgecolor=BLUE, markeredgewidth=1.6, zorder=4)
    ax1.axhline(ceiling, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax1.text(spacings[-1], ceiling * 0.93, f"grid ceiling χ = {ceiling}",
             color=INK_MUTED, fontsize=8, va="top", ha="right")
    ax1.set_yscale("log")
    ax1.set_yticks([16, 32, 64, 128, 192, 256, 384, 512])
    ax1.set_yticklabels(["16", "32", "64", "128", "192", "256", "384", "512"])
    ax1.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax1.set_ylabel("χ* to reach 10⁻⁹ Ha", color=INK_SECONDARY, fontsize=9)
    ax1.set_title("Bond dimension needed", color=INK, fontsize=10, loc="left", pad=6)
    if any(unreached):
        ax1.text(0.97, 0.06, "△ open marker: not reached on this grid",
                 transform=ax1.transAxes, ha="right", fontsize=8, color=INK_SECONDARY)

    ax2.plot(spacings, s_conv, color=ORANGE, linewidth=2.0, marker="o",
             markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax2.set_ylabel("max bipartite entropy  S", color=INK_SECONDARY, fontsize=9)
    ax2.set_title("Entanglement of the same states", color=INK, fontsize=10,
                  loc="left", pad=6)

    ax3.plot(spacings, spread, color=INK_SECONDARY, linewidth=2.0, marker="o",
             markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax3.set_yscale("log")
    ax3.set_ylabel("worst seed spread  (max/min)", color=INK_SECONDARY, fontsize=9)
    ax3.set_title("Reproducibility across MPS seeds", color=INK, fontsize=10,
                  loc="left", pad=6)
    ax3.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))

    fig.suptitle(
        "Bond dimension for a fixed energy target is not set by the entropy",
        color=INK, fontsize=11.5, x=0.055, ha="left", y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def write_csv(data: dict, fits: dict, path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "R_angstrom", "chi", "n_seeds", "error_min_Ha", "error_median_Ha",
            "error_max_Ha", "seed_spread_ratio", "max_bipartite_entropy_min",
            "max_bipartite_entropy_max", "final_discarded_weight_median",
            "walltime_median_s",
        ])
        for R, by_chi in data.items():
            for chi, rows in by_chi.items():
                errors = [r["error"] for r in rows]
                s = [r["s_max"] for r in rows]
                writer.writerow([
                    f"{R:g}", chi, len(rows),
                    f"{min(errors):.6e}", f"{np.median(errors):.6e}",
                    f"{max(errors):.6e}",
                    f"{max(errors) / min(errors):.2f}" if min(errors) > 0 else "inf",
                    f"{min(s):.4f}", f"{max(s):.4f}",
                    f"{np.median([r['dw'] for r in rows]):.3e}",
                    f"{np.median([r['walltime'] for r in rows]):.2f}",
                ])
        writer.writerow([])
        writer.writerow([
            "R_angstrom", "chi_star_to_1e-9_Ha", "precliff_exponent_a", "r_squared",
        ])
        for R in sorted(fits):
            exponent, r2, chi_star, *_ = fits[R]
            writer.writerow([
                f"{R:g}",
                "not reached" if chi_star is None else chi_star,
                f"{exponent:.3f}", f"{r2:.4f}",
            ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store", type=Path,
        default=Path("data/raw/chi_convergence/h10_sto-3g.jsonl"),
    )
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--cutoff", type=float, default=1e-20,
        help="which series to plot: the singular-value cutoff its runs used "
             "(default 1e-20, the runs with no effective cutoff)",
    )
    parser.add_argument("--stem", default="h10_chi_convergence")
    args = parser.parse_args(argv)

    data = load(args.store, args.cutoff)
    if not data:
        print(f"no converged records with cutoff={args.cutoff:g} in {args.store}")
        return 1
    args.outdir.mkdir(parents=True, exist_ok=True)

    fits = figure_small_multiples(data, args.outdir, args.stem)
    figure_summary(data, fits, args.outdir, args.stem.replace("convergence", "summary"))
    write_csv(data, fits, args.outdir / f"{args.stem}.csv")

    print(f"{'R':>5} {'chi*':>7} {'S':>7} {'a':>6} {'r2':>6} {'max seed spread':>16}")
    for R, by_chi in data.items():
        exponent, r2, chi_star, *_ = fits[R]
        spread = seed_spread(by_chi)
        s_conv = max(r["s_max"] for r in by_chi[max(by_chi)])
        star = "> %d" % max(by_chi) if chi_star is None else str(chi_star)
        print(f"{R:>5.1f} {star:>7} {s_conv:>7.3f} {exponent:>6.2f} {r2:>6.3f} "
              f"{spread:>15.1f}x")
    print(f"\nwrote {args.outdir}/{args.stem}.{{svg,pdf,png,csv}} and "
          f"{args.stem.replace('convergence', 'summary')}.{{svg,pdf,png}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
