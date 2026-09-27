"""Experiment 3 figures, built only from cached data.

Renders the two figures the entanglement bridge calls for:

* ``figures/h10_orbital_entropy.{svg,pdf,png}`` -- the single-orbital entropy
  ``s_i`` at three correlation regimes, on one axis so the growth is visible;
* ``figures/h10_mutual_information.{svg,pdf,png}`` -- the ``I_ij`` matrix at the
  same three geometries, as heatmaps on a shared colour scale.

This script performs no electronic-structure calculation.

Run with::

    uv run python scripts/make_entanglement_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

#: One categorical hue per geometry, from the validated palette.
REGIME_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")

#: Magnitude, so a single-hue sequential ramp. Never a rainbow.
INFORMATION_CMAP = "PuBu"

MAX_SINGLE_ORBITAL_ENTROPY = float(np.log(4.0))


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=3, width=0.8)


def _save(fig, stem: Path) -> list[Path]:
    written = []
    for suffix in ("svg", "pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight",
                    format=suffix)
        written.append(path)
    plt.close(fig)
    return written


def make_entropy_figure(results: dict, stem: Path) -> list[Path]:
    """``s_i`` in two orbital bases, side by side.

    The second panel is the point, and it says more than "the values differ".
    The two bases swap roles across the curve: mean ``s_i`` runs 0.25 -> 1.34 in
    canonical orbitals as the chain is stretched, and 1.35 -> 0.74 in localised
    ones, crossing near R = 1.8 A. Near-ln-4 values appear in whichever basis is
    the *wrong* one for the regime -- delocalised orbitals for a Mott-like
    stretched chain, localised orbitals for a compressed metallic one.

    So a saturated ``s_i`` diagnoses an unsuitable orbital partition, not a
    state beyond the reach of the method.
    """
    entries = sorted(results.items(), key=lambda kv: float(kv[0]))
    has_localised = all(
        entry.get("localised_single_orbital_entropy") for _, entry in entries
    )
    n_panels = 2 if has_localised else 1

    fig, axes = plt.subplots(
        1, n_panels, figsize=(6.4 * n_panels, 5.3), facecolor=SURFACE,
        sharey=True, gridspec_kw={"wspace": 0.08},
    )
    axes = np.atleast_1d(axes)

    panels = [("single_orbital_entropy", "canonical RHF orbitals")]
    if has_localised:
        panels.append(
            ("localised_single_orbital_entropy", "Löwdin AOs — one per atom")
        )

    for ax, (field, title) in zip(axes, panels):
        _style(ax)
        for colour, (_, entry) in zip(REGIME_COLOURS, entries):
            entropies = np.asarray(entry[field])
            label = f"R = {entry['geometry']['R']:g} Å — {entry['geometry']['regime']}"
            ax.plot(np.arange(1, len(entropies) + 1), entropies, "-o", color=colour,
                    linewidth=2.0, markersize=5, markeredgecolor=SURFACE,
                    markeredgewidth=1.2, label=label)

        # ln 4 is labelled above its line and ln 2 below, because a curve sits
        # just above ln 2 in each panel (green on the right, blue nowhere near
        # on the left) and a label on that side collides with it.
        for value, text, side in (
            (MAX_SINGLE_ORBITAL_ENTROPY, "ln 4 — maximum for one spatial orbital",
             "bottom"),
            (float(np.log(2.0)), "ln 2 — one electron per orbital", "top"),
        ):
            ax.axhline(value, color=INK_SECONDARY, linewidth=1.0,
                       alpha=1.0 if value > 1.0 else 0.55)
            offset = 0.012 if side == "bottom" else -0.012
            ax.text(0.985, value / (MAX_SINGLE_ORBITAL_ENTROPY * 1.06) + offset, text,
                    transform=ax.transAxes, color=INK_SECONDARY, fontsize=8,
                    va=side, ha="right")

        ax.set_xlabel("orbital index", color=INK_SECONDARY, fontsize=10)
        ax.set_xticks(np.arange(1, 11))
        ax.set_ylim(0, MAX_SINGLE_ORBITAL_ENTROPY * 1.06)
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)

    axes[0].set_ylabel("single-orbital entropy  s$_i$", color=INK_SECONDARY,
                       fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, frameon=False, fontsize=9,
                        loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.suptitle("H$_{10}$/STO-3G single-orbital entropy — the same states in two "
                 "orbital bases", color=INK, fontsize=12.5, x=0.02, ha="left", y=1.03)
    fig.text(0.02, 0.985,
             "The same three states in both panels, at identical energy — an "
             "orbital rotation cannot move it. Mean s$_i$ runs 0.25 → 1.34 in "
             "canonical orbitals\nand 1.35 → 0.74 in localised ones: the bases "
             "swap roles, crossing near R = 1.8 Å. Near-ln-4 values appear in "
             "whichever basis is wrong for the regime —\ndelocalised orbitals for "
             "a stretched Mott-like chain, localised orbitals for a compressed "
             "metallic one. Saturation therefore diagnoses an unsuitable orbital\n"
             "partition, not a state beyond the method: DMRG reproduces exact FCI "
             "to 4 × 10⁻¹¹ Ha at every point shown, with no bond-dimension "
             "truncation at all.",
             color=INK_SECONDARY, fontsize=8.5, ha="left", va="top", linespacing=1.5,
             transform=fig.transFigure)
    fig.subplots_adjust(top=0.70, bottom=0.14)
    return _save(fig, stem)


def make_information_figure(results: dict, stem: Path) -> list[Path]:
    entries = sorted(results.items(), key=lambda kv: float(kv[0]))
    matrices = [np.asarray(e["mutual_information"]) for _, e in entries]
    vmax = max(m.max() for m in matrices)

    fig, axes = plt.subplots(
        1, len(entries), figsize=(4.0 * len(entries) + 1.4, 4.3), facecolor=SURFACE,
        gridspec_kw={"wspace": 0.22},
    )
    axes = np.atleast_1d(axes)

    for ax, matrix, (_, entry) in zip(axes, matrices, entries):
        _style(ax)
        n = matrix.shape[0]
        mesh = ax.imshow(
            matrix, cmap=INFORMATION_CMAP, vmin=0.0, vmax=vmax,
            origin="upper", extent=(0.5, n + 0.5, n + 0.5, 0.5),
        )
        ax.set_xticks(np.arange(1, n + 1))
        ax.set_yticks(np.arange(1, n + 1))
        ax.set_title(
            f"R = {entry['geometry']['R']:g} Å — {entry['geometry']['regime']}",
            color=INK, fontsize=10.5, loc="left", pad=8,
        )
        ax.set_xlabel("orbital j", color=INK_SECONDARY, fontsize=9.5)
        ax.set_ylabel("orbital i", color=INK_SECONDARY, fontsize=9.5)

    bar = fig.colorbar(mesh, ax=axes.tolist(), pad=0.02, fraction=0.030)
    bar.set_label("mutual information  I$_{ij}$", color=INK_SECONDARY, fontsize=9.5)
    bar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    bar.outline.set_visible(False)

    fig.suptitle(
        "H$_{10}$/STO-3G two-orbital mutual information,  I$_{ij}$ = (s$_i$ + s$_j$ − s$_{ij}$)/2",
        color=INK, fontsize=12.5, x=0.02, ha="left", y=1.04,
    )
    fig.text(0.02, 0.98,
             "The bright anti-diagonal is the pairing of symmetry-related "
             "bonding and antibonding canonical orbitals (i + j = 11 here).\n"
             "Shared colour scale across panels. Diagonal set to zero by "
             "definition.",
             color=INK_SECONDARY, fontsize=8.5, ha="left", va="top", linespacing=1.5,
             transform=fig.transFigure)
    return _save(fig, stem)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path,
        default=Path("data/raw/entanglement/h10_sto-3g_orbital_entanglement.json"),
    )
    parser.add_argument("--figdir", type=Path, default=Path("figures"))
    args = parser.parse_args(argv)

    if not args.records.exists():
        parser.error(
            f"{args.records} not found. Run "
            "scripts/generate_orbital_entanglement.py first; this script never "
            "launches calculations of its own."
        )
    args.figdir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(args.records.read_text())
    results = payload["results"]
    written = make_entropy_figure(results, args.figdir / "h10_orbital_entropy")
    written += make_information_figure(results, args.figdir / "h10_mutual_information")

    print(f"{len(results)} geometries")
    for key, entry in sorted(results.items(), key=lambda kv: float(kv[0])):
        matrix = np.asarray(entry["mutual_information"])
        localised = entry.get("localised_single_orbital_entropy")
        localised_text = (
            "" if not localised
            else f"  mean s_i {np.mean(entry['single_orbital_entropy']):.4f} (MO) "
                 f"vs {np.mean(localised):.4f} (Löwdin)"
        )
        print(f"  R = {float(key):g} Å  sum s_i = "
              f"{sum(entry['single_orbital_entropy']):.4f}  "
              f"max I_ij = {matrix.max():.4f}{localised_text}")
    for path in written:
        print(f"  {path}  ({path.stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
