"""Experiment 5 figures and table, built only from cached results.

Reads the store written by ``train_surrogate.py`` and renders:

* ``figures/h10_surrogate_learning_curves.{svg,pdf,png}`` -- test error against
  the number of expensive FCI labels, one panel for the random interpolation
  split and one for the blocked-R holdout;
* ``figures/h10_surrogate.csv`` -- the same numbers as a table, which is also
  the relief the low-contrast series need to stay legible.

This script fits nothing and runs no electronic-structure calculation.

Run with::

    uv run python scripts/make_surrogate_figures.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RESULTS = Path("data/processed/surrogate/h10_surrogate_results.jsonl")
DEFAULT_FIGURES = Path("figures")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

#: Two validated categorical hues for the two models under comparison, and
#: neutral dash-encoded reference lines for the two baselines -- the same
#: convention as the dissociation figure, where FCI is the black reference
#: rather than a fifth series colour. The pair passes the lightness, chroma,
#: CVD-separation, normal-vision and contrast checks on this surface.
STYLE = {
    "constant": {"color": "#8f8e8a", "linestyle": ":", "label": "constant correction"},
    "spline": {"color": INK_SECONDARY, "linestyle": "--", "label": "tensor-product spline"},
    "mlp_direct": {"color": "#2a78d6", "linestyle": "-", "label": "MLP, direct on $E_\\mathrm{FCI}$"},
    "mlp_delta": {"color": "#eb6834", "linestyle": "-", "label": "MLP, $\\Delta$ on $E_\\mathrm{FCI}-E_\\mathrm{HF}$"},
}
ORDER = ("constant", "spline", "mlp_direct", "mlp_delta")

HARTREE_TO_KCAL = 627.5094740631


def load_results(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def aggregate(rows: list[dict], metric: str = "mae_kcal_per_mol"):
    """Median and spread over splits and seeds, per (panel, model, size).

    The median rather than the best run: a learning curve drawn through the
    luckiest seed is a statement about luck. The band is the full range, so a
    single bad seed stays visible instead of being smoothed away.

    The panel key is the split *label* for blocked holdouts and the split
    *kind* for random ones. Pooling the three blocked holdouts was the first
    thing tried and it is wrong: the compressed and stretched blocks move in
    opposite directions, so their median describes neither and their band spans
    two orders of magnitude for a reason that is structure, not noise.
    """
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        panel = row["split_kind"] if row["split_kind"] == "random" else row["split_label"]
        grouped[(panel, row["model"], row["n_train"])].append(row["metrics"][metric])
    return grouped


def _panel_order(grouped) -> list[str]:
    """Random first, then the blocked holdouts left to right in R."""
    blocked = sorted(
        {key[0] for key in grouped if key[0].startswith("blocked")},
        key=lambda label: float(label.split("=")[1].split(",")[0]),
    )
    return (["random"] if any(k[0] == "random" for k in grouped) else []) + blocked


PANEL_TITLES = {
    "random": "random interpolation split",
    "blocked-R=1,1.2": "held out $R$ = 1.0-1.2 $\\mathrm{\\AA}$ (compressed)",
    "blocked-R=1.8,2": "held out $R$ = 1.8-2.0 $\\mathrm{\\AA}$ (interior gap)",
    "blocked-R=2.8,3": "held out $R$ = 2.8-3.0 $\\mathrm{\\AA}$ (stretched)",
}


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=3, width=0.8)
    ax.grid(True, which="major", color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def learning_curve_figure(rows: list[dict], out: Path) -> None:
    grouped = aggregate(rows)
    panels = _panel_order(grouped)
    n_cols = 2
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(9.6, 4.1 * n_rows), sharey=True, sharex=True,
        facecolor=SURFACE,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, panel in zip(axes, panels, strict=False):
        _style_axis(ax)
        for model in ORDER:
            sizes = sorted(n for (p, m, n) in grouped if p == panel and m == model)
            if not sizes:
                continue
            median = np.array([np.median(grouped[(panel, model, n)]) for n in sizes])
            low = np.array([np.min(grouped[(panel, model, n)]) for n in sizes])
            high = np.array([np.max(grouped[(panel, model, n)]) for n in sizes])
            style = STYLE[model]
            ax.fill_between(sizes, low, high, color=style["color"], alpha=0.13,
                            linewidth=0)
            ax.plot(sizes, median, color=style["color"],
                    linestyle=style["linestyle"], linewidth=2.0, marker="o",
                    markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.8,
                    label=style["label"], zorder=3)
        # The 1 kcal/mol line is a chemistry landmark, not a claim: these are
        # minimal-basis energies, so crossing it means the surrogate tracks its
        # own FCI labels that closely, not that the numbers are that accurate.
        ax.axhline(1.0, color=INK, linewidth=0.9, alpha=0.45)
        ax.text(62, 1.08, "1 kcal/mol", fontsize=8, color=INK_SECONDARY,
                va="bottom", ha="right")
        ax.set_yscale("log")
        ax.set_title(PANEL_TITLES.get(panel, panel), fontsize=10.5, color=INK,
                     pad=8)

    for ax in axes[len(panels):]:
        ax.set_visible(False)
    for ax in axes[max(0, len(panels) - n_cols):len(panels)]:
        ax.set_xlabel("expensive FCI labels in the training fold", fontsize=9.5,
                      color=INK_SECONDARY)
    for index in range(0, len(panels), n_cols):
        axes[index].set_ylabel("test MAE (kcal/mol)", fontsize=9.5,
                               color=INK_SECONDARY)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY,
                   loc="lower left", handlelength=2.4)

    fig.suptitle(
        "H$_{10}$/STO-3G $(R,\\delta)$ surface: matched surrogate learning curves",
        fontsize=11.5, color=INK, x=0.02, ha="left", y=0.995,
    )
    fig.text(
        0.02, -0.01,
        "Median over 5 random splits (first panel) or 3 model seeds (blocked "
        "panels); band is the full range across splits and seeds.\nAll four "
        "models share one draw of training indices at every label count. "
        "$\\Delta$-learning wins on interpolation and into the compressed "
        "wall,\nand loses on the stretched block where the correlation energy "
        "is the less smooth of the two targets. Interpolation on one "
        "H$_{10}$ surface, not molecular transferability.",
        fontsize=8, color=INK_SECONDARY, ha="left", va="top",
    )
    fig.tight_layout()
    for suffix in ("svg", "pdf", "png"):
        path = out.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def write_table(rows: list[dict], path: Path) -> None:
    grouped_ha = aggregate(rows, "mae_hartree")
    grouped_kcal = aggregate(rows, "mae_kcal_per_mol")
    grouped_max = aggregate(rows, "max_abs_kcal_per_mol")
    grouped_rmse = aggregate(rows, "rmse_hartree")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "split_kind", "model", "n_train", "n_runs",
            "mae_hartree_median", "rmse_hartree_median",
            "mae_kcal_per_mol_median", "mae_kcal_per_mol_min",
            "mae_kcal_per_mol_max", "max_abs_kcal_per_mol_median",
        ])
        for key in sorted(grouped_ha, key=lambda k: (k[0], ORDER.index(k[1]), k[2])):  # noqa: E501
            writer.writerow([
                key[0], key[1], key[2], len(grouped_ha[key]),
                f"{np.median(grouped_ha[key]):.6e}",
                f"{np.median(grouped_rmse[key]):.6e}",
                f"{np.median(grouped_kcal[key]):.4f}",
                f"{np.min(grouped_kcal[key]):.4f}",
                f"{np.max(grouped_kcal[key]):.4f}",
                f"{np.median(grouped_max[key]):.4f}",
            ])
    print(f"wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()

    rows = load_results(args.results)
    print(f"{len(rows)} result rows")
    args.figures.mkdir(parents=True, exist_ok=True)
    learning_curve_figure(rows, args.figures / "h10_surrogate_learning_curves")
    write_table(rows, args.figures / "h10_surrogate.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
