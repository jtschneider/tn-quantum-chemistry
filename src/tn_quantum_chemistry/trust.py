"""Does an entanglement diagnostic predict where the energy surrogate fails?

The surrogate of Experiment 5 is trained on geometry alone. It has one bad
regime -- the stretched blocked holdout, where delta-learning loses to direct
learning by 13x -- and nothing in the model announces that. The question here is
whether a tensor-network diagnostic of the *state* flags those geometries, and
whether it does so better than the signals a practitioner already has for free.

**This measures predictive content, not a workflow.** On H10/STO-3G a converged
DMRG costs 5.4 s against FCI's 2.4 s, so obtaining the diagnostic at a new
geometry costs more than simply computing the exact answer. No acquisition claim
can be demonstrated on a system this small; what can be established is the
precondition for one -- whether the signal carries information about surrogate
error at all. If it does not, no cheaper way of obtaining it would help.

Three free baselines have to be beaten before an entanglement signal means
anything:

* **distance to the nearest training point**, the standard model-agnostic
  proxy for extrapolation;
* **ensemble disagreement** across model seeds, the standard ML uncertainty
  estimate, which costs nothing beyond seeds already run;
* **the spacing R itself**, because an entanglement measure that merely tracks
  the coordinate the model already consumes adds nothing to it.

Correlations are computed *within* a split before pooling. Error magnitudes
differ by orders of magnitude between the random and blocked splits, so a naive
pooled correlation would mostly measure which split a point came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Entanglement descriptors taken from a stored ``sector_profile``. Each is a
#: scalar summary of the converged MPS at one geometry.
DESCRIPTORS = (
    "s_max",
    "s_central",
    "number_entropy_central",
    "configurational_central",
    "n_sectors_central",
    "var_n_left_central",
)

#: Signals available without running any tensor network, used as the bar the
#: entanglement descriptors have to clear.
BASELINE_SIGNALS = ("dist_to_train", "ensemble_std", "spacing_R")


def entanglement_descriptors(profile: dict[str, Any]) -> dict[str, float]:
    """Scalar summaries of one stored ``sector_profile``.

    The central cut is used for the per-cut quantities because it is the one
    bond every orbital ordering has in common; ``s_max`` is reported beside it
    because the maximum need not sit at the centre for a dimerised chain.
    """
    entanglement = np.asarray(profile["entanglement"], dtype=float)
    middle = len(entanglement) // 2
    out = {
        "s_max": float(entanglement.max()),
        "s_central": float(entanglement[middle]),
    }
    for name, key in (
        ("number_entropy_central", "number_entropy"),
        ("var_n_left_central", "var_n_left"),
        ("n_sectors_central", "n_sectors"),
    ):
        values = profile.get(key)
        out[name] = float(np.asarray(values, dtype=float)[middle]) if values else np.nan

    # configurational_entropies is keyed by Renyi index, not a flat per-cut
    # list; "1.0" is the von Neumann member, the one that completes the sum
    # rule S = S_number + S_configurational.
    configurational = (profile.get("configurational_entropies") or {}).get("1.0")
    out["configurational_central"] = (
        float(np.asarray(configurational, dtype=float)[middle])
        if configurational is not None
        else np.nan
    )
    return out


def nearest_training_distance(
    x_test: np.ndarray, x_train: np.ndarray, *, scale: np.ndarray | None = None
) -> np.ndarray:
    """Distance from each test point to the closest training point.

    Coordinates are scaled before the distance is taken, because R and delta
    span different ranges and an unscaled Euclidean distance would be almost
    entirely the R difference.
    """
    x_test = np.atleast_2d(np.asarray(x_test, dtype=float))
    x_train = np.atleast_2d(np.asarray(x_train, dtype=float))
    if scale is None:
        scale = np.ptp(x_train, axis=0)
    scale = np.where(np.asarray(scale, dtype=float) > 0, scale, 1.0)
    diff = (x_test[:, None, :] - x_train[None, :, :]) / scale
    return np.sqrt((diff**2).sum(axis=-1)).min(axis=1)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, with ties averaged. NaN when it is undefined."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = _rank(a[ok]), _rank(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, so tied signals cannot fake a correlation."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        sums = np.zeros(unique.size)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def top_quantile_auc(signal: np.ndarray, error: np.ndarray, *, quantile: float = 0.75) -> float:
    """AUC for using ``signal`` to identify the worst-error points.

    0.5 is chance. Reported beside the rank correlation because ranking every
    point correctly and flagging the few bad ones are different jobs, and only
    the second is what a trust signal is for.
    """
    signal = np.asarray(signal, dtype=float)
    error = np.asarray(error, dtype=float)
    ok = np.isfinite(signal) & np.isfinite(error)
    signal, error = signal[ok], error[ok]
    if signal.size < 4:
        return float("nan")
    threshold = np.quantile(error, quantile)
    positive = error >= threshold
    if positive.all() or not positive.any():
        return float("nan")
    ranks = _rank(signal)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


@dataclass(frozen=True)
class TrustEvaluation:
    """Per-split agreement between one signal and the surrogate's error."""

    signal: str
    model: str
    split_label: str
    n_test: int
    spearman: float
    auc: float
    extra: dict[str, Any] = field(default_factory=dict)


def evaluate_signal(
    signal: np.ndarray,
    error: np.ndarray,
    *,
    signal_name: str,
    model: str,
    split_label: str,
) -> TrustEvaluation:
    return TrustEvaluation(
        signal=signal_name,
        model=model,
        split_label=split_label,
        n_test=int(np.size(error)),
        spearman=spearman(signal, error),
        auc=top_quantile_auc(signal, error),
    )


def partial_rank_correlation(
    signal: np.ndarray, error: np.ndarray, control: np.ndarray
) -> float:
    """Rank correlation of ``signal`` with ``error`` after removing ``control``.

    The decisive statistic for an entanglement trust signal. The surrogate
    already consumes the geometry, so a descriptor that correlates with the
    error only because it tracks the spacing R adds nothing a practitioner
    could not read off the input. This residualises both ranks on the control's
    rank and correlates what is left.
    """
    signal, error, control = (np.asarray(v, dtype=float) for v in (signal, error, control))
    ok = np.isfinite(signal) & np.isfinite(error) & np.isfinite(control)
    if ok.sum() < 4:
        return float("nan")
    rs, re, rc = _rank(signal[ok]), _rank(error[ok]), _rank(control[ok])
    if rc.std() == 0:
        return spearman(signal[ok], error[ok])

    def residual(values):
        slope = np.cov(values, rc, bias=True)[0, 1] / np.var(rc)
        return values - slope * (rc - rc.mean()) - values.mean()

    a, b = residual(rs), residual(re)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def pooled_rank_correlation(
    per_split: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Spearman over points whose ranks were taken *within* their own split.

    Pooling raw values across splits would mostly measure which split a point
    came from, because the blocked holdouts miss by orders of magnitude more
    than the random ones. Ranking inside each split first removes that offset
    while keeping the extra statistical power of the pooled set.
    """
    signals, errors = [], []
    for signal, error in per_split:
        signal = np.asarray(signal, dtype=float)
        error = np.asarray(error, dtype=float)
        ok = np.isfinite(signal) & np.isfinite(error)
        if ok.sum() < 3:
            continue
        n = int(ok.sum())
        signals.append(_rank(signal[ok]) / max(n - 1, 1))
        errors.append(_rank(error[ok]) / max(n - 1, 1))
    if not signals:
        return float("nan")
    return spearman(np.concatenate(signals), np.concatenate(errors))
