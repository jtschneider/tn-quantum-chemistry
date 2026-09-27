"""Statistics behind the entanglement-as-trust-signal analysis.

The conclusion of that analysis is a negative result, which makes the statistics
carrying it worth testing rather than trusting: a rank correlation that silently
mishandled ties, or a partial correlation that failed to remove its control,
would manufacture exactly the finding being reported.
"""

from __future__ import annotations

import numpy as np
import pytest

from tn_quantum_chemistry.trust import (
    entanglement_descriptors,
    nearest_training_distance,
    partial_rank_correlation,
    pooled_rank_correlation,
    spearman,
    top_quantile_auc,
)


def test_spearman_is_one_for_any_monotone_map():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman(x, np.exp(x)) == pytest.approx(1.0)
    assert spearman(x, -np.exp(x)) == pytest.approx(-1.0)


def test_spearman_averages_ties_rather_than_ordering_them():
    """Tied signal values must not be given an arbitrary order.

    A constant signal correlates with nothing; if ties were broken by input
    order it would correlate perfectly with any error sorted the same way.
    """
    constant = np.ones(6)
    assert np.isnan(spearman(constant, np.arange(6.0)))


def test_spearman_undefined_below_three_points():
    assert np.isnan(spearman([1.0, 2.0], [1.0, 2.0]))


def test_partial_correlation_removes_a_confound():
    """A signal that mostly relabels the control must lose most of its apparent
    correlation once the control is partialled out."""
    rng = np.random.default_rng(0)
    control = rng.normal(size=60)
    error = control * 2.0 + rng.normal(scale=0.05, size=60)
    signal = control * 3.0 + rng.normal(scale=0.05, size=60)
    assert abs(spearman(signal, error)) > 0.9
    assert abs(partial_rank_correlation(signal, error, control)) < 0.4


def test_partial_correlation_is_undefined_for_a_perfect_confound():
    """Rank-identical signal and control leave no residual to correlate.

    Returning nan is the honest answer: the data cannot distinguish the two, so
    there is no partial correlation to report. Returning 0.0 instead would look
    like evidence of no effect.
    """
    control = np.arange(30.0)
    signal = control * 3.0 + 1.0
    error = control * 2.0
    assert np.isnan(partial_rank_correlation(signal, error, control))


def test_partial_correlation_keeps_an_independent_signal():
    rng = np.random.default_rng(1)
    control = rng.normal(size=60)
    independent = rng.normal(size=60)
    error = independent                     # error depends only on the signal
    assert abs(partial_rank_correlation(independent, error, control)) > 0.8


def test_top_quantile_auc_is_one_for_perfect_ranking_and_zero_for_reversed():
    error = np.arange(12.0)
    assert top_quantile_auc(error, error) == pytest.approx(1.0)
    assert top_quantile_auc(-error, error) == pytest.approx(0.0)


def test_top_quantile_auc_is_chance_for_a_constant_signal():
    assert top_quantile_auc(np.ones(12), np.arange(12.0)) == pytest.approx(0.5)


def test_nearest_training_distance_is_zero_at_a_training_point():
    train = np.array([[1.0, 0.0], [2.0, 0.5]])
    assert nearest_training_distance(train, train)[0] == pytest.approx(0.0)


def test_nearest_training_distance_scales_the_coordinates():
    """R and delta span different ranges, so an unscaled distance is all R.

    Two test points sit half a coordinate range from the training grid, one
    along R and one along delta. Scaled, they must be equally far; unscaled, the
    R move looks 20x larger purely because R is measured in bigger numbers.
    """
    train = np.array([[1.0, 0.0], [3.0, 0.0], [1.0, 0.1], [3.0, 0.1]])
    half_range_in_R = np.array([[2.0, 0.0]])
    half_range_in_delta = np.array([[1.0, 0.05]])

    scaled_R = nearest_training_distance(half_range_in_R, train)[0]
    scaled_delta = nearest_training_distance(half_range_in_delta, train)[0]
    assert scaled_R == pytest.approx(scaled_delta)

    unit = np.ones(2)
    raw_R = nearest_training_distance(half_range_in_R, train, scale=unit)[0]
    raw_delta = nearest_training_distance(half_range_in_delta, train, scale=unit)[0]
    assert raw_R > 10 * raw_delta


def test_pooled_rank_correlation_is_immune_to_per_split_offsets():
    """Splits whose errors differ by orders of magnitude must not create signal.

    Two splits, each with *no* internal signal, but with wildly different error
    scales. Pooling raw values would report a strong correlation; ranking within
    each split first must not.
    """
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    per_split = [
        (signal, np.array([5.0, 4.0, 3.0, 2.0, 1.0]) * 1e-6),
        (signal, np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * 1e-1),
    ]
    assert abs(pooled_rank_correlation(per_split)) < 1e-9


def test_entanglement_descriptors_satisfy_the_entropy_sum_rule():
    """S = S_number + S_configurational at the cut the descriptors are read at."""
    profile = {
        "entanglement": [0.1, 0.5, 0.9, 0.5, 0.1],
        "number_entropy": [0.05, 0.2, 0.4, 0.2, 0.05],
        "configurational_entropies": {"1.0": [0.05, 0.3, 0.5, 0.3, 0.05]},
        "var_n_left": [0.01, 0.2, 0.4, 0.2, 0.01],
        "n_sectors": [2, 5, 9, 5, 2],
    }
    d = entanglement_descriptors(profile)
    assert d["s_central"] == pytest.approx(
        d["number_entropy_central"] + d["configurational_central"]
    )
    assert d["s_max"] == pytest.approx(0.9)


def test_entanglement_descriptors_read_the_von_neumann_renyi_member():
    """configurational_entropies is keyed by Renyi index, not a flat list.

    Reading it as a list silently indexed the *key order* and produced a
    TypeError here; picking the wrong key would have been worse, since 0.5 and
    2.0 are real numbers that break the sum rule quietly.
    """
    profile = {
        "entanglement": [0.0, 1.0, 0.0],
        "number_entropy": [0.0, 0.25, 0.0],
        "configurational_entropies": {"0.5": [9.0, 9.0, 9.0], "1.0": [0.0, 0.75, 0.0]},
        "var_n_left": [0.0, 0.1, 0.0],
        "n_sectors": [1, 3, 1],
    }
    assert entanglement_descriptors(profile)["configurational_central"] == pytest.approx(0.75)
