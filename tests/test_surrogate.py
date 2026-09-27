"""Invariants of the Experiment 5 surrogate study.

The ML gate in the plan asks for four properties that are invisible in a
learning curve and silent when broken: matched label budgets, preprocessing
statistics fitted on the training fold only, results over several seeds, and
model derivatives checked against finite differences. Each is pinned here,
because each fails by producing a *better-looking* number rather than an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from tn_quantum_chemistry.surrogate import (
    ConstantCorrection,
    Split,
    Standardiser,
    SurfaceData,
    TensorProductSpline,
    blocked_spacing_splits,
    evaluate_models,
    fit_mlp,
    gradient_check,
    load_surface,
    metrics,
    random_splits,
    subsample,
)

STORE = "data/raw/h10_surface/h10_sto-3g_surface.jsonl"


def _synthetic_surface(n_r=11, n_d=7):
    """A smooth analytic stand-in with the shape of the real dataset."""
    spacings = np.linspace(1.0, 3.0, n_r)
    dimerisations = np.linspace(-0.6, 0.6, n_d)
    grid = np.array([[r, d] for r in spacings for d in dimerisations])
    baseline = -5.0 + 0.3 * np.exp(-grid[:, 0]) + 0.05 * grid[:, 1] ** 2
    target = baseline - 0.4 - 0.2 * np.tanh(grid[:, 0] - 2.0) + 0.03 * grid[:, 1]
    return SurfaceData(
        x=grid,
        y_target=target,
        y_baseline=baseline,
        target_method="FCI",
        baseline_method="RHF",
        basis="sto-3g",
        target_walltime_seconds=np.full(grid.shape[0], 3.0),
        baseline_walltime_seconds=np.full(grid.shape[0], 0.06),
    )


# --------------------------------------------------------------------------- #
# Splits: disjointness and where they are drawn
# --------------------------------------------------------------------------- #


def test_random_split_train_and_test_are_disjoint_and_complete():
    splits = random_splits(77, test_fraction=0.2, n_repeats=3, seed=7)
    for split in splits:
        assert np.intersect1d(split.train, split.test).size == 0
        assert np.union1d(split.train, split.test).size == 77


def test_overlapping_split_is_rejected_at_construction():
    with pytest.raises(ValueError, match="overlap"):
        Split(kind="bad", label="bad", train=np.array([1, 2]), test=np.array([2, 3]))


def test_blocked_split_holds_out_exactly_the_named_spacings():
    data = _synthetic_surface()
    (split,) = blocked_spacing_splits(data.x, held_out_blocks=[[1.0, 1.2]])
    held = np.unique(data.x[split.test, 0])
    kept = np.unique(data.x[split.train, 0])
    assert np.allclose(held, [1.0, 1.2])
    assert not np.intersect1d(held, kept).size


def test_subsample_draws_only_from_the_training_pool():
    data = _synthetic_surface()
    split = random_splits(data.n_points, test_fraction=0.2, n_repeats=1, seed=3)[0]
    drawn = subsample(split, 12, seed=0)
    assert drawn.size == 12
    assert np.isin(drawn, split.train).all()
    assert not np.isin(drawn, split.test).any()


def test_subsample_is_deterministic_given_the_seed():
    split = random_splits(77, test_fraction=0.2, n_repeats=1, seed=3)[0]
    assert np.array_equal(subsample(split, 16, seed=5), subsample(split, 16, seed=5))


# --------------------------------------------------------------------------- #
# Matched budgets and no leakage
# --------------------------------------------------------------------------- #


def test_every_model_reports_the_same_label_budget():
    """All four models must be fitted on one shared draw of training indices."""
    data = _synthetic_surface()
    split = random_splits(data.n_points, test_fraction=0.2, n_repeats=1, seed=3)[0]
    rows = evaluate_models(
        data, split, 16, draw_seed=0, model_seeds=(0, 1), steps=200
    )
    assert {row.model for row in rows} == {
        "constant",
        "spline",
        "mlp_direct",
        "mlp_delta",
    }
    assert {row.n_train for row in rows} == {16}
    assert {row.n_test for row in rows} == {split.test.size}
    assert {row.model_seed for row in rows if row.model.startswith("mlp")} == {0, 1}


def test_predictions_do_not_depend_on_held_out_labels():
    """The strongest available leakage check.

    Corrupt the test-fold energies, refit, and demand the *predictions* are
    bit-identical. Any preprocessing statistic fitted on the full dataset --
    the classic silent leak -- moves them.
    """
    data = _synthetic_surface()
    split = random_splits(data.n_points, test_fraction=0.2, n_repeats=1, seed=3)[0]
    train_idx = subsample(split, 20, seed=0)

    corrupted_target = data.y_target.copy()
    corrupted_target[split.test] += 17.0
    corrupted_baseline = data.y_baseline.copy()
    corrupted_baseline[split.test] += 17.0

    clean = fit_mlp(data.x[train_idx], data.y_target[train_idx], seed=0, steps=200)
    dirty = fit_mlp(data.x[train_idx], corrupted_target[train_idx], seed=0, steps=200)
    assert np.array_equal(clean.predict(data.x[split.test]),
                          dirty.predict(data.x[split.test]))

    spline_clean = TensorProductSpline.fit(data.x[train_idx], data.y_target[train_idx])
    spline_dirty = TensorProductSpline.fit(
        data.x[train_idx], corrupted_target[train_idx]
    )
    assert np.array_equal(
        spline_clean.predict(data.x[split.test]),
        spline_dirty.predict(data.x[split.test]),
    )

    constant = ConstantCorrection.fit(
        data.y_target[train_idx], data.y_baseline[train_idx]
    )
    corrupted_constant = ConstantCorrection.fit(
        corrupted_target[train_idx], corrupted_baseline[train_idx]
    )
    assert constant.shift == corrupted_constant.shift


def test_standardiser_uses_only_the_values_it_was_fitted_on():
    train = np.array([[1.0, -0.6], [2.0, 0.0], [3.0, 0.6]])
    scaler = Standardiser.fit(train)
    assert np.allclose(scaler.mean, train.mean(axis=0))
    assert np.allclose(scaler.transform(train).mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(scaler.inverse(scaler.transform(train)), train)


def test_standardiser_survives_a_constant_column():
    """delta is constant on a single-row slice; dividing by its zero spread
    would give NaN rather than an error."""
    constant_column = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    scaled = Standardiser.fit(constant_column).transform(constant_column)
    assert np.isfinite(scaled).all()


# --------------------------------------------------------------------------- #
# The models themselves
# --------------------------------------------------------------------------- #


def test_constant_correction_is_the_mean_training_correlation_energy():
    data = _synthetic_surface()
    idx = np.arange(0, data.n_points, 3)
    model = ConstantCorrection.fit(data.y_target[idx], data.y_baseline[idx])
    expected = np.mean(data.y_target[idx] - data.y_baseline[idx])
    assert model.shift == pytest.approx(expected)
    assert model.predict(data.y_baseline[:5]) == pytest.approx(
        data.y_baseline[:5] + expected
    )


def test_spline_reproduces_a_bilinear_function_exactly():
    """A tensor-product basis of degree >= 1 contains the bilinear functions,
    so an exact fit is a property of the basis, not of the optimiser."""
    grid = np.array([[r, d] for r in np.linspace(1.0, 3.0, 11)
                     for d in np.linspace(-0.6, 0.6, 7)])
    values = 0.5 + 2.0 * grid[:, 0] - 3.0 * grid[:, 1] + 1.5 * grid[:, 0] * grid[:, 1]
    fitted = TensorProductSpline.fit(grid, values)
    assert np.allclose(fitted.predict(grid), values, atol=1e-8)


def test_spline_capacity_is_bounded_by_the_training_set_size():
    """Capacity must be a stated function of the label budget: a baseline whose
    free parameters outnumber its labels is not a baseline."""
    grid = np.array([[r, d] for r in np.linspace(1.0, 3.0, 11)
                     for d in np.linspace(-0.6, 0.6, 7)])
    values = np.sin(grid[:, 0]) + grid[:, 1] ** 2
    for n_train in (8, 16, 32, 62):
        idx = np.sort(np.random.default_rng(0).permutation(grid.shape[0])[:n_train])
        fitted = TensorProductSpline.fit(grid[idx], values[idx])
        n_parameters = fitted.n_basis[0] * fitted.n_basis[1]
        assert n_parameters <= max(4, n_train // 2)
        assert fitted.degree <= 3


def test_mlp_is_reproducible_from_its_seed():
    data = _synthetic_surface()
    first = fit_mlp(data.x, data.y_target, seed=11, steps=200)
    second = fit_mlp(data.x, data.y_target, seed=11, steps=200)
    assert np.array_equal(first.predict(data.x), second.predict(data.x))
    assert first.final_loss == second.final_loss


def test_different_seeds_give_different_models():
    """If they did not, 'repeat over seeds' would report one model three times."""
    data = _synthetic_surface()
    first = fit_mlp(data.x, data.y_target, seed=0, steps=200)
    second = fit_mlp(data.x, data.y_target, seed=1, steps=200)
    assert not np.array_equal(first.predict(data.x), second.predict(data.x))


# --------------------------------------------------------------------------- #
# Derivatives (the plan's ML gate)
# --------------------------------------------------------------------------- #


def test_autodiff_gradient_matches_central_differences():
    data = _synthetic_surface()
    model = fit_mlp(data.x, data.y_target, seed=0, steps=400)
    check = gradient_check(model, data.x, step=1e-4)
    assert check["max_rel_deviation"] < 1e-4


def test_gradient_has_one_component_per_generalized_coordinate():
    """``-grad(E)`` here is a two-component generalized force in (R, delta),
    not a 30-component Cartesian atomic force."""
    data = _synthetic_surface()
    model = fit_mlp(data.x, data.y_target, seed=0, steps=200)
    assert model.gradient(data.x).shape == (data.n_points, 2)


# --------------------------------------------------------------------------- #
# Metrics and the real store
# --------------------------------------------------------------------------- #


def test_metrics_are_consistent_across_units():
    predicted = np.array([0.0, 0.1, -0.2])
    reference = np.array([0.0, 0.0, 0.0])
    computed = metrics(predicted, reference)
    assert computed["mae_hartree"] == pytest.approx(0.1)
    assert computed["max_abs_hartree"] == pytest.approx(0.2)
    assert computed["rmse_hartree"] >= computed["mae_hartree"]
    assert computed["mae_kcal_per_mol"] == pytest.approx(0.1 * 627.5094740631)


def test_surface_loads_with_a_converged_label_at_every_geometry():
    data = load_surface(STORE)
    assert data.n_points == 77
    assert data.basis == "sto-3g"
    assert np.isfinite(data.y_target).all()
    assert np.isfinite(data.y_baseline).all()
    # HF sits above FCI everywhere: variational, same finite Hamiltonian.
    assert (data.delta_target < 0).all()
