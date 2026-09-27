"""The convergence-metric logic behind the bond-dimension study.

``chi*`` -- the smallest bond dimension reaching a target accuracy -- is the
headline number of that experiment, so it is tested rather than trusted. The
functions live in ``scripts/`` because they are analysis of a stored result, not
part of the solver stack; the tests reach into the script the same way
``test_conventions.py`` reaches into it to check solver thresholds.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "make_chi_convergence_figures.py"
_spec = importlib.util.spec_from_file_location("_chi_figures", _SCRIPT)
chi_figures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chi_figures)


def test_converged_chi_returns_the_smallest_sufficient_bond_dimension():
    chis = [16, 32, 64, 128, 256]
    errors = [1e-1, 1e-3, 1e-6, 1e-10, 1e-13]
    assert chi_figures.converged_chi(chis, errors, 1e-9) == 128


def test_converged_chi_is_none_when_the_grid_never_reaches_the_target():
    """A grid that stops short must say so, not report its largest entry.

    This is the R = 1.0 case in the stored data: the error is still 1.4e-7 at
    the largest chi sampled. Reporting 256 there would assert convergence that
    was never observed.
    """
    assert chi_figures.converged_chi([16, 64, 256], [1e-1, 1e-4, 1e-7], 1e-9) is None


def test_converged_chi_ignores_the_order_of_the_input():
    unsorted = [256, 16, 64]
    errors = [1e-13, 1e-1, 1e-10]
    assert chi_figures.converged_chi(unsorted, errors, 1e-9) == 64


def test_fit_exponent_recovers_a_known_power_law():
    chis = [16, 32, 64, 128]
    errors = [1e-2 * (c / 16.0) ** -3.0 for c in chis]
    exponent, r2 = chi_figures.fit_exponent(chis, errors)
    assert exponent == pytest.approx(3.0, abs=1e-6)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_fit_exponent_excludes_points_at_the_reference_floor():
    """Points that have hit the FCI floor measure the reference, not the MPS.

    Including them would bend the fitted slope toward whatever the floor
    happens to be, turning a property of the comparison into an apparent
    property of the state.
    """
    chis = [16, 32, 64, 128, 192, 256]
    clean = [1e-2 * (c / 16.0) ** -3.0 for c in chis[:4]]
    floored = [3e-13, 1e-13]
    exponent, _ = chi_figures.fit_exponent(chis, clean + floored)
    assert exponent == pytest.approx(3.0, abs=1e-6)


def test_fit_exponent_declines_to_fit_too_few_points():
    exponent, r2 = chi_figures.fit_exponent([16, 32], [1e-2, 1e-3])
    assert np.isnan(exponent) and np.isnan(r2)


def test_converged_tolerance_sits_above_the_reference_floor():
    """chi* must measure the MPS, not the last digits of the FCI comparison."""
    assert chi_figures.CONVERGED_TOL > 100 * chi_figures.REFERENCE_FLOOR
