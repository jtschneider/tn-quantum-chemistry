"""Invariants of the N2 chemistry vignette.

The vignette exists to demonstrate judgement about references, active spaces
and basis incompleteness, so the checks here are the ones that would catch a
*plausible-looking* wrong answer: a UHF that is secretly RHF, an active space
that quietly changed character, a variational bound violated, or a wall time
that measured only part of the work.
"""

from __future__ import annotations

import numpy as np
import pytest

from tn_quantum_chemistry.schema import ReferenceType, Status
from tn_quantum_chemistry.vignette import (
    DEFAULT_BASES,
    DEFAULT_BOND_LENGTHS,
    VALENCE_CAS_ELECTRONS,
    VALENCE_CAS_ORBITALS,
    VignetteSpec,
    build_molecule,
    run_vignette_point,
)


def _by_method(records):
    return {record.method: record for record in records}


@pytest.fixture(scope="module")
def equilibrium():
    return _by_method(
        run_vignette_point(VignetteSpec(bond_length=1.098, basis="sto-3g"))
    )


@pytest.fixture(scope="module")
def stretched():
    return _by_method(
        run_vignette_point(VignetteSpec(bond_length=2.400, basis="sto-3g"))
    )


# --------------------------------------------------------------------------- #
# Fast: specification only
# --------------------------------------------------------------------------- #


def test_active_space_is_the_full_valence_space_of_n2():
    """CAS(10e,8o): 14 electrons minus the two 1s pairs, in the 2s/2p orbitals."""
    assert VALENCE_CAS_ELECTRONS == 10
    assert VALENCE_CAS_ORBITALS == 8
    molecule = build_molecule(VignetteSpec(bond_length=1.098, basis="sto-3g"))
    assert molecule.nelectron == 14
    assert molecule.nelectron - VALENCE_CAS_ELECTRONS == 4  # two frozen 1s pairs


def test_minimal_basis_active_space_exhausts_the_non_core_orbitals():
    """In STO-3G the CAS *is* frozen-core FCI, which is why NEVPT2 has almost
    nothing left to recover in that basis. cc-pVDZ is what makes the external
    correlation visible."""
    minimal = build_molecule(VignetteSpec(bond_length=1.098, basis="sto-3g"))
    assert minimal.nao_nr() - 2 == VALENCE_CAS_ORBITALS
    larger = build_molecule(VignetteSpec(bond_length=1.098, basis="cc-pvdz"))
    assert larger.nao_nr() - 2 > VALENCE_CAS_ORBITALS


def test_the_vignette_spans_both_bases_and_three_geometries():
    assert len(DEFAULT_BOND_LENGTHS) == 3
    assert DEFAULT_BOND_LENGTHS == tuple(sorted(DEFAULT_BOND_LENGTHS))
    assert "sto-3g" in DEFAULT_BASES and "cc-pvdz" in DEFAULT_BASES


# --------------------------------------------------------------------------- #
# Slow: the calculations themselves
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_casci_lies_above_all_electron_fci(equilibrium):
    """CASCI freezes the core, so it is variationally above all-electron FCI in
    the same basis, and the gap is the core correlation energy."""
    gap = equilibrium["FCI"].e_total - equilibrium["CASCI(10e,8o)"].e_total
    assert gap < 0  # FCI is lower
    assert abs(gap) < 1e-2  # and only by core correlation, not by a bug


@pytest.mark.slow
def test_orbital_optimisation_lowers_the_energy(equilibrium):
    """CASSCF optimises the orbitals CASCI merely inherits from RHF."""
    assert equilibrium["CASSCF(10e,8o)"].e_total <= equilibrium["CASCI(10e,8o)"].e_total


@pytest.mark.slow
def test_nevpt2_lowers_the_casscf_energy(equilibrium):
    correction = equilibrium["NEVPT2/CASSCF(10e,8o)"].extra["nevpt2_correction_hartree"]
    assert correction < 0


@pytest.mark.slow
def test_rhf_is_stable_at_equilibrium_and_unstable_when_stretched(
    equilibrium, stretched
):
    """The stability analysis is the point: it says *when* the restricted
    reference stops being a minimum, rather than leaving it to be assumed."""
    assert equilibrium["RHF"].extra["internally_stable"] is True
    assert stretched["RHF"].extra["internally_stable"] is False


@pytest.mark.slow
def test_uhf_matches_rhf_at_equilibrium_and_breaks_symmetry_when_stretched(
    equilibrium, stretched
):
    """A UHF that never breaks symmetry is RHF under another name. The spin
    contamination is reported because the broken-symmetry solution is a
    different state, not a more accurate singlet."""
    assert equilibrium["UHF"].e_total == pytest.approx(
        equilibrium["RHF"].e_total, abs=1e-6
    )
    assert equilibrium["UHF"].extra["spin_squared"] == pytest.approx(0.0, abs=1e-6)

    assert stretched["UHF"].e_total < stretched["RHF"].e_total - 0.1
    assert stretched["UHF"].extra["spin_squared"] > 1.0
    assert stretched["UHF"].extra["instabilities_followed"] >= 1


@pytest.mark.slow
def test_coupled_cluster_breaks_down_at_the_stretched_geometry(stretched):
    """Not a defect of this code: the amplitude equations have no useful
    solution once the reference is qualitatively wrong. It must be recorded as
    unconverged rather than plotted as a result."""
    assert stretched["CCSD"].status == Status.NOT_CONVERGED
    assert stretched["CCSD(T)"].status == Status.NOT_CONVERGED
    # And the energy is far above FCI, in the same direction as on H10.
    assert stretched["CCSD(T)"].e_total > stretched["FCI"].e_total


@pytest.mark.slow
def test_active_space_occupations_track_the_bond_breaking(equilibrium, stretched):
    """A triple bond at equilibrium is nearly (2,2,2,0,0,0,...); stretched, it
    moves toward six singly occupied orbitals. If instead an orbital left the
    space, the occupations would not follow this pattern and the two geometries
    would not be the same state."""
    for point in (equilibrium, stretched):
        for method in ("CASCI(10e,8o)", "CASSCF(10e,8o)"):
            occupations = np.array(point[method].extra["active_natural_occupations"])
            assert occupations.size == VALENCE_CAS_ORBITALS
            assert occupations.sum() == pytest.approx(VALENCE_CAS_ELECTRONS, abs=1e-6)
            assert (occupations >= -1e-8).all() and (occupations <= 2.0 + 1e-8).all()

    assert equilibrium["CASSCF(10e,8o)"].extra["n_partially_occupied"] <= 6
    assert stretched["CASSCF(10e,8o)"].extra["n_partially_occupied"] == 6
    # Stretched: further from integer occupancy than at equilibrium.
    assert (
        stretched["CASSCF(10e,8o)"].extra["max_deviation_from_integer"]
        > equilibrium["CASSCF(10e,8o)"].extra["max_deviation_from_integer"]
    )


@pytest.mark.slow
def test_every_tier_records_a_wall_time_that_includes_its_reference(equilibrium):
    """The convention from ``methods.py``: a tier's cost includes the SCF it is
    built on. A correlated tier timed at less than its own reference would be
    measuring only part of the work."""
    scf = equilibrium["RHF"].walltime_seconds
    for method in ("MP2", "CCSD", "CCSD(T)", "CASCI(10e,8o)", "CASSCF(10e,8o)"):
        assert equilibrium[method].walltime_seconds >= scf


@pytest.mark.slow
def test_reference_types_are_recorded_per_method(equilibrium):
    assert equilibrium["RHF"].reference_type == ReferenceType.RHF
    assert equilibrium["UHF"].reference_type == ReferenceType.UHF
    assert equilibrium["CASSCF(10e,8o)"].reference_type == ReferenceType.CASSCF
    assert equilibrium["CCSD"].n_frozen_core == 2
