"""Day-2 regression checks: the validated energies must not drift.

Two kinds of assertion live here.

*External* -- the N2/STO-3G energy at 1.1 A is compared against the value
published in block2's own documentation. That is a genuine reproduction of
someone else's number, not self-consistency.

*Internal* -- the remaining values are regression locks on energies this
project has validated with two independent solvers. They are recorded to eight
decimal places so a refactor that changes any convention is caught immediately.
A failure here means something changed; it does not by itself say which value
is right.
"""

from __future__ import annotations

import pytest

from tn_quantum_chemistry.validation import (
    BLOCK2_N2_STO3G_REFERENCE,
    DMRGSchedule,
    HamiltonianSpec,
    active_space_hamiltonian,
    build_mean_field,
    diatomic,
    linear_hydrogen_chain,
    run_dmrg,
    run_fci,
    validate_solvers,
)

STO3G = HamiltonianSpec(basis="sto-3g")

#: Validated with PySCF FCI and block2 DMRG agreeing to better than 1e-10 Ha.
REGRESSION_ENERGIES = {
    "N2_1.1": {"e_hf": -107.49650051, "e_fci": -107.65412245},
    "H10_R1.0": {"e_hf": -5.21406880, "e_fci": -5.37995475},
    "H10_R2.4": {"e_hf": -3.59913828, "e_fci": -4.68711356},
}


@pytest.fixture(scope="module")
def n2_comparison():
    return validate_solvers(
        diatomic("N", 1.1, label="N2_block2_reference"),
        STO3G,
        reference_energy=BLOCK2_N2_STO3G_REFERENCE,
        reference_source="block2 documentation",
    )


# --------------------------------------------------------------------------- #
# External reference
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_n2_reproduces_the_published_block2_energy(n2_comparison):
    """block2's tutorial reports -107.654122447524415 Ha for N2/STO-3G at 1.1 A.

    Their calculation uses D2h symmetry; this one uses C1, so agreement also
    confirms the energy does not depend on whether point-group symmetry is
    exploited.
    """
    assert n2_comparison.e_fci_total == pytest.approx(BLOCK2_N2_STO3G_REFERENCE, abs=1e-10)
    assert n2_comparison.e_dmrg_total == pytest.approx(BLOCK2_N2_STO3G_REFERENCE, abs=1e-10)


# --------------------------------------------------------------------------- #
# Solver agreement -- the Experiment 1 gate
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_n2_gate_passes(n2_comparison):
    assert n2_comparison.passed
    assert n2_comparison.delta_total < 1e-8
    assert n2_comparison.delta_electronic < 1e-8


@pytest.mark.slow
def test_dmrg_respects_the_variational_bound(n2_comparison):
    """DMRG cannot lie below exact diagonalisation of the same Hamiltonian."""
    assert n2_comparison.variational_bound_respected
    assert n2_comparison.dmrg_minus_fci > -1e-8


@pytest.mark.slow
@pytest.mark.parametrize(
    ("label", "spacing"), [("H10_R1.0", 1.0), ("H10_R2.4", 2.4)]
)
def test_h10_solvers_agree(label, spacing):
    comparison = validate_solvers(
        linear_hydrogen_chain(10, R=spacing, delta=0.0, label=label), STO3G
    )
    assert comparison.passed
    assert comparison.e_fci_total == pytest.approx(
        REGRESSION_ENERGIES[label]["e_fci"], abs=1e-8
    )


# --------------------------------------------------------------------------- #
# Regression locks
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.parametrize("label", sorted(REGRESSION_ENERGIES))
def test_reference_energies_have_not_drifted(label):
    geometry = (
        diatomic("N", 1.1, label=label)
        if label.startswith("N2")
        else linear_hydrogen_chain(10, R=float(label.split("_R")[1]), delta=0.0, label=label)
    )
    mf = build_mean_field(geometry, STO3G)
    e_fci, _ = run_fci(active_space_hamiltonian(mf, STO3G))

    expected = REGRESSION_ENERGIES[label]
    assert float(mf.e_tot) == pytest.approx(expected["e_hf"], abs=1e-8)
    assert e_fci == pytest.approx(expected["e_fci"], abs=1e-8)


# --------------------------------------------------------------------------- #
# Method ordering
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_correlation_recovers_monotonically_for_a_weakly_correlated_system():
    """RHF > MP2 > CCSD > CCSD(T) > FCI on H2O/STO-3G, with DMRG matching FCI.

    Guards against a mis-wired method: an ordering violation on a
    weakly correlated system means something is wrong with the setup, not with
    the chemistry.
    """
    from tn_quantum_chemistry.methods import run_all_methods
    from tn_quantum_chemistry.validation import water

    energies = {
        r.method: r.e_total
        for r in run_all_methods(water(), STO3G, with_dmrg=False)
    }
    assert (
        energies["RHF"]
        > energies["MP2"]
        > energies["CCSD"]
        > energies["CCSD(T)"]
        > energies["FCI"]
    )

    asham = active_space_hamiltonian(build_mean_field(water(), STO3G), STO3G)
    e_dmrg, _, _ = run_dmrg(asham, DMRGSchedule.default())
    assert e_dmrg == pytest.approx(energies["FCI"], abs=1e-8)


# --------------------------------------------------------------------------- #
# Symmetry-resolved gate conditions
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_gate_records_the_symmetry_resolved_structure(n2_comparison):
    """Every validated geometry ships its sector profile and spin sector.

    The profile comes from the same solver run as the energy, so a passing gate
    now certifies the state as well as the number.
    """
    from tn_quantum_chemistry.entanglement import SectorProfile

    assert n2_comparison.sector_profile is not None
    profile = SectorProfile.from_dict(n2_comparison.sector_profile)
    assert profile.n_cuts == n2_comparison.n_orbitals - 1

    assert n2_comparison.profile_sum_rule_error < 1e-9
    assert n2_comparison.profile_normalisation_error < 1e-9


@pytest.mark.slow
def test_gate_verifies_the_spin_sector(n2_comparison):
    """<S^2> must equal S(S+1) for the requested sector.

    Under SU2 this is structurally guaranteed, so the measurement records
    evidence rather than testing the solver. It is a gate condition because
    energy agreement alone does not imply sector purity: an SZ calculation of
    the same H10 system converges to within 2.4e-5 Ha of FCI while carrying
    <S^2> = 9.0e-3, and at chi = 64 lands on a clean triplet.
    """
    assert n2_comparison.expected_spin_squared == pytest.approx(0.0)
    assert n2_comparison.spin_squared == pytest.approx(0.0, abs=1e-10)
    assert n2_comparison.spin_sector_verified
    assert n2_comparison.passed


@pytest.mark.slow
def test_gate_tracks_a_non_singlet_target():
    """The spin check is not trivially zero: a triplet target must give S(S+1)=2."""
    from tn_quantum_chemistry.validation import Geometry

    chain = linear_hydrogen_chain(6, R=1.8, delta=0.0, label="H6_triplet")
    triplet = Geometry(
        label=chain.label, atoms=chain.atoms, unit=chain.unit,
        charge=0, spin=2, parameters=chain.parameters,
    )
    comparison = validate_solvers(triplet, STO3G)

    assert comparison.expected_spin_squared == pytest.approx(2.0)
    assert comparison.spin_squared == pytest.approx(2.0, abs=1e-8)
    assert comparison.spin_sector_verified
    assert comparison.passed
