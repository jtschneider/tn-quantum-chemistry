"""Orbital entropies and mutual information: invariants and cross-route agreement.

``s_i`` here comes from spin-traced reduced density matrices by an analytic
route, because block2's own ``get_orbital_entropies`` raises in SU2 mode. That
makes an independent check essential rather than optional, so the same quantity
is also taken from an SZ calculation through block2's routine and the two are
required to agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from tn_quantum_chemistry.orbital_entanglement import (
    MAX_SINGLE_ORBITAL_ENTROPY,
    compute_orbital_entanglement,
    entropy_from_probabilities,
    mutual_information,
    rdm_invariants,
    single_orbital_entropy,
    single_orbital_occupations,
    su2_reduced_density_matrices,
)
from tn_quantum_chemistry.validation import (
    HamiltonianSpec,
    active_space_hamiltonian,
    build_mean_field,
    linear_hydrogen_chain,
)

STO3G = HamiltonianSpec(basis="sto-3g")


@pytest.fixture(scope="module")
def hamiltonian():
    return active_space_hamiltonian(
        build_mean_field(linear_hydrogen_chain(6, R=2.0), STO3G), STO3G
    )


@pytest.fixture(scope="module")
def rdms(hamiltonian):
    _, dm1, dm2 = su2_reduced_density_matrices(hamiltonian)
    return dm1, dm2


# --------------------------------------------------------------------------- #
# Closed-form checks
# --------------------------------------------------------------------------- #


def test_entropy_of_a_pure_state_is_zero_and_uniform_is_maximal():
    assert entropy_from_probabilities(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert entropy_from_probabilities(
        np.full(4, 0.25)
    ) == pytest.approx(MAX_SINGLE_ORBITAL_ENTROPY)


def test_mutual_information_is_symmetric_with_a_zero_diagonal():
    s_single = np.array([0.4, 0.9, 0.6])
    s_two = np.array([[0.0, 1.2, 1.0], [1.2, 0.0, 1.4], [1.0, 1.4, 0.0]])
    information = mutual_information(s_single, s_two)

    assert np.allclose(information, information.T)
    assert np.allclose(np.diag(information), 0.0)
    # The halving convention is stated explicitly because both appear in print.
    assert information[0, 1] == pytest.approx(0.5 * (0.4 + 0.9 - 1.2))
    unhalved = mutual_information(s_single, s_two, halved=False)
    assert unhalved[0, 1] == pytest.approx(2 * information[0, 1])


# --------------------------------------------------------------------------- #
# Invariants on a real calculation
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_orbital_occupations_are_a_probability_distribution(rdms):
    """Positivity and normalisation of the one-orbital reduced density matrix."""
    probabilities = single_orbital_occupations(*rdms)
    assert probabilities.shape[1] == 4
    assert np.all(probabilities >= -1e-12)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    # Spin symmetry of a singlet: the two singly occupied states are equally likely.
    assert np.allclose(probabilities[:, 1], probabilities[:, 2], atol=1e-12)


@pytest.mark.slow
def test_single_orbital_entropies_respect_their_bound(rdms):
    entropies = single_orbital_entropy(*rdms)
    assert np.all(entropies >= 0.0)
    assert np.all(entropies <= MAX_SINGLE_ORBITAL_ENTROPY + 1e-12)


@pytest.mark.slow
def test_reduced_density_matrix_invariants_hold(rdms, hamiltonian):
    invariants = rdm_invariants(*rdms, hamiltonian.n_electrons)
    assert invariants["trace_1rdm_error"] < 1e-9
    assert invariants["hermiticity_error_1rdm"] < 1e-12
    assert invariants["trace_2rdm_error"] < 1e-8
    assert invariants["occupation_bound_violation"] == 0.0
    assert invariants["min_orbital_probability"] > -1e-12
    assert invariants["probability_normalisation_error"] < 1e-12
    assert invariants["entropy_bound_violation"] == 0.0


# --------------------------------------------------------------------------- #
# Cross-route agreement -- the reason the analytic route is trustworthy
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_analytic_and_block2_single_orbital_entropies_agree(hamiltonian):
    """The two routes must describe the same state.

    ``compute_orbital_entanglement`` raises if the SZ state's energy, spin
    sector or ``s_i`` disagree with the validated SU2 route, so reaching the
    assertions at all already means the checks passed; they are restated here so
    a failure names what went wrong.
    """
    result = compute_orbital_entanglement(hamiltonian, sz_bond_dim=500)
    check = result.cross_check

    assert check["energy_difference"] < 1e-8
    assert check["spin_squared"] == pytest.approx(0.0, abs=1e-8)
    assert check["single_orbital_entropy_difference"] < 1e-5

    invariants = result.invariants
    assert invariants["subadditivity_violation"] == 0.0
    assert invariants["araki_lieb_violation"] == 0.0
    assert invariants["two_orbital_bound_violation"] == 0.0
    assert invariants["mutual_information_symmetry_error"] < 1e-12
    assert invariants["min_mutual_information"] >= -1e-12


@pytest.mark.slow
def test_stretching_the_chain_raises_the_entanglement():
    """Physical sanity: a stretched chain is more strongly correlated.

    Not a tautology -- it is the check that these numbers track correlation
    rather than an artefact of the orbital basis.
    """
    totals = {}
    for spacing in (1.0, 2.6):
        hamiltonian = active_space_hamiltonian(
            build_mean_field(linear_hydrogen_chain(6, R=spacing), STO3G), STO3G
        )
        _, dm1, dm2 = su2_reduced_density_matrices(hamiltonian)
        totals[spacing] = float(np.sum(single_orbital_entropy(dm1, dm2)))

    assert totals[2.6] > totals[1.0]


# --------------------------------------------------------------------------- #
# Basis dependence
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_energy_is_invariant_under_orbital_rotation_but_entropy_is_not():
    """The sharpest statement of what ``s_i`` does and does not measure.

    Localising the orbitals is a unitary change of basis, so the FCI energy
    cannot move. The single-orbital entropy moves a great deal, because it
    depends on how the Hilbert space was partitioned into orbitals.

    This matters for reading the figures. On H10 at R = 2.6 A the canonical
    ``s_i`` sits within a few percent of its ln 4 ceiling, which looks like a
    state so correlated that no method could describe it. In localised orbitals
    the same state gives ``s_i`` just above ln 2 with one electron per atom --
    the Mott signature, and nowhere near saturated. The near-ceiling values are
    a property of the orbital basis.
    """
    from tn_quantum_chemistry.orbital_entanglement import lowdin_hamiltonian

    geometry = linear_hydrogen_chain(6, R=2.6, delta=0.0)
    mf = build_mean_field(geometry, STO3G)
    canonical = active_space_hamiltonian(mf, STO3G)
    localised = lowdin_hamiltonian(mf, STO3G)

    from tn_quantum_chemistry.validation import run_fci

    e_canonical, _ = run_fci(canonical)
    e_localised, _ = run_fci(localised)
    assert e_canonical == pytest.approx(e_localised, abs=1e-9)

    _, dm1_canonical, dm2_canonical = su2_reduced_density_matrices(canonical)
    _, dm1_localised, dm2_localised = su2_reduced_density_matrices(localised)
    s_canonical = single_orbital_entropy(dm1_canonical, dm2_canonical)
    s_localised = single_orbital_entropy(dm1_localised, dm2_localised)

    # The entropy is emphatically not invariant.
    assert s_canonical.mean() > s_localised.mean() + 0.3

    # Localised orbitals on a symmetric chain carry one electron each, and the
    # entropy sits just above ln 2 rather than near ln 4.
    assert np.allclose(np.diag(dm1_localised), 1.0, atol=1e-6)
    assert s_localised.mean() == pytest.approx(np.log(2.0), abs=0.12)
    assert s_canonical.mean() > 0.85 * MAX_SINGLE_ORBITAL_ENTROPY


@pytest.mark.slow
def test_localisation_is_rejected_when_a_core_is_frozen():
    """A frozen core is defined by the canonical orbitals and cannot survive."""
    from tn_quantum_chemistry.orbital_entanglement import lowdin_hamiltonian

    frozen = HamiltonianSpec(basis="sto-3g", n_frozen_core=1, n_active_orbitals=5)
    mf = build_mean_field(linear_hydrogen_chain(6, R=2.0), frozen)
    with pytest.raises(ValueError, match="frozen core"):
        lowdin_hamiltonian(mf, frozen)
