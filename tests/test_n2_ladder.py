"""Invariants of the N2 active-space ladder.

Three things here are worth asserting rather than assuming, because each one
already produced a wrong number during development:

* the determinant counts that decide which rungs have an exact reference;
* the memory guard on the Davidson subspace, since the project's ``max_space``
  convention was set on a system with 0.5 MB CI vectors and asks for 61 GB here;
* the orbital permutation, which must leave the spectrum untouched -- a
  reordering that changed the energy would make the whole ordering comparison a
  comparison of two different problems.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tn_quantum_chemistry.n2_ladder import (
    ACTIVE_SPACE_LADDER,
    DAVIDSON_MAX_SPACE,
    DAVIDSON_MIN_SPACE,
    N2_FROZEN_CORE,
    PHASE_1_RUNGS,
    PHASE_2_RUNGS,
    VALENCE_ELECTRONS,
    LadderPoint,
    ci_vector_bytes,
    davidson_max_space,
    determinant_count,
    dmrg_method_label,
    record_key,
    reorder_active_space,
)
from tn_quantum_chemistry.validation import ActiveSpaceHamiltonian, DMRGSchedule

#: The ladder as it appears in ``notes/n2-tensor-network-experiment.md``. If
#: these move, the note is wrong and so is the argument built on it.
EXPECTED_DETERMINANTS = {
    8: 3_136,
    12: 627_264,
    16: 19_079_424,
    20: 240_374_016,
    26: 4_327_008_400,
}


def test_determinant_counts_match_the_planned_ladder():
    for n_orb, expected in EXPECTED_DETERMINANTS.items():
        assert determinant_count(n_orb, VALENCE_ELECTRONS) == expected


def test_the_ladder_constant_covers_exactly_the_planned_rungs():
    assert set(ACTIVE_SPACE_LADDER) == set(EXPECTED_DETERMINANTS)


def test_determinant_count_handles_open_shell_sectors():
    # 2S = 2 puts six alpha and four beta electrons in the space.
    assert determinant_count(8, 10, spin=2) == math.comb(8, 6) * math.comb(8, 4)


def test_determinant_count_rejects_impossible_fillings():
    with pytest.raises(ValueError):
        determinant_count(4, 10)


def test_ci_vector_bytes_is_eight_per_determinant():
    assert ci_vector_bytes(12, 10) == 8 * 627_264


def test_davidson_subspace_is_capped_at_the_project_convention():
    # Small spaces are not given a wider subspace than the rest of the project
    # uses, so a ladder rung and an H10 point are converged the same way.
    assert davidson_max_space(8, 10) == DAVIDSON_MAX_SPACE
    assert davidson_max_space(12, 10) == DAVIDSON_MAX_SPACE


def test_davidson_subspace_shrinks_once_the_vectors_get_large():
    space = davidson_max_space(16, 10)
    assert space is not None
    assert DAVIDSON_MIN_SPACE <= space < DAVIDSON_MAX_SPACE


def test_davidson_subspace_refuses_spaces_that_do_not_fit():
    # 20 and 26 orbitals are Phase 2: DMRG only, and the honest answer to "what
    # is the exact energy" is that it was not computed.
    assert davidson_max_space(20, 10) is None
    assert davidson_max_space(26, 10) is None


def test_davidson_subspace_is_monotone_in_the_active_space():
    spaces = [davidson_max_space(n, 10) for n in (8, 12, 16)]
    assert all(s is not None for s in spaces)
    assert spaces == sorted(spaces, reverse=True)


def test_davidson_budget_is_what_decides():
    # A larger budget buys a wider subspace at 16 orbitals; the guard is the
    # memory, not a hard-coded orbital count.
    tight = davidson_max_space(16, 10, budget_bytes=4 << 30)
    generous = davidson_max_space(16, 10, budget_bytes=32 << 30)
    assert tight is not None and generous is not None
    assert tight < generous
    # And a budget too small for even the minimum subspace refuses outright
    # rather than quietly running with a subspace that cannot separate the
    # near-degenerate roots stretched N2 produces.
    assert davidson_max_space(16, 10, budget_bytes=1 << 30) is None


def _toy_hamiltonian(n: int, seed: int = 0) -> ActiveSpaceHamiltonian:
    """A small Hamiltonian with the full eight-fold permutational symmetry."""
    rng = np.random.default_rng(seed)
    h1e = rng.normal(size=(n, n))
    h1e = 0.5 * (h1e + h1e.T)
    a = rng.normal(size=(n, n, n, n))
    # (pq|rs) = (qp|rs) = (pq|sr) = (rs|pq)
    a = a + a.transpose(1, 0, 2, 3)
    a = a + a.transpose(0, 1, 3, 2)
    g2e = a + a.transpose(2, 3, 0, 1)
    return ActiveSpaceHamiltonian(
        n_orbitals=n, n_electrons=2, spin=0, e_core_constant=1.25,
        e_nuclear_repulsion=1.25, h1e=h1e, g2e=g2e, orb_sym=list(range(n)),
    )


def test_reordering_relabels_the_integrals_consistently():
    asham = _toy_hamiltonian(5)
    order = [3, 0, 4, 1, 2]
    moved = reorder_active_space(asham, order)
    for i, p in enumerate(order):
        for j, q in enumerate(order):
            assert moved.h1e[i, j] == asham.h1e[p, q]
    assert moved.orb_sym == [asham.orb_sym[p] for p in order]


def test_reordering_is_invertible():
    asham = _toy_hamiltonian(6, seed=1)
    order = [4, 2, 0, 5, 1, 3]
    inverse = np.argsort(order)
    back = reorder_active_space(reorder_active_space(asham, order), inverse)
    assert np.allclose(back.h1e, asham.h1e)
    assert np.allclose(back.g2e, asham.g2e)
    assert back.digest() == asham.digest()


def test_reordering_preserves_the_eight_fold_symmetry():
    moved = reorder_active_space(_toy_hamiltonian(5, seed=2), [2, 4, 0, 3, 1])
    g = moved.g2e
    assert np.allclose(g, g.transpose(1, 0, 2, 3))
    assert np.allclose(g, g.transpose(0, 1, 3, 2))
    assert np.allclose(g, g.transpose(2, 3, 0, 1))


def test_reordering_changes_the_digest():
    # The digest is what proves two solvers consumed the same Hamiltonian. Two
    # orderings are the same *physics* but not the same integral arrays, and the
    # digest has to say so or it could never detect a real mismatch.
    asham = _toy_hamiltonian(5, seed=3)
    assert reorder_active_space(asham, [1, 0, 2, 3, 4]).digest() != asham.digest()


def test_reordering_rejects_anything_that_is_not_a_permutation():
    asham = _toy_hamiltonian(4)
    for bad in ([0, 1, 2], [0, 1, 2, 2], [0, 1, 2, 4]):
        with pytest.raises(ValueError):
            reorder_active_space(asham, bad)


def _labels_for(**overrides) -> str:
    point_kwargs = {"bond_length": 1.098, "n_active_orbitals": 8,
                    "orbital_order": "canonical"}
    schedule_kwargs = {"bond_dim": 250, "seed": 1234}
    for key, value in overrides.items():
        if key in point_kwargs:
            point_kwargs[key] = value
        else:
            schedule_kwargs[key] = value
    point = LadderPoint(**point_kwargs)
    return dmrg_method_label(point, DMRGSchedule.ramped(**schedule_kwargs))


def test_every_setting_that_changes_the_calculation_changes_the_method_label():
    # The store key is built from the system and the method, never from chi,
    # seed or ordering -- those are not properties of the molecule. So they have
    # to reach the key through the label, or a whole sweep collapses onto one
    # cached point.
    base = _labels_for()
    assert _labels_for(bond_dim=500) != base
    assert _labels_for(seed=5678) != base
    assert _labels_for(orbital_order="fiedler") != base
    assert _labels_for(n_active_orbitals=12) != base
    assert _labels_for(cutoff=1e-9) != base


def test_the_label_is_stable_for_an_unchanged_calculation():
    assert _labels_for() == _labels_for()


def test_distinct_labels_give_distinct_store_keys():
    point = LadderPoint(1.098, n_active_orbitals=8)
    keys = {
        record_key(point, _labels_for()),
        record_key(point, _labels_for(bond_dim=500)),
        record_key(point, "CASCI(10e,8o)"),
    }
    assert len(keys) == 3


def test_the_frozen_core_matches_the_valence_active_space():
    # Every rung correlates the same ten valence electrons; the two N 1s pairs
    # stay inactive throughout, which is what makes energies comparable along
    # the ladder.
    assert N2_FROZEN_CORE == 2
    assert VALENCE_ELECTRONS == 10
    point = LadderPoint(1.098, n_active_orbitals=26)
    assert point.hamiltonian.n_frozen_core == N2_FROZEN_CORE
    assert point.hamiltonian.n_active_orbitals == 26


def test_the_point_reports_the_determinant_count_of_its_own_rung():
    assert LadderPoint(1.6, n_active_orbitals=16).n_determinants == 19_079_424


@pytest.mark.slow
def test_reordering_leaves_the_exact_energy_unchanged():
    """The claim the whole ordering comparison rests on, on real integrals.

    A permutation of orbitals cannot move an eigenvalue. If it did, the
    canonical and Fiedler runs would be solving different problems and their
    cost difference would mean nothing.
    """
    from tn_quantum_chemistry.n2_ladder import build_active_space, fiedler_ordering
    from tn_quantum_chemistry.validation import run_fci

    point = LadderPoint(1.098, n_active_orbitals=8)
    _, asham, _ = build_active_space(point)
    e_canonical, _ = run_fci(asham, max_space=DAVIDSON_MAX_SPACE)
    order = fiedler_ordering(asham)
    e_fiedler, _ = run_fci(
        reorder_active_space(asham, order), max_space=DAVIDSON_MAX_SPACE
    )
    assert abs(e_fiedler - e_canonical) < 1e-11


@pytest.mark.slow
def test_the_active_space_hamiltonian_is_reproducible_across_runs():
    """Two builds of the same rung give the same integrals, digest included.

    Without the SCF thread pinning this fails: the RHF energies still agree to
    3e-13 while the orbitals differ, so the active-space Hamiltonian is a
    different one on almost every run and the exact reference moves by ~1e-9.
    That surfaced as DMRG appearing to beat the variational bound.
    """
    from pyscf import lib

    from tn_quantum_chemistry.n2_ladder import build_active_space

    point = LadderPoint(1.098, n_active_orbitals=8)
    digests = []
    for threads in (1, 32):
        with lib.with_omp_threads(threads):
            _, asham, _ = build_active_space(point)
        digests.append(asham.digest())
    assert len(set(digests)) == 1


# --------------------------------------------------------------------------- #
# The all-electron rung
# --------------------------------------------------------------------------- #


def test_the_all_electron_rung_is_full_ci_in_the_basis():
    """CAS(14e,28o) correlates every electron in every cc-pVDZ orbital.

    It exists because PySCF's NEVPT2 excites out of the CASSCF core, so the
    vignette's NEVPT2 energy carries core correlation that the frozen-core
    CAS(10e,26o) does not -- 3.9 mHa at 1.098 A, measured with frozen-core
    against all-electron CCSD(T). Only this rung is comparable to it without a
    correction.
    """
    point = LadderPoint(1.098, n_active_orbitals=28, n_frozen_core=0)
    assert point.n_electrons == 14
    assert point.hamiltonian.n_frozen_core == 0
    assert point.n_determinants == math.comb(28, 7) ** 2 == 1_401_950_721_600


def test_the_valence_ladder_keeps_ten_electrons_at_every_rung():
    for n_orb in PHASE_1_RUNGS + PHASE_2_RUNGS:
        assert LadderPoint(1.098, n_active_orbitals=n_orb).n_electrons == 10


def test_freezing_the_core_changes_the_store_key():
    """The frozen core is part of the record key, so the two ladders cannot
    collide even at an active-space size they could both reach."""
    valence = LadderPoint(1.098, n_active_orbitals=26, n_frozen_core=2)
    all_electron = LadderPoint(1.098, n_active_orbitals=26, n_frozen_core=0)
    method = "CASCI(10e,26o)"
    assert record_key(valence, method) != record_key(all_electron, method)


def test_the_method_label_reports_the_electron_count_it_correlated():
    point = LadderPoint(1.098, n_active_orbitals=28, n_frozen_core=0)
    label = dmrg_method_label(point, DMRGSchedule.ramped(1000))
    assert label.startswith("DMRG(14e,28o,")


def test_no_phase_two_rung_has_an_affordable_exact_reference():
    """The split between the phases is a memory fact, not a convention."""
    for n_orb in PHASE_1_RUNGS:
        assert davidson_max_space(n_orb, 10) is not None
    for n_orb in PHASE_2_RUNGS:
        assert davidson_max_space(n_orb, 10) is None
    assert davidson_max_space(28, 14) is None
