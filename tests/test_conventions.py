"""Day-2 convention tests: units, spin, frozen core, orbital ordering, nuclear repulsion.

These are the conventions that silently corrupt an electronic-structure
comparison when they are assumed rather than checked. Each one is pinned here
as an executable assertion so a later refactor cannot quietly change it.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pytest

from tn_quantum_chemistry.entanglement import symmetry_resolved_profile
from tn_quantum_chemistry.schema import BOHR_TO_ANGSTROM
from tn_quantum_chemistry.validation import (
    DMRGSchedule,
    Geometry,
    HamiltonianSpec,
    active_space_hamiltonian,
    build_mean_field,
    diatomic,
    giner_delta_to_project,
    linear_hydrogen_chain,
    project_delta_to_giner,
    run_dmrg,
    run_fci,
)

STO3G = HamiltonianSpec(basis="sto-3g")


@pytest.fixture(scope="module")
def n2_hamiltonian():
    """N2/STO-3G at 1.1 A, the Day-2 reference system."""
    mf = build_mean_field(diatomic("N", 1.1, label="N2"), STO3G)
    return mf, active_space_hamiltonian(mf, STO3G)


# --------------------------------------------------------------------------- #
# Geometry conventions
# --------------------------------------------------------------------------- #


def test_hydrogen_chain_bonds_alternate_by_half_delta():
    """This project's convention is bond lengths R +/- delta/2, long bond first."""
    R, delta = 1.8, 0.4
    z = np.array([a[1][2] for a in linear_hydrogen_chain(10, R=R, delta=delta).atoms])
    bonds = np.diff(z)
    assert np.allclose(bonds[0::2], R + delta / 2)
    assert np.allclose(bonds[1::2], R - delta / 2)


def test_hydrogen_chain_is_uniform_and_centred_at_zero_delta():
    z = np.array([a[1][2] for a in linear_hydrogen_chain(10, R=1.8).atoms])
    assert np.allclose(np.diff(z), 1.8)
    assert abs(z.mean()) < 1e-12


def test_giner_convention_conversion_matches_bond_lengths():
    """Giner et al. are read as R +/- 2*delta against our R +/- delta/2.

    The test compares the quantity that actually matters -- the bond length --
    rather than the nominal parameter value. Note what it cannot establish: the
    premise that their convention really is ``R +/- 2*delta`` comes from the
    project plan, not from the paper, which is paywalled. This checks the
    conversion is self-consistent with our own geometry builder, nothing more.
    """
    R, delta_giner = 1.8, 0.1
    delta_project = giner_delta_to_project(delta_giner)
    assert delta_project == pytest.approx(4 * delta_giner)
    assert project_delta_to_giner(delta_project) == pytest.approx(delta_giner)

    z = np.array([a[1][2] for a in linear_hydrogen_chain(10, R=R, delta=delta_project).atoms])
    assert np.diff(z)[0] == pytest.approx(R + 2 * delta_giner)
    assert np.diff(z)[1] == pytest.approx(R - 2 * delta_giner)


def test_dimerisation_larger_than_spacing_is_rejected():
    with pytest.raises(ValueError, match="non-positive bond length"):
        linear_hydrogen_chain(4, R=1.0, delta=3.0)


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #


def test_angstrom_and_bohr_geometries_give_the_same_energy():
    """The same physical geometry expressed in either unit must agree."""
    r_ang = 1.1
    in_angstrom = diatomic("N", r_ang, label="N2_ang")
    half_bohr = 0.5 * r_ang / BOHR_TO_ANGSTROM
    in_bohr = Geometry(
        label="N2_bohr",
        atoms=[("N", (0.0, 0.0, -half_bohr)), ("N", (0.0, 0.0, half_bohr))],
        unit="Bohr",
    )
    e_ang = build_mean_field(in_angstrom, STO3G).e_tot
    e_bohr = build_mean_field(in_bohr, STO3G).e_tot
    assert e_ang == pytest.approx(e_bohr, abs=1e-10)


# --------------------------------------------------------------------------- #
# Nuclear repulsion
# --------------------------------------------------------------------------- #


def test_project_bohr_constant_matches_pyscf():
    """One Bohr/Angstrom convention in the repository, and it is PySCF's.

    PySCF's constant governs every geometry built here, so the project uses the
    same value rather than the newer CODATA 2018 figure. If a PySCF upgrade
    changes it, this test says so.
    """
    from pyscf.data import nist

    assert BOHR_TO_ANGSTROM == nist.BOHR


def test_nuclear_repulsion_matches_the_analytic_diatomic_value():
    """E_nuc for a diatomic is Z_A Z_B / r with r in Bohr -- checked independently."""
    r_ang = 1.1
    mf = build_mean_field(diatomic("N", r_ang, label="N2"), STO3G)
    expected = 7 * 7 / (r_ang / BOHR_TO_ANGSTROM)
    assert float(mf.mol.energy_nuc()) == pytest.approx(expected, rel=1e-12)


def test_electronic_plus_nuclear_equals_total(n2_hamiltonian):
    _, asham = n2_hamiltonian
    e_total, _ = run_fci(asham)
    e_electronic = e_total - asham.e_nuclear_repulsion
    assert e_electronic + asham.e_nuclear_repulsion == pytest.approx(e_total, abs=1e-14)


def test_core_constant_equals_nuclear_repulsion_without_a_frozen_core(n2_hamiltonian):
    _, asham = n2_hamiltonian
    assert asham.e_core_constant == pytest.approx(asham.e_nuclear_repulsion, abs=1e-10)


# --------------------------------------------------------------------------- #
# Frozen core
# --------------------------------------------------------------------------- #


def test_frozen_core_reduces_the_space_and_shifts_only_the_core_constant():
    """Freezing the two N 1s orbitals gives CAS(10e,8o).

    The core constant absorbs the frozen electrons' energy while the nuclear
    repulsion is untouched, which is exactly why electronic energy must be
    computed as E_total - E_nuc and never as E_total - ecore.
    """
    geometry = diatomic("N", 1.1, label="N2")
    all_electron = active_space_hamiltonian(build_mean_field(geometry, STO3G), STO3G)

    frozen_spec = HamiltonianSpec(basis="sto-3g", n_frozen_core=2, n_active_orbitals=8)
    frozen = active_space_hamiltonian(build_mean_field(geometry, frozen_spec), frozen_spec)

    assert (frozen.n_orbitals, frozen.n_electrons) == (8, 10)
    assert frozen.e_nuclear_repulsion == pytest.approx(all_electron.e_nuclear_repulsion)
    assert frozen.e_core_constant != pytest.approx(frozen.e_nuclear_repulsion, abs=1.0)


def test_frozen_core_energy_is_variationally_above_all_electron_fci():
    """Restricting the correlation space can only raise the energy."""
    geometry = diatomic("N", 1.1, label="N2")
    e_all, _ = run_fci(active_space_hamiltonian(build_mean_field(geometry, STO3G), STO3G))

    frozen_spec = HamiltonianSpec(basis="sto-3g", n_frozen_core=2, n_active_orbitals=8)
    e_frozen, _ = run_fci(
        active_space_hamiltonian(build_mean_field(geometry, frozen_spec), frozen_spec)
    )
    assert e_frozen > e_all


# --------------------------------------------------------------------------- #
# Spin sector
# --------------------------------------------------------------------------- #


def test_block2_spin_two_s_matches_pyscf_alpha_beta_counts(n2_hamiltonian):
    """block2's ``spin`` is 2S, and PySCF's nelec tuple must follow from it."""
    _, asham = n2_hamiltonian
    assert asham.spin == 0
    assert asham.nelec_pair == (7, 7)
    assert dataclasses.replace(asham, spin=2).nelec_pair == (8, 6)


@pytest.mark.slow
def test_triplet_sector_agrees_between_solvers(n2_hamiltonian):
    """The Ms=1 / S=1 sector must give the same energy in both solvers."""
    _, asham = n2_hamiltonian
    triplet = dataclasses.replace(asham, spin=2)
    e_fci, _ = run_fci(triplet)
    e_dmrg, _, _ = run_dmrg(triplet, DMRGSchedule.default())
    assert e_fci == pytest.approx(e_dmrg, abs=1e-8)
    assert e_fci > run_fci(asham)[0]  # the triplet lies above the singlet


@pytest.mark.slow
def test_multi_root_fci_never_returns_a_higher_energy_than_single_root(n2_hamiltonian):
    """The ``nroots`` guard in ``run_fci`` can only help, never hurt.

    The guard exists because PySCF's single-root Davidson follows the state
    seeded from the lowest-diagonal determinant and can converge to an excited
    root. On N2/STO-3G in the Ms=1 sector it has been observed returning
    -107.343458537272, the third root, where the true sector ground state is the
    doubly degenerate -107.356943001688 that block2 finds directly.
    ``fci.addons.fix_spin_`` does not help: the problem is root selection, not
    spin contamination, and every root there already has <S^2> = 2.

    Whether the trap fires is not asserted here. It depends on floating-point
    detail -- block2 alters global MKL threading, which is enough to change
    which root Davidson lands on -- so an assertion that the bug manifests would
    pass or fail according to test ordering. What is stable, and what the guard
    actually promises, is that asking for more roots never yields a higher
    energy. Correctness of the guarded value is established separately, by
    agreement with DMRG in :func:`test_triplet_sector_agrees_between_solvers`.
    """
    _, asham = n2_hamiltonian
    triplet = dataclasses.replace(asham, spin=2)
    e_guarded, _ = run_fci(triplet)
    e_single_root, _ = run_fci(triplet, nroots=1)
    assert e_guarded <= e_single_root + 1e-9


# --------------------------------------------------------------------------- #
# Orbital ordering
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_converged_energies_are_invariant_under_orbital_permutation():
    """Both solvers must return the same energy for a relabelled orbital basis.

    DMRG cost and truncation error do depend on ordering; the converged energy
    does not. The integral digest is expected to change, because ordering is
    part of the Hamiltonian as presented to the solver even when the spectrum
    is not.
    """
    geometry = linear_hydrogen_chain(10, R=1.6, delta=0.0, label="H10")
    asham = active_space_hamiltonian(build_mean_field(geometry, STO3G), STO3G)
    schedule = DMRGSchedule.default()

    e_fci, _ = run_fci(asham)
    e_dmrg, _, _ = run_dmrg(asham, schedule)

    permutation = np.random.default_rng(7).permutation(asham.n_orbitals)
    permuted = dataclasses.replace(
        asham,
        h1e=asham.h1e[np.ix_(permutation, permutation)],
        g2e=asham.g2e[np.ix_(permutation, permutation, permutation, permutation)],
        orb_sym=[asham.orb_sym[i] for i in permutation],
    )
    e_fci_permuted, _ = run_fci(permuted)
    e_dmrg_permuted, _, _ = run_dmrg(permuted, schedule)

    assert e_fci_permuted == pytest.approx(e_fci, abs=1e-10)
    assert e_dmrg_permuted == pytest.approx(e_dmrg, abs=1e-9)
    assert permuted.digest() != asham.digest()


def test_integral_digest_is_stable_and_sector_sensitive(n2_hamiltonian):
    """The digest identifies the Hamiltonian both solvers were handed."""
    _, asham = n2_hamiltonian
    assert asham.digest() == asham.digest()
    assert dataclasses.replace(asham, spin=2).digest() != asham.digest()


# --------------------------------------------------------------------------- #
# Dimerisation sign
# --------------------------------------------------------------------------- #


def test_positive_and_negative_delta_are_different_geometries():
    """An open H10 chain has nine bonds, so the sign of delta matters.

    With ``delta > 0`` the pattern starts long and ends long: five long bonds
    and four short. With ``delta < 0`` it is five short and four long. These are
    not related by any symmetry of the chain -- reflection maps the sequence
    onto itself, not onto its complement -- so they are distinct systems.
    """
    positive = linear_hydrogen_chain(10, R=1.8, delta=0.4)
    negative = linear_hydrogen_chain(10, R=1.8, delta=-0.4)

    bonds_pos = np.diff([atom[1][2] for atom in positive.atoms])
    bonds_neg = np.diff([atom[1][2] for atom in negative.atoms])

    assert len(bonds_pos) == 9  # odd, which is the whole point
    assert (bonds_pos > 1.8).sum() == 5
    assert (bonds_neg > 1.8).sum() == 4
    # Reflection maps each onto itself, not onto the other.
    assert np.allclose(bonds_pos, bonds_pos[::-1])
    assert not np.allclose(bonds_pos, bonds_neg)


@pytest.mark.slow
def test_negative_delta_is_lower_in_energy():
    """Five short bonds beat four, and the gap is far from negligible.

    ``delta < 0`` pairs all ten atoms into five H2 molecules; ``delta > 0``
    gives four H2 and two unpaired terminal atoms. The fully paired arrangement
    is markedly more stable -- 50 mHa at R = 1.8 A, and 210 mHa at R = 1.0 A.

    A surface sampled only over ``delta >= 0`` would therefore be half a
    surface, missing the lower-energy half. The symmetry that would justify
    sampling one sign holds for rings and for chains with an even number of
    bonds, not for this one.
    """
    energies = {}
    for dimerisation in (0.4, -0.4):
        geometry = linear_hydrogen_chain(10, R=1.8, delta=dimerisation)
        asham = active_space_hamiltonian(build_mean_field(geometry, STO3G), STO3G)
        energies[dimerisation], _ = run_fci(asham)

    assert energies[-0.4] < energies[0.4]
    assert energies[0.4] - energies[-0.4] == pytest.approx(5.05e-2, rel=0.05)


def _all_schedules() -> dict[str, DMRGSchedule]:
    """Every constructor on :class:`DMRGSchedule`, at a few representative sizes.

    Enumerated in one place because the guards below are only as good as their
    coverage: ``ramped`` was absent from this list, and shipped with a noise ramp
    that finished before the sweep reached its own bond dimension.
    """
    return {
        "default": DMRGSchedule.default(),
        "fixed_bond_dimension(64)": DMRGSchedule.fixed_bond_dimension(64),
        "fixed_bond_dimension(500)": DMRGSchedule.fixed_bond_dimension(500),
        "ramped(250)": DMRGSchedule.ramped(250),
        "ramped(500)": DMRGSchedule.ramped(500, n_sweeps=24),
        "ramped(1000)": DMRGSchedule.ramped(1000, n_sweeps=24),
    }


def test_every_schedule_carries_noise_at_its_own_bond_dimension():
    """Perturbative noise must still be on once the sweep has room to use it.

    This is the invariant ``DMRGSchedule.ramped`` violated. It spent its whole
    noise budget while chi was a sixteenth of the target and then reached the
    full bond dimension with noise already at zero. On N2 in CAS(10e,16o) at
    1.6 A that converged -- reproducibly, at every chi and both seeds, with a
    final discarded weight of 5e-10 -- to a state 0.19 Ha above the ground
    state. The same point converges correctly under a schedule whose noise
    overlaps its peak bond dimension.

    Nothing available without an exact reference detected it: the discarded
    weight was tiny, the seeds agreed with each other, and the entanglement of
    the wrong state was not distinguishable from the right one. So the schedule
    itself has to be correct by construction.
    """
    for name, schedule in _all_schedules().items():
        peak = max(schedule.bond_dims)
        noises = list(schedule.noises)
        noises += [0.0] * (len(schedule.bond_dims) - len(noises))
        noisy_near_peak = [
            i
            for i, (bond_dim, noise) in enumerate(zip(schedule.bond_dims, noises))
            if bond_dim >= peak / 2 and noise > 0.0
        ]
        assert noisy_near_peak, (
            f"{name}: the noise ramp ends before the sweep reaches half its own "
            f"bond dimension, so the sweep cannot escape a local minimum once it "
            f"has the freedom to; bond_dims={schedule.bond_dims}, noises={noises}"
        )


def test_every_schedule_ends_without_noise():
    """The reported energy has to be a variational expectation value.

    A sweep still carrying perturbative noise reports an energy with that noise
    folded in, which is neither an upper bound nor comparable with an exact
    reference. Every schedule therefore has to switch the noise off before it
    stops.
    """
    for name, schedule in _all_schedules().items():
        noises = list(schedule.noises)
        noises += [0.0] * (len(schedule.bond_dims) - len(noises))
        assert noises[-2:] == [0.0, 0.0], f"{name}: last sweeps still carry noise"


def test_every_schedule_specifies_all_of_its_own_sweeps():
    """No schedule may rely on the backend padding a short list.

    ``block2`` extends a schedule that is shorter than ``n_sweeps`` by repeating
    its last entry, so a mismatch is silent and the sweeps beyond the list run
    under settings nobody chose.
    """
    for name, schedule in _all_schedules().items():
        assert len(schedule.bond_dims) == schedule.n_sweeps, name
        assert len(schedule.davidson_thresholds) == schedule.n_sweeps, name


def test_every_schedule_converges_the_local_eigensolver_tightly():
    """No schedule may quietly ship a loose Davidson threshold.

    This guards a regression that survived for most of the project: the
    fixed-bond-dimension schedule lived in a *script* with its own
    ``davidson_thresholds=[1e-12]`` while the library schedules had been
    tightened to 1e-14. The stored sector profiles were therefore built with a
    different solver setting from the dissociation curve they were compared
    against, and nothing failed -- the numbers were merely inconsistent.

    A loose local solver is indistinguishable from bond-dimension truncation in
    the final energy, which is what made the DMRG/FCI agreement drift with chain
    spacing before the thresholds were tightened. The tolerance is therefore
    part of the project's conventions, not a tuning knob, and is asserted here
    for every constructor rather than trusted to review.
    """
    for name, schedule in _all_schedules().items():
        assert max(schedule.davidson_thresholds) <= 1e-14, name
        assert schedule.energy_tol <= 1e-14, name

    # The schedules must also be the only place a threshold is chosen: a script
    # defining its own is how the inconsistency arose in the first place.
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = [
        f"{path.relative_to(root)}:{n}"
        for path in sorted((root / "scripts").rglob("*.py"))
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if "davidson_thresholds=" in line
    ]
    assert offenders == [], (
        "scripts must use a DMRGSchedule constructor rather than assembling "
        f"their own thresholds: {offenders}"
    )


def test_singular_value_cutoff_can_bind_before_the_bond_dimension():
    """The effective bond dimension is set by whichever limit binds first.

    A bond-dimension cap alone imposes the same number of renormalised states on
    every geometry, whether or not the state needs them. With a cutoff active the
    kept basis adapts: where the Schmidt spectrum decays quickly the cutoff binds
    and the run is cheaper than requested, and where it does not the cap binds as
    before.

    Asserted by running the same geometry twice at a bond dimension that is
    deliberately larger than the state needs, once with a loose cutoff and once
    with block2's default -- which is far below anything these runs reach and so
    never binds.
    """
    geometry = linear_hydrogen_chain(10, R=1.0, delta=0.0, label="H10_cutoff")
    asham = active_space_hamiltonian(build_mean_field(geometry, STO3G), STO3G)

    capped = DMRGSchedule.fixed_bond_dimension(200, n_sweeps=8)
    cut = DMRGSchedule.fixed_bond_dimension(200, n_sweeps=8, cutoff=1e-5)
    assert capped.cutoff == 1e-20

    def bond_dims(schedule):
        """Bond dimensions of the converged MPS.

        ``sweep_bond_dims`` in the returned statistics echoes the *requested*
        schedule and is identical with and without a cutoff, so it cannot answer
        this question; the MPS itself has to be measured.
        """
        captured = {}
        run_dmrg(asham, schedule, post_process=lambda driver, ket: captured.update(
            profile=symmetry_resolved_profile(driver, ket, n_orbitals=asham.n_orbitals)
        ))
        return captured["profile"].to_dict()["bond_dimensions"]

    assert max(bond_dims(cut)) < max(bond_dims(capped))


def test_fci_uses_the_project_davidson_subspace_everywhere():
    """The FCI Davidson subspace is a convention, not a per-call knob.

    PySCF's default of 12 is not enough for a near-dissociated chain: the
    subspace is repeatedly restarted and loses the directions it needs, and the
    symptom is a converged MPS sitting *below* FCI by an amount that grows with
    chain spacing. Measured at R = 3.4 A, raising it from 30 to 200 removes the
    last 3.9e-12 Ha, and 200 agrees with 300 to every digit.

    Asserted rather than reviewed because the same class of drift -- one caller
    quietly keeping a looser setting than the rest of the project -- already cost
    this repository weeks with the DMRG sweep schedule.
    """
    import inspect

    from tn_quantum_chemistry import vignette

    default = inspect.signature(run_fci).parameters["max_space"].default
    assert default >= 200, f"run_fci default subspace fell to {default}"
    assert vignette.FCI_MAX_SPACE >= 200, (
        f"the vignette solver kept its own subspace of {vignette.FCI_MAX_SPACE}"
    )
