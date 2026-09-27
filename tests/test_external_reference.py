"""Reproduction of a published benchmark: Motta et al., Phys. Rev. X 7, 031059.

This is the project's strongest external check. The Simons Collaboration
hydrogen-chain benchmark publishes RHF, CCSD, CCSD(T), FCI and DMRG energies for
the H10 linear chain in a minimal basis -- the same system and nearly the same
method ladder run here -- so agreement tests the whole pipeline against numbers
this project had no hand in producing.

Unlike the internal regression locks in ``test_reference_energies.py``, a
failure here means our result disagrees with the literature, not merely with our
own previous result.

Two conventions had to be established by reproduction, because the source
repository documents neither: the spacings are in **Bohr** and ``basis-STO`` is
**STO-6G**. ``test_basis_assignment_is_not_a_guess`` pins that down, so a future
reader does not have to take it on trust.
"""

from __future__ import annotations

import pytest

from tn_quantum_chemistry.methods import run_all_methods
from tn_quantum_chemistry.reference_data import (
    MOTTA_BASIS,
    motta_cc_is_comparable,
    motta_energy,
    motta_geometry,
    motta_spacings,
    motta_tolerance,
)
from tn_quantum_chemistry.validation import (
    HamiltonianSpec,
    active_space_hamiltonian,
    build_mean_field,
    run_fci,
)

STO6G = HamiltonianSpec(basis=MOTTA_BASIS)

#: A spread across the curve: near-equilibrium, intermediate, and stretched
#: where correlation is strong and coupled cluster has broken down.
SAMPLED = (1.4, 2.0, 2.8, 3.6)


def _fci(spacing: float) -> float:
    mf = build_mean_field(motta_geometry(spacing), STO6G)
    energy, _ = run_fci(active_space_hamiltonian(mf, STO6G))
    return energy


# --------------------------------------------------------------------------- #
# The reproduction
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.parametrize("spacing", motta_spacings())
def test_fci_reproduces_the_published_benchmark(spacing):
    """Our FCI must match theirs at every published spacing."""
    published = motta_energy(spacing, "FCI")
    assert _fci(spacing) == pytest.approx(published, abs=motta_tolerance("FCI"))


@pytest.mark.slow
@pytest.mark.parametrize("spacing", SAMPLED)
def test_method_ladder_reproduces_the_published_benchmark(spacing):
    """RHF, FCI and DMRG all match, not only the exact solver.

    Checking more than FCI catches a class of error the FCI comparison cannot.
    FCI is invariant under orbital rotation, so it is blind to the reference
    determinant entirely; RHF and DMRG are not.

    Coupled cluster is compared only where its equations still have a physical
    solution -- see :func:`test_diverged_coupled_cluster_is_not_compared`.
    """
    records = {
        record.method: record
        for record in run_all_methods(
            motta_geometry(spacing), STO6G, dmrg_bond_dims=(500,)
        )
    }
    comparisons = [("RHF", "RHF"), ("FCI", "FCI"), ("DMRG(chi=500)", "DMRG")]
    if motta_cc_is_comparable(spacing):
        comparisons += [("CCSD", "CCSD"), ("CCSD(T)", "CCSD(T)")]

    for method, key in comparisons:
        published = motta_energy(spacing, key)
        assert published is not None, f"{key} missing from the vendored data"
        assert records[method].e_total == pytest.approx(
            published, abs=motta_tolerance(key)
        ), f"{method} disagrees with the published {key} at R = {spacing} Bohr"


@pytest.mark.slow
def test_diverged_coupled_cluster_is_not_compared():
    """Where CC has diverged, the two datasets agree qualitatively and not
    numerically -- and that is the correct expectation.

    At R = 3.6 Bohr the published CCSD(T) sits 1.86 Ha below FCI while ours sits
    0.15 Ha below. Neither is wrong: once the amplitude equations have no
    physical solution, which unphysical one a code reaches depends on its solver
    and starting guess.

    The energy is not reproducible even within one code. At R = 3.2 Bohr,
    repeated runs here give +0.068 and +0.127 Ha against FCI -- unconverged
    amplitudes, landing somewhere different each time. So neither the value nor
    its sign can be asserted; only the failure itself, which shows up as
    non-convergence or as a deviation far beyond chemical accuracy.
    """
    diverged = [s for s in motta_spacings() if not motta_cc_is_comparable(s)]
    assert diverged, "expected the published CC to diverge somewhere on this curve"

    for spacing in diverged:
        published_error = motta_energy(spacing, "CCSD(T)") - motta_energy(spacing, "FCI")
        assert published_error < -0.1  # theirs fails, and fails downward

        records = {
            record.method: record
            for record in run_all_methods(
                motta_geometry(spacing), STO6G, with_dmrg=False
            )
        }
        record = records["CCSD(T)"]
        deviation = abs(record.e_total - records["FCI"].e_total)
        assert not record.converged or deviation > 1e-2, (
            f"expected coupled cluster to fail at R = {spacing} Bohr, but it "
            f"converged to within {deviation:.2e} Ha of FCI"
        )


# --------------------------------------------------------------------------- #
# The conventions the source does not document
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_basis_assignment_is_not_a_guess():
    """STO-6G reproduces the benchmark; STO-3G is off by ~1e-2 Ha.

    The repository labels the basis only as 'STO'. The identification rests on
    this gap being four orders of magnitude, not on the name.
    """
    spacing = 1.8
    published = motta_energy(spacing, "FCI")

    assert _fci(spacing) == pytest.approx(published, abs=motta_tolerance("FCI"))

    mf = build_mean_field(motta_geometry(spacing), HamiltonianSpec(basis="sto-3g"))
    sto3g, _ = run_fci(active_space_hamiltonian(mf, HamiltonianSpec(basis="sto-3g")))
    assert abs(sto3g - published) > 1e-2


@pytest.mark.slow
def test_spacings_are_bohr_not_angstrom():
    """Reading the published spacings as Angstrom reproduces nothing.

    A silent unit mismatch is the most likely way this comparison could appear
    to succeed while being meaningless, so it is tested rather than assumed.
    """
    from tn_quantum_chemistry.validation import linear_hydrogen_chain

    spacing = 1.8
    published = motta_energy(spacing, "FCI")
    mf = build_mean_field(
        linear_hydrogen_chain(10, R=spacing, delta=0.0), STO6G
    )  # same number, read as Angstrom
    as_angstrom, _ = run_fci(active_space_hamiltonian(mf, STO6G))
    assert abs(as_angstrom - published) > 0.1


# --------------------------------------------------------------------------- #
# What the published data itself shows
# --------------------------------------------------------------------------- #


def test_published_coupled_cluster_also_falls_below_fci():
    """The CC breakdown is in the benchmark too, not an artefact of our setup.

    Reads only vendored numbers -- no calculation -- so it is a statement about
    the literature rather than about this code.
    """
    below = {
        spacing: motta_energy(spacing, "CCSD(T)") - motta_energy(spacing, "FCI")
        for spacing in motta_spacings()
        if motta_energy(spacing, "CCSD(T)") < motta_energy(spacing, "FCI")
    }
    assert below, "expected the published CCSD(T) to fall below FCI somewhere"
    assert max(below) >= 2.4  # the failure is at the stretched end
