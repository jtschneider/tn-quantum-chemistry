"""Published near-exact N2/cc-pVDZ energies, and the conventions they pin down.

The N2 ladder's top rung is full CI in cc-pVDZ -- 1.4e12 determinants -- and
nothing in this repository can check it. ``tests/test_n2_ladder.py`` asserts that
no Phase 2 rung has an affordable exact reference, which is exactly the problem:
above sixteen orbitals the ladder's uncertainty rests on the gap between two
orbital orderings, and that estimator has never been tested against a known
answer.

Two published values can test it, and neither is at a geometry this project
already runs:

* **all-electron**, 2.118 Bohr: CDFCI and the DMRG of Chan et al. agree to
  1.6e-5 Ha. This is the sharp one -- roughly seventy times finer than the
  ladder's own uncertainty at that rung.
* **frozen-core**, 1.0977 Angstrom: SHCI, quoted as -109.2769(1). Not
  variational, and it resolves only at 1e-4.

The two sources quote bond lengths in *different units* and they are 0.023 A
apart, so they are two distinct geometries rather than one value in two
conventions. That is the trap these tests exist to keep closed; see the
``unit_trap`` field of ``data/reference/n2_ccpvdz_external.json``.

The comparison of the ladder's own energies against these values lives in
``scripts/n2_external_check.py``, which reads the stores. Here we test only the
published values and the conventions under which they are comparable at all.
"""

from __future__ import annotations

import pytest

from tn_quantum_chemistry.n2_ladder import ALL_ELECTRON_ORBITALS, determinant_count
from tn_quantum_chemistry.reference_data import (
    N2_BASIS,
    n2_best_estimate,
    n2_bond_length_angstrom,
    n2_curve,
    n2_curve_energy,
    n2_curve_interpolation_error,
    n2_energy,
    n2_external,
    n2_geometry,
    n2_reference,
)
from tn_quantum_chemistry.validation import HamiltonianSpec, build_mean_field

CCPVDZ = HamiltonianSpec(basis=N2_BASIS)

#: The published RHF energy carries seven decimals, so this is its own rounding
#: rather than a claim about our SCF. Achieved: 2.1e-8.
RHF_TOLERANCE = 5e-8

#: D2h has order eight. The CDFCI paper quotes the determinant count of the
#: symmetry-reduced space; the ladder runs in C1 and quotes the full one.
D2H_ORDER = 8


# --------------------------------------------------------------------------
# The conventions, established by reproduction rather than assumed
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_published_rhf_is_reproduced_at_the_published_geometry():
    """The sharpest cheap check that we and they mean the same Hamiltonian.

    RHF is a few seconds and fixes the basis, the geometry, the unit and the
    spherical/Cartesian convention all at once. If this passes, a disagreement
    further up the ladder is a statement about correlation treatment rather than
    about setup.
    """
    published = n2_energy("all_electron", "RHF")
    mf = build_mean_field(n2_geometry("all_electron"), CCPVDZ)
    assert mf.converged
    assert abs(mf.e_tot - published) < RHF_TOLERANCE


@pytest.mark.slow
def test_the_basis_is_spherical_and_that_is_not_a_guess():
    """Cartesian d functions give a different basis, and would pass unnoticed.

    cc-pVDZ with pure d gives 28 spatial orbitals for N2 and with Cartesian d
    gives 30. Both converge, both look like cc-pVDZ in a log file, and only one
    reproduces the published RHF energy. The all-electron rung is called "full
    CI in cc-pVDZ" on the strength of that orbital count, so it is worth pinning.
    """
    from pyscf import gto, scf

    geometry = n2_geometry("all_electron")
    published = n2_energy("all_electron", "RHF")

    spherical = gto.M(
        atom=geometry.atom_spec, unit=geometry.unit, basis=N2_BASIS, cart=False, verbose=0
    )
    cartesian = gto.M(
        atom=geometry.atom_spec, unit=geometry.unit, basis=N2_BASIS, cart=True, verbose=0
    )
    assert spherical.nao_nr() == ALL_ELECTRON_ORBITALS == 28
    assert cartesian.nao_nr() == 30

    cart_mf = scf.RHF(cartesian)
    cart_mf.conv_tol = CCPVDZ.scf_conv_tol
    cart_mf.kernel()
    # Not merely different: different by four orders of magnitude more than the
    # tolerance the spherical basis meets.
    assert abs(cart_mf.e_tot - published) > 1e-4


@pytest.mark.slow
def test_the_shci_geometry_offset_is_small_against_our_own_uncertainty():
    """Why the frozen-core comparison is allowed without a new calculation.

    Our surface has a point at 1.098 A; SHCI is at 1.0977 A. The comparison is
    only meaningful if that 0.0003 A offset moves the energy by less than the
    uncertainty we quote on the rung, which is 5.3e-4 Ha. CCSD(T) is used to
    measure the offset because the *slope* is what is wanted here, and every
    correlated method agrees on it to within a few times 1e-5.
    """
    from pyscf import cc, scf

    from tn_quantum_chemistry.validation import Geometry

    def ccsd_t(bond_length: float) -> float:
        geometry = Geometry(
            label=f"N2_R{bond_length}",
            atoms=[
                ("N", (0.0, 0.0, -bond_length / 2.0)),
                ("N", (0.0, 0.0, bond_length / 2.0)),
            ],
            unit="Angstrom",
            charge=0,
            spin=0,
            parameters={"bond_length": bond_length},
        )
        mf = build_mean_field(geometry, CCPVDZ)
        assert isinstance(mf, scf.hf.RHF)
        mycc = cc.CCSD(mf, frozen=2)
        mycc.conv_tol = 1e-11
        mycc.kernel()
        assert mycc.converged
        return mycc.e_tot + mycc.ccsd_t()

    shift = ccsd_t(1.098) - ccsd_t(n2_bond_length_angstrom("frozen_core"))
    assert abs(shift) < 5.3e-4, (
        "the geometry offset is no longer small against the ladder's own "
        "uncertainty, so the frozen-core comparison needs a run at 1.0977 A"
    )


# --------------------------------------------------------------------------
# The vendored values themselves
# --------------------------------------------------------------------------


def test_the_two_published_geometries_are_different_points():
    """They are quoted in different units, and are not the same bond length."""
    all_electron = n2_bond_length_angstrom("all_electron")
    frozen_core = n2_bond_length_angstrom("frozen_core")
    assert n2_reference("all_electron")["bond_length_unit"] == "Bohr"
    assert n2_reference("frozen_core")["bond_length_unit"] == "Angstrom"
    assert abs(all_electron - frozen_core) > 0.02
    assert "unit_trap" in n2_external()


def test_the_geometry_is_returned_in_the_unit_its_source_used():
    """No Bohr/Angstrom constant may enter the comparison itself."""
    assert n2_geometry("all_electron").unit == "Bohr"
    assert n2_geometry("frozen_core").unit == "Angstrom"


def test_the_all_electron_rung_is_the_space_the_publication_used():
    """Same electrons, same orbitals -- and the determinant counts reconcile."""
    block = n2_reference("all_electron")
    assert block["n_frozen_core"] == 0
    assert block["n_orbitals"] == ALL_ELECTRON_ORBITALS
    assert determinant_count(block["n_orbitals"], block["n_electrons"]) == block["n_determinants"]
    # The paper quotes 1.75e11 for the same space: this count reduced by D2h.
    assert block["n_determinants"] / D2H_ORDER == pytest.approx(1.75e11, rel=0.01)


def test_the_frozen_core_rung_is_the_valence_ladder_top():
    """SHCI freezes both N 1s cores, which is this project's valence rung."""
    block = n2_reference("frozen_core")
    assert block["n_frozen_core"] == 2
    assert block["n_electrons"] == 10
    assert block["n_orbitals"] == 26


def test_the_two_near_exact_methods_bracket_the_approximate_ones():
    """A transcription check with physical content.

    Every approximate method in the published table must sit above the near-exact
    value, and the ordering CCSD < MRCISD < MRCCSD < CCSDTQ < CDFCI must hold. A
    mistyped digit almost certainly breaks one of these.
    """
    best, _ = n2_best_estimate("all_electron")
    ladder = ["CCSD", "MRCISD", "MRCCSD", "CCSDTQ"]
    energies = [n2_energy("all_electron", method) for method in ladder]
    assert all(energy > best for energy in energies)
    assert energies == sorted(energies, reverse=True)
    assert n2_energy("all_electron", "RHF") > max(energies)


def test_the_uncertainty_covers_the_spread_between_the_two_near_exact_methods():
    """The stated uncertainty is measured, not asserted.

    CDFCI and DMRG are independent, so their disagreement is the honest floor on
    how well this reference is known. Quoting anything tighter would claim a
    precision two published methods do not agree on.
    """
    best, uncertainty = n2_best_estimate("all_electron")
    dmrg = n2_energy("all_electron", "DMRG")
    assert abs(best - dmrg) <= uncertainty


def test_the_frozen_core_value_is_not_presented_as_variational():
    """SHCI's total energy includes a stochastic perturbative correction.

    It may legitimately fall *below* the exact answer, so nothing in this project
    may use it as a variational bound.
    """
    block = n2_reference("frozen_core")
    assert block["best_estimate_method"] == "SHCI"
    assert "not variational" in block["uncertainty_note"]
    assert block["best_estimate_uncertainty"] == 1e-4


def test_the_all_electron_reference_is_sharper_than_the_ladder_uncertainty():
    """The point of the exercise: the reference must be able to test the rung.

    The ladder quotes 7.3e-4 Ha at the 28-orbital rung. A reference no better
    than that would confirm nothing.
    """
    _, uncertainty = n2_best_estimate("all_electron")
    assert uncertainty < 7.3e-4 / 10


def test_every_published_value_carries_its_citation():
    """No number without a source, and no source without a DOI or arXiv id."""
    citations = n2_external()["citations"]
    assert set(citations) >= {"chan2004", "cdfci2019", "shci2017"}
    for key, text in citations.items():
        assert any(marker in text for marker in ("doi:", "arXiv:", "J. Chem. Phys.")), key
    for space in ("all_electron", "frozen_core"):
        assert n2_reference(space)["source"]


# --------------------------------------------------------------------------
# The published curve, and interpolating onto it honestly
# --------------------------------------------------------------------------


def test_the_curve_reproduces_the_published_point_it_shares():
    """At 2.118 a0 the curve and the headline value are the same number.

    They come from different tables of the same paper, so this is a transcription
    check -- and it is also what justifies giving the curve the headline value's
    uncertainty.
    """
    headline, _ = n2_best_estimate("all_electron")
    interpolated, _ = n2_curve_energy(2.118)
    assert interpolated == pytest.approx(headline, abs=1e-12)


def test_the_curve_refuses_to_extrapolate():
    """2.4 Angstrom is 4.535 a0 and the curve stops at 4.500.

    An extrapolated value would be this project's own smoothness assumption
    wearing someone else's citation, so it is refused rather than returned with a
    wider error bar.
    """
    from pyscf.data.nist import BOHR

    with pytest.raises(ValueError, match="outside the published curve"):
        n2_curve_energy(2.4 / BOHR)


def test_the_two_checkable_geometries_are_inside_the_curve():
    """The other two of this project's geometries can be checked for free."""
    from pyscf.data.nist import BOHR

    for bond_length in (1.098, 1.600):
        energy, uncertainty = n2_curve_energy(bond_length / BOHR)
        assert -110.0 < energy < -108.0
        assert 0.0 < uncertainty < 1e-4


def test_the_interpolation_error_is_measured_rather_than_assumed():
    """Leave-one-out on the published grid, not a smoothness claim.

    The check is that the measurement is actually being done and is small where
    this project uses it -- not that some particular number comes out.
    """
    from pyscf.data.nist import BOHR

    for bond_length in (1.098, 1.600):
        error = n2_curve_interpolation_error(bond_length / BOHR)
        assert 0.0 < error < 5e-5
    # And that it is folded into what callers get back.
    _, uncertainty = n2_curve_energy(1.098 / BOHR)
    assert uncertainty > n2_curve_interpolation_error(1.098 / BOHR)


def test_the_curve_is_the_same_active_space_as_the_top_rung():
    """A curve for a different correlation space would not be comparable."""
    curve_block = n2_external()["all_electron_curve"]
    point_block = n2_reference("all_electron")
    for field in ("n_electrons", "n_orbitals", "n_frozen_core"):
        assert curve_block[field] == point_block[field]


def test_the_curve_is_monotone_on_each_side_of_its_minimum():
    """A transcription check with physical content, over all 42 points."""
    curve = n2_curve()
    lengths = sorted(curve)
    minimum = min(lengths, key=lambda r: curve[r])
    assert minimum == pytest.approx(2.118)
    rising = [curve[r] for r in lengths if r <= minimum]
    falling = [curve[r] for r in lengths if r >= minimum]
    assert rising == sorted(rising, reverse=True)
    assert falling == sorted(falling)
