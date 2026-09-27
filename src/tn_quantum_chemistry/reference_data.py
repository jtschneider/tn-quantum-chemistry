"""Published external reference values, vendored into the repository.

Two benchmarks are vendored. The first is the Simons Collaboration
hydrogen-chain benchmark (Motta et al.,
Phys. Rev. X 7, 031059 (2017), CC BY 4.0), which publishes RHF, CCSD, CCSD(T),
FCI and DMRG energies for the H10 linear chain in a minimal basis -- the same
system and very nearly the same method ladder this project runs.

The values live in ``data/reference/`` rather than being fetched on demand, so
the regression suite depends on neither the network nor a third-party repository
staying where it is. ``scripts/fetch_motta_reference.py`` refreshes them.

Their geometry unit and basis are not documented in the source repository; both
were established by reproduction and are recorded in the vendored file. See
:func:`motta_geometry` for the consequence: these spacings are in **Bohr**, and
the basis is **STO-6G**, so comparing against them means leaving this project's
usual Angstrom/STO-3G defaults behind on purpose.

The second is a pair of published near-exact N2/cc-pVDZ energies, which is the
only external check the N2 active-space ladder has above sixteen orbitals --
where the ladder has no reference of its own. See :func:`n2_external` and the
``unit_trap`` field of the vendored file: the two sources quote their bond
lengths in different units, and they are two different geometries, not one.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "data" / "reference"
MOTTA_PATH = REFERENCE_DIR / "motta_h10_obc_sto6g.json"
N2_EXTERNAL_PATH = REFERENCE_DIR / "n2_ccpvdz_external.json"

#: Basis and unit the benchmark actually uses, established by reproduction.
MOTTA_BASIS = "sto-6g"
MOTTA_UNIT = "Bohr"


@lru_cache(maxsize=1)
def motta_h10() -> dict[str, Any]:
    """The vendored benchmark, as a plain dict."""
    if not MOTTA_PATH.exists():
        raise FileNotFoundError(
            f"{MOTTA_PATH} is missing; run scripts/fetch_motta_reference.py"
        )
    return json.loads(MOTTA_PATH.read_text())


def motta_spacings() -> list[float]:
    """Published nearest-neighbour spacings, in Bohr."""
    return sorted(float(key) for key in motta_h10()["energies"])


def motta_energy(spacing: float, method: str) -> float | None:
    """Published total energy in Hartree, or None if that method is absent."""
    row = motta_h10()["energies"].get(f"{spacing:.1f}")
    return None if row is None else row.get(method)


#: Comparison tolerances, in Hartree. Mostly a few units in the last published
#: decimal, with one deliberate exception.
#:
#: RHF is printed to ten decimals but only reproduces to about 3e-6. The cause
#: is not our SCF: PySCF's solution is internally stable at every spacing, and
#: the discrepancy varies smoothly with R and changes sign near R = 1.7 Bohr,
#: which is the signature of a slightly different converged solution rather than
#: an error. FCI cannot see it at all -- it is invariant under orbital rotation,
#: and reproduces to 4e-9 at the same geometries. Using the printed decimals here
#: would assert a precision the published value does not carry.
MOTTA_TOLERANCES = {
    "FCI": 5e-8,       # 8 published decimals; achieved 4e-9
    "DMRG": 5e-6,      # 6 published decimals; achieved 1.8e-6
    "CCSD": 5e-6,      # 6 published decimals; achieved 4e-7 where CC converges
    "CCSD(T)": 5e-6,   # as above
    "RHF": 5e-6,       # empirical, not from the printed decimals -- see above
    "UHF": 5e-6,
}

#: Beyond this separation from FCI the coupled-cluster amplitude equations have
#: no physical solution left, and the energy stops being a reproducible
#: quantity: which unphysical solution a code lands on depends on its solver and
#: initial guess. Both datasets diverge past R = 3.2 Bohr, but to different
#: values -- theirs to 1.9 Ha below FCI, ours to 0.11 Ha below. Comparing those
#: numbers would be meaningless, so :func:`motta_cc_is_comparable` excludes them
#: and the divergence is asserted qualitatively instead.
CC_DIVERGENCE_THRESHOLD = 0.1


def motta_tolerance(method: str) -> float:
    """Comparison tolerance in Hartree for one method. See :data:`MOTTA_TOLERANCES`."""
    return MOTTA_TOLERANCES[method]


def motta_cc_is_comparable(spacing: float) -> bool:
    """Whether coupled cluster is still a meaningful number at this spacing."""
    published = motta_energy(spacing, "CCSD(T)")
    reference = motta_energy(spacing, "FCI")
    if published is None or reference is None:
        return False
    return abs(published - reference) < CC_DIVERGENCE_THRESHOLD


def motta_geometry(spacing: float, label: str | None = None):
    """The benchmark geometry: H10, open chain, spacing in Bohr, centred.

    Returned in Bohr rather than converted, so no Bohr/Angstrom constant enters
    the comparison at all.
    """
    from tn_quantum_chemistry.validation import Geometry

    z = np.arange(10) * spacing
    z = z - z.mean()
    return Geometry(
        label=label or f"H10_motta_R{spacing:.1f}bohr",
        atoms=[("H", (0.0, 0.0, float(value))) for value in z],
        unit=MOTTA_UNIT,
        charge=0,
        spin=0,
        parameters={"R_bohr": spacing, "n_atoms": 10.0},
    )


# --------------------------------------------------------------------------
# N2 / cc-pVDZ: published near-exact energies
# --------------------------------------------------------------------------
#
# The ladder's top rung is full CI in cc-pVDZ, 1.4e12 determinants, and nothing
# in this repository can check it. These two published values can, and they are
# at *different geometries in different units* -- see ``unit_trap`` in the
# vendored file. Neither geometry is one this project already runs, so comparing
# against them means running the ladder at their bond length, not interpolating
# to it.

#: cc-pVDZ with spherical d functions: 28 spatial orbitals for N2. Cartesian d
#: would give 30 and a different energy, so this is part of the comparison.
N2_BASIS = "cc-pvdz"


@lru_cache(maxsize=1)
def n2_external() -> dict[str, Any]:
    """The vendored N2/cc-pVDZ benchmark values, as a plain dict."""
    if not N2_EXTERNAL_PATH.exists():
        raise FileNotFoundError(f"{N2_EXTERNAL_PATH} is missing")
    return json.loads(N2_EXTERNAL_PATH.read_text())


def n2_reference(space: str) -> dict[str, Any]:
    """One published rung: ``"all_electron"`` or ``"frozen_core"``."""
    block = n2_external().get(space)
    if block is None:
        raise KeyError(
            f"unknown N2 reference space {space!r}; "
            f"expected one of {sorted(k for k in n2_external() if isinstance(n2_external()[k], dict) and 'best_estimate' in n2_external()[k])}"
        )
    return block


def n2_energy(space: str, method: str) -> float | None:
    """Published total energy in Hartree, or None if that method is absent."""
    return n2_reference(space)["energies"].get(method)


def n2_best_estimate(space: str) -> tuple[float, float]:
    """The near-exact value and its uncertainty, both in Hartree.

    The uncertainty is not a printed-decimal count. For the all-electron rung it
    is the spread between two independent near-exact methods (CDFCI and the DMRG
    of Chan et al.); for the frozen-core rung it is the published statistical
    error on a semistochastic perturbative correction, which is why that value
    resolves only at 1e-4 and is not variational.
    """
    block = n2_reference(space)
    return float(block["best_estimate"]), float(block["best_estimate_uncertainty"])


def n2_geometry(space: str, label: str | None = None):
    """The published geometry, in the unit the source actually used.

    Returned unconverted for the same reason as :func:`motta_geometry`: the
    all-electron source quotes Bohr and the frozen-core source Angstrom, so
    converting either one would put a Bohr/Angstrom constant inside the
    comparison it is meant to test.
    """
    from tn_quantum_chemistry.validation import Geometry

    block = n2_reference(space)
    bond_length = float(block["bond_length"])
    unit = block["bond_length_unit"]
    return Geometry(
        label=label or f"N2_{space}_R{bond_length:g}{unit.lower()[:4]}",
        atoms=[
            ("N", (0.0, 0.0, -bond_length / 2.0)),
            ("N", (0.0, 0.0, bond_length / 2.0)),
        ],
        unit=unit,
        charge=0,
        spin=0,
        parameters={"bond_length": bond_length},
    )


def n2_bond_length_angstrom(space: str) -> float:
    """The published bond length converted to Angstrom, for the ladder driver.

    ``scripts/n2_active_space_ladder.py`` takes Angstrom. This is the one place
    the conversion is allowed to happen, and it uses PySCF's own constant so the
    geometry the driver builds is bit-identical to :func:`n2_geometry`.
    """
    from pyscf.data.nist import BOHR

    block = n2_reference(space)
    bond_length = float(block["bond_length"])
    if block["bond_length_unit"] == "Angstrom":
        return bond_length
    return bond_length * BOHR


#: How many published points on each side are used to measure the interpolation
#: error near a target. Four is enough to see the local curvature without
#: averaging over the whole curve, where the error is dominated by the steep
#: repulsive wall rather than by the region being asked about.
N2_CURVE_LOO_NEIGHBOURS = 4


@lru_cache(maxsize=1)
def n2_curve() -> dict[float, float]:
    """The published all-electron binding curve: bond length in Bohr to energy."""
    block = n2_external()["all_electron_curve"]
    return {float(k): float(v) for k, v in block["energies"].items()}


@lru_cache(maxsize=1)
def _n2_curve_loo_errors() -> list[tuple[float, float]]:
    """Leave-one-out spline error at every interior published point.

    This is the honest way to put an uncertainty on an interpolated reference:
    drop a point the source actually published, predict it from its neighbours,
    and see how far off the prediction is. Nothing here is assumed about how
    smooth the curve is -- it is measured on the curve itself.

    It errs the safe way. Deleting a point *doubles* the local grid spacing, so
    the prediction is made across a gap twice as wide as the one a real query
    sits inside. The measured error is therefore an over-estimate of the error
    this project actually incurs, which is the direction an uncertainty on a
    reference value should be wrong in.
    """
    from scipy.interpolate import CubicSpline

    curve = n2_curve()
    x = np.array(sorted(curve))
    y = np.array([curve[value] for value in x])
    errors = []
    # The two endpoints on each side have no two-sided neighbourhood to be
    # predicted from, so they cannot be scored this way.
    for index in range(2, len(x) - 2):
        kept_x = np.delete(x, index)
        kept_y = np.delete(y, index)
        prediction = float(CubicSpline(kept_x, kept_y)(x[index]))
        errors.append((float(x[index]), abs(prediction - y[index])))
    return errors


def n2_curve_interpolation_error(bond_length_bohr: float) -> float:
    """Measured interpolation error near this bond length, in Hartree."""
    errors = _n2_curve_loo_errors()
    nearest = sorted(errors, key=lambda item: abs(item[0] - bond_length_bohr))
    return max(error for _, error in nearest[:N2_CURVE_LOO_NEIGHBOURS])


def n2_curve_energy(bond_length_bohr: float) -> tuple[float, float]:
    """Published energy at an arbitrary bond length, with its uncertainty.

    Raises ``ValueError`` outside the published range rather than extrapolating.
    This project's own 2.4 Angstrom geometry is 4.535 Bohr and the curve stops at
    4.500, so that point genuinely cannot be checked this way -- and an
    extrapolated "reference" would be this project's number dressed up as
    someone else's.

    The uncertainty adds the curve's own to the locally measured interpolation
    error, because both are present and neither dominates.
    """
    from scipy.interpolate import CubicSpline

    curve = n2_curve()
    x = np.array(sorted(curve))
    if not x.min() <= bond_length_bohr <= x.max():
        raise ValueError(
            f"{bond_length_bohr:.6f} Bohr is outside the published curve "
            f"[{x.min()}, {x.max()}]; extrapolation is refused"
        )
    y = np.array([curve[value] for value in x])
    energy = float(CubicSpline(x, y)(bond_length_bohr))
    block = n2_external()["all_electron_curve"]
    uncertainty = float(block["uncertainty"]) + n2_curve_interpolation_error(bond_length_bohr)
    return energy, uncertainty
