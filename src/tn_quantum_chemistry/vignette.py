"""Day 6: the chemistry-facing N2 vignette.

H10/STO-3G hides every choice a practising computational chemist actually
makes. There is one sensible basis, no core to freeze, no active space to
select, and no orbital whose character can change under the geometry. This
module makes those choices visible on a molecule where they matter, at three
bond lengths and in two bases:

* **RHF and UHF**, with the RHF stability analysis that says *when* the
  restricted solution stops being a minimum, and ``<S^2>`` for the broken-symmetry
  solution that replaces it. A UHF energy below RHF is not an improvement in
  accuracy; it is a different, spin-contaminated state.
* **MP2, CCSD and CCSD(T)**, which are excellent near equilibrium and become
  unreliable exactly where the reference stops being single-determinantal.
* **CASCI and CASSCF** in the full-valence ``CAS(10e,8o)`` obtained by leaving
  the two N 1s orbitals inactive, with active-space natural occupations
  recorded at every geometry, because an active space that silently changes
  character along a stretch produces a smooth-looking curve of different
  states.
* **NEVPT2** on the converged CASSCF, which is the step that puts back the
  dynamical correlation outside the active space -- the thing a bare CASSCF
  number does not have and a bare DMRG-in-an-active-space number does not have
  either.

The vignette is deliberately three geometries and two bases. It is not a second
dataset and carries no headline claim; its job is to demonstrate judgement about
references, active spaces and basis incompleteness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tn_quantum_chemistry.methods import _patch_pyscf_numpy2_diis
from tn_quantum_chemistry.schema import (
    CalculationRecord,
    ReferenceType,
    Status,
)
from tn_quantum_chemistry.validation import Geometry, HamiltonianSpec, diatomic

#: Full-valence active space for N2: the 1s pair on each nitrogen stays
#: inactive, leaving 10 valence electrons in the 8 orbitals built from 2s and
#: 2p. This is the textbook choice for the triple-bond dissociation and the one
#: that keeps sigma, sigma*, pi and pi* all inside the space.
VALENCE_CAS_ELECTRONS = 10
VALENCE_CAS_ORBITALS = 8

#: Equilibrium, moderately stretched, and stretched far enough that the
#: restricted reference and the coupled-cluster expansion are both in trouble.
DEFAULT_BOND_LENGTHS = (1.098, 1.600, 2.400)

#: A minimal basis for continuity with the rest of the project, and one
#: correlation-consistent basis to make basis incompleteness measurable rather
#: than merely mentioned.
DEFAULT_BASES = ("sto-3g", "cc-pvdz")

#: Occupations further from 0 or 2 than this mark an orbital as genuinely
#: partially occupied. 0.02 is the usual informal threshold for "this system is
#: multireference"; it is a reporting convention, not a physical constant.
OCCUPATION_TOLERANCE = 0.02

#: Threads for the whole vignette. One, and not as a compromise: on N2 in
#: these bases the single-threaded run is *faster* (2.7 s against 9.9 s on 32
#: threads -- the system is far too small to amortise the thread overhead) and
#: it is bit-reproducible, while the parallel run is not. The parallel
#: non-determinism is real and was measured rather than assumed: repeated
#: NEVPT2 calls on one converged CASSCF object differ by ~2e-5 Ha on 8 threads
#: and not at all on one, and two parallel CASSCF runs reach the same energy to
#: 1e-12 with orbitals differing by O(1) -- the degenerate pi pair is free to
#: rotate, and NEVPT2 then lands 1e-5 Ha away. Physically irrelevant beside a
#: 150 mHa correction; not irrelevant in a table printed to six decimals.
OMP_THREADS = 1

#: Convergence thresholds. Set explicitly rather than inherited, so that a
#: PySCF default change cannot silently move a published number.
#: 1e-12 and not the 1e-14 that ``run_fci`` uses on H10. PySCF's tolerance is
#: absolute, and N2 sits near -109 Ha against H10's -5 Ha: 1e-14 there is 1e-16
#: relative, which is machine epsilon, so the solver can never report
#: convergence and every CASCI came back flagged as failed. An absolute
#: threshold is not transferable between systems of different total energy.
FCI_CONV_TOL = 1e-12
CASSCF_CONV_TOL = 1e-10
CASSCF_CONV_TOL_GRAD = 1e-6

#: Raised from PySCF's default of 100 so the wider subspace has room to work.
FCI_MAX_CYCLE = 2000

#: Davidson subspace, raised from PySCF's default of 12 to match ``run_fci``.
#: Stretched N2 has the same problem as the stretched hydrogen chain: the
#: M_S = 0 sector fills with near-degenerate triplets that a small subspace
#: cannot separate, so the reference drifts systematically with bond length.
#: Wider is not slower here -- it converges instead of restarting.
FCI_MAX_SPACE = 200


@dataclass(frozen=True)
class VignetteSpec:
    """One (bond length, basis) point of the vignette."""

    bond_length: float
    basis: str
    run_nevpt2: bool = True
    run_cc: bool = True
    omp_threads: int = OMP_THREADS
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def geometry(self) -> Geometry:
        return diatomic("N", self.bond_length, label=f"N2_R{self.bond_length:.3f}")

    @property
    def hamiltonian(self) -> HamiltonianSpec:
        return HamiltonianSpec(
            basis=self.basis,
            n_frozen_core=0,
            n_active_orbitals=VALENCE_CAS_ORBITALS,
            orbital_source="RHF canonical / CASSCF optimised",
        )


def _base_fields(spec: VignetteSpec, mol) -> dict[str, Any]:
    geometry = spec.geometry
    return {
        "system_label": geometry.label,
        "basis": spec.basis,
        "atoms": [[symbol, *position] for symbol, position in geometry.atoms],
        "geometry_parameters": {"bond_length": spec.bond_length},
        "coordinate_unit": geometry.unit,
        "charge": geometry.charge,
        "spin_two_s": geometry.spin,
        "n_orbitals": int(mol.nao_nr()),
        "n_electrons": int(mol.nelectron),
        "e_nuclear_repulsion": float(mol.energy_nuc()),
    }


def _record(spec, mol, method, *, reference, e_total, walltime, status, extra=None,
            n_frozen_core=0) -> CalculationRecord:
    return CalculationRecord(
        method=method,
        reference_type=reference,
        n_frozen_core=n_frozen_core,
        e_total=float(e_total),
        status=status,
        walltime_seconds=float(walltime),
        extra=extra or {},
        **_base_fields(spec, mol),
    )


def build_molecule(spec: VignetteSpec):
    from pyscf import gto

    geometry = spec.geometry
    return gto.M(
        atom=geometry.atom_spec,
        basis=spec.basis,
        unit=geometry.unit,
        charge=geometry.charge,
        spin=geometry.spin,
        symmetry=False,
        verbose=0,
    )


def natural_occupations(mc) -> dict[str, Any]:
    """Active-space natural occupations of a converged CASCI/CASSCF solution.

    The diagnostic that says whether the active space still describes the same
    physics: a triple bond at equilibrium has occupations near (2,2,2,0,0,0),
    and a stretched one moves toward six singly occupied orbitals. If instead
    an orbital leaves the space and a core-like or diffuse one takes its place,
    the energies stay smooth while the *state* changes underneath them.
    """
    density = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
    occupations = np.sort(np.linalg.eigvalsh(density))[::-1]
    partially_occupied = int(
        np.sum(
            (occupations > OCCUPATION_TOLERANCE)
            & (occupations < 2.0 - OCCUPATION_TOLERANCE)
        )
    )
    return {
        "active_natural_occupations": [float(value) for value in occupations],
        "n_partially_occupied": partially_occupied,
        "occupation_sum": float(occupations.sum()),
        "max_deviation_from_integer": float(
            np.max(np.minimum(occupations, np.abs(2.0 - occupations)))
        ),
        "occupation_tolerance": OCCUPATION_TOLERANCE,
    }


def _follow_instabilities(mf, *, max_cycles: int = 5) -> tuple[Any, int, bool]:
    """Re-converge an SCF solution until its internal stability analysis passes.

    A converged SCF is a stationary point, not necessarily a minimum. For a
    closed-shell singlet the default UHF guess lands on the RHF solution; only
    following the instability finds the broken-symmetry solution that is lower
    in energy at stretched geometries. Reporting "UHF" without this step
    reports RHF under a different name.
    """
    followed = 0
    for _ in range(max_cycles):
        try:
            mo, _, stable, _ = mf.stability(return_status=True)
        except (TypeError, ValueError):
            return mf, followed, False
        if stable:
            return mf, followed, True
        density = mf.make_rdm1(mo, mf.mo_occ)
        mf = mf.run(density)
        followed += 1
    return mf, followed, False


def run_references(
    spec: VignetteSpec, mol
) -> tuple[list[CalculationRecord], Any, float]:
    """RHF with its stability analysis, and the stabilised UHF solution."""
    from pyscf import scf

    _patch_pyscf_numpy2_diis()
    records: list[CalculationRecord] = []

    start = time.perf_counter()
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-12
    rhf.kernel()
    rhf_walltime = time.perf_counter() - start

    try:
        _, _, stable_internal, stable_external = rhf.stability(return_status=True)
    except (TypeError, ValueError):
        stable_internal = stable_external = None

    records.append(
        _record(
            spec, mol, "RHF",
            reference=ReferenceType.RHF,
            e_total=rhf.e_tot,
            walltime=rhf_walltime,
            status=Status.CONVERGED if rhf.converged else Status.NOT_CONVERGED,
            extra={
                "internally_stable": stable_internal,
                "externally_stable": stable_external,
                "spin_squared": 0.0,
            },
        )
    )

    start = time.perf_counter()
    uhf = scf.UHF(mol)
    uhf.conv_tol = 1e-12
    uhf.kernel()
    uhf, n_followed, uhf_stable = _follow_instabilities(uhf)
    uhf_walltime = time.perf_counter() - start
    spin_squared, multiplicity = uhf.spin_square()

    records.append(
        _record(
            spec, mol, "UHF",
            reference=ReferenceType.UHF,
            e_total=uhf.e_tot,
            walltime=uhf_walltime,
            status=Status.CONVERGED if uhf.converged else Status.NOT_CONVERGED,
            extra={
                "spin_squared": float(spin_squared),
                "multiplicity": float(multiplicity),
                "instabilities_followed": n_followed,
                "internally_stable": bool(uhf_stable),
                # Positive means UHF found a lower, symmetry-broken solution.
                # That is a different state, not a better energy for the same one.
                "rhf_minus_uhf_hartree": float(rhf.e_tot - uhf.e_tot),
            },
        )
    )
    return records, rhf, rhf_walltime


def run_coupled_cluster(
    spec: VignetteSpec, mol, rhf, *, n_frozen: int, scf_walltime: float
) -> list[CalculationRecord]:
    """MP2, CCSD and CCSD(T) on the RHF reference, with the core frozen.

    The core is frozen to the same two orbitals left inactive in
    ``CAS(10e,8o)``, so the single-reference and multireference tiers correlate
    the same electrons and their energies are comparable.

    Wall times include the SCF the tier is built on, matching the convention in
    ``methods.py``: the number answers "what does this tier cost" rather than
    "what does this step cost once someone else has paid for the reference".
    """
    from pyscf import cc, mp

    records: list[CalculationRecord] = []
    common = {"reference": ReferenceType.RHF, "n_frozen_core": n_frozen}

    start = time.perf_counter() - scf_walltime
    try:
        mp2 = mp.MP2(rhf, frozen=n_frozen).run()
        records.append(
            _record(spec, mol, "MP2", e_total=mp2.e_tot,
                    walltime=time.perf_counter() - start,
                    status=Status.CONVERGED, **common)
        )
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        records.append(
            _record(spec, mol, "MP2", e_total=np.nan,
                    walltime=time.perf_counter() - start,
                    status=Status.FAILED, extra={"error": f"{type(exc).__name__}: {exc}"},
                    **common)
        )

    start = time.perf_counter() - scf_walltime
    ccsd = cc.CCSD(rhf, frozen=n_frozen)
    ccsd.max_cycle = 100
    ccsd.kernel()
    ccsd_walltime = time.perf_counter() - start
    records.append(
        _record(spec, mol, "CCSD", e_total=ccsd.e_tot, walltime=ccsd_walltime,
                status=Status.CONVERGED if ccsd.converged else Status.NOT_CONVERGED,
                extra={"t1_diagnostic": float(ccsd.get_t1_diagnostic())
                       if hasattr(ccsd, "get_t1_diagnostic") else None},
                **common)
    )

    start = time.perf_counter()
    try:
        triples = ccsd.ccsd_t()
        records.append(
            _record(spec, mol, "CCSD(T)", e_total=ccsd.e_tot + triples,
                    walltime=ccsd_walltime + (time.perf_counter() - start),
                    # The (T) correction inherits the amplitudes: if CCSD did
                    # not converge, CCSD(T) is not converged either, whatever
                    # the perturbative step returns.
                    status=Status.CONVERGED if ccsd.converged else Status.NOT_CONVERGED,
                    extra={"triples_correction_hartree": float(triples)},
                    **common)
        )
    except Exception as exc:  # noqa: BLE001
        records.append(
            _record(spec, mol, "CCSD(T)", e_total=np.nan,
                    walltime=time.perf_counter() - start, status=Status.FAILED,
                    extra={"error": f"{type(exc).__name__}: {exc}"}, **common)
        )
    return records


def run_multireference(
    spec: VignetteSpec, mol, rhf, *, n_frozen: int, scf_walltime: float
) -> list[CalculationRecord]:
    """CASCI on RHF orbitals, CASSCF with optimised orbitals, then NEVPT2."""
    from pyscf import mcscf

    records: list[CalculationRecord] = []
    n_orb, n_elec = VALENCE_CAS_ORBITALS, VALENCE_CAS_ELECTRONS
    space = {"active_electrons": n_elec, "active_orbitals": n_orb}

    start = time.perf_counter() - scf_walltime
    casci = mcscf.CASCI(rhf, n_orb, n_elec)
    casci.fcisolver.conv_tol = FCI_CONV_TOL
    casci.fcisolver.max_space = FCI_MAX_SPACE
    casci.fcisolver.max_cycle = FCI_MAX_CYCLE
    casci.kernel()
    records.append(
        _record(spec, mol, f"CASCI({n_elec}e,{n_orb}o)",
                reference=ReferenceType.RHF, n_frozen_core=n_frozen,
                e_total=casci.e_tot, walltime=time.perf_counter() - start,
                status=Status.CONVERGED if casci.converged else Status.NOT_CONVERGED,
                extra={**space, **natural_occupations(casci),
                       "orbitals": "RHF canonical"})
    )

    start = time.perf_counter() - scf_walltime
    casscf = mcscf.CASSCF(rhf, n_orb, n_elec)
    casscf.max_cycle_macro = 100
    # Tighter than the default, because the vignette quotes these energies to
    # six decimals and reports an NEVPT2 correction on top of them. At PySCF's
    # default tolerances two runs of the same deterministic calculation differed
    # by ~1e-5 Ha -- orbital-rotation convergence noise, harmless in a bond
    # energy and not harmless in a printed table.
    casscf.conv_tol = CASSCF_CONV_TOL
    casscf.conv_tol_grad = CASSCF_CONV_TOL_GRAD
    casscf.fcisolver.conv_tol = FCI_CONV_TOL
    casscf.fcisolver.max_space = FCI_MAX_SPACE
    casscf.fcisolver.max_cycle = FCI_MAX_CYCLE
    casscf.kernel()
    casscf_walltime = time.perf_counter() - start
    records.append(
        _record(spec, mol, f"CASSCF({n_elec}e,{n_orb}o)",
                reference=ReferenceType.CASSCF, n_frozen_core=n_frozen,
                e_total=casscf.e_tot, walltime=casscf_walltime,
                status=Status.CONVERGED if casscf.converged else Status.NOT_CONVERGED,
                extra={**space, **natural_occupations(casscf),
                       "orbitals": "CASSCF optimised"})
    )

    if spec.run_nevpt2:
        from pyscf import mrpt

        start = time.perf_counter()
        try:
            # The thread count is pinned for the whole point (see OMP_THREADS);
            # this call is the one most sensitive to it.
            correction = mrpt.NEVPT(casscf).kernel()
            records.append(
                _record(spec, mol, f"NEVPT2/CASSCF({n_elec}e,{n_orb}o)",
                        reference=ReferenceType.CASSCF, n_frozen_core=n_frozen,
                        e_total=casscf.e_tot + correction,
                        walltime=casscf_walltime + (time.perf_counter() - start),
                        status=Status.CONVERGED if casscf.converged
                        else Status.NOT_CONVERGED,
                        extra={**space,
                               "nevpt2_correction_hartree": float(correction),
                               "omp_threads": 1,
                               "note": "external dynamical correlation on top "
                                       "of the active space"})
            )
        except Exception as exc:  # noqa: BLE001
            records.append(
                _record(spec, mol, f"NEVPT2/CASSCF({n_elec}e,{n_orb}o)",
                        reference=ReferenceType.CASSCF, n_frozen_core=n_frozen,
                        e_total=np.nan, walltime=time.perf_counter() - start,
                        status=Status.FAILED,
                        extra={"error": f"{type(exc).__name__}: {exc}"})
            )
    return records


def run_fci_reference(
    spec: VignetteSpec, mol, rhf, *, scf_walltime: float, max_orbitals: int = 12
) -> list[CalculationRecord]:
    """All-electron FCI, where the basis is small enough to afford it.

    In STO-3G this is affordable and turns the vignette's minimal-basis column
    into a checkable statement rather than an assertion: N2/STO-3G has ten
    orbitals, so leaving the two 1s orbitals inactive makes ``CAS(10e,8o)``
    *exactly* frozen-core FCI -- the active space is the entire remaining
    space. The gap between this all-electron FCI and CASCI is therefore the
    core correlation energy and nothing else, and the near-vanishing NEVPT2
    correction in this basis is explained rather than mysterious: there are no
    external orbitals left for it to correlate.

    cc-pVDZ has 28 orbitals and no such reference, which is the point of
    running both.
    """
    if mol.nao_nr() > max_orbitals:
        return []
    from pyscf import fci

    start = time.perf_counter() - scf_walltime
    try:
        solver = fci.FCI(rhf)
        solver.conv_tol = FCI_CONV_TOL
        solver.max_space = FCI_MAX_SPACE
        solver.max_cycle = FCI_MAX_CYCLE
        e_total = solver.kernel()[0]
    except Exception as exc:  # noqa: BLE001
        return [
            _record(spec, mol, "FCI", reference=ReferenceType.RHF, e_total=np.nan,
                    walltime=time.perf_counter() - start, status=Status.FAILED,
                    extra={"error": f"{type(exc).__name__}: {exc}"})
        ]
    return [
        _record(spec, mol, "FCI", reference=ReferenceType.RHF, e_total=e_total,
                walltime=time.perf_counter() - start, status=Status.CONVERGED,
                extra={"note": "all-electron; exact for this basis, electron "
                               "number and symmetry sector only"})
    ]


def run_vignette_point(spec: VignetteSpec) -> list[CalculationRecord]:
    """Every tier at one (bond length, basis) point."""
    from pyscf import lib

    mol = build_molecule(spec)
    n_frozen = 2  # the two N 1s orbitals, matching the CAS(10e,8o) inactive space
    with lib.with_omp_threads(spec.omp_threads):
        records, rhf, scf_walltime = run_references(spec, mol)
        if spec.run_cc:
            records += run_coupled_cluster(
                spec, mol, rhf, n_frozen=n_frozen, scf_walltime=scf_walltime
            )
        records += run_multireference(
            spec, mol, rhf, n_frozen=n_frozen, scf_walltime=scf_walltime
        )
        records += run_fci_reference(spec, mol, rhf, scf_walltime=scf_walltime)
    for record in records:
        # Wall times are only comparable at the thread count they were measured
        # at, so the count travels with them.
        record.extra = {**(record.extra or {}), "omp_threads": spec.omp_threads}
    return _attach_fci_discrepancies(records)


def _attach_fci_discrepancies(
    records: list[CalculationRecord],
) -> list[CalculationRecord]:
    """Fill in ``fci_discrepancy`` wherever an exact reference exists.

    Only STO-3G has one here. It turns "MP2 looks wrong at 2.4 A" into a
    number: MP2 lands 0.71 Ha *below* FCI for the same finite Hamiltonian,
    which is a failure of the perturbation series rather than an improvement,
    and it is recorded even though MP2 reports no convergence flag of its own
    to fail. In cc-pVDZ the field stays ``None``, because asserting a
    discrepancy against a reference that was never computed would be worse
    than leaving it empty.
    """
    reference = next(
        (r for r in records if r.method == "FCI" and r.status == Status.CONVERGED),
        None,
    )
    if reference is None:
        return records
    for record in records:
        if record.e_total is not None and np.isfinite(record.e_total):
            record.fci_discrepancy = float(record.e_total - reference.e_total)
    return records
