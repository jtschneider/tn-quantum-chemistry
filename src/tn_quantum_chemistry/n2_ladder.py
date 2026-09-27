"""N2 active-space ladder: the one system where the tensor network has work to do.

H10/STO-3G has ten orbitals for ten electrons. Nothing sits outside the active
space, so DMRG can only ever reproduce FCI, and it does so more slowly (5.4 s
against 2.1 s). Longer chains fail differently: past H16 there is no exact
reference left, and this project measured block2's discarded weight calling an
unconverged run converged, so the labels would be unverifiable.

N2 in cc-pVDZ escapes both. It has 28 orbitals for 14 electrons; freezing the
two N 1s cores leaves **26 orbitals for 10 valence electrons**, with a
determinant ladder underneath:

===============  ===============  ==========================
Active space     Determinants     Exact diagonalisation?
===============  ===============  ==========================
CAS(10e, 8o)               3,136  trivial (the vignette's space)
CAS(10e,12o)             627,264  easy
CAS(10e,16o)          19,079,424  feasible
CAS(10e,20o)         240,374,016  borderline
CAS(10e,26o)       4,327,008,400  out of reach here
===============  ===============  ==========================

DMRG spans the whole range, so there is an *overlap region* where it must
reproduce an exact answer and a *DMRG-only region* above it -- same molecule,
same basis, same orbitals on both sides. That is the structure the hydrogen
chains never had, and it is what makes a claim about the large space credible.

Phase 1 (this module's first use) validates DMRG against exact diagonalisation
in the 8-, 12- and 16-orbital spaces. Phase 2 goes above the wall and compares
the variational out-of-active-space correlation energy,
``E_DMRG(26o) - E_CASSCF(8o)``, against the NEVPT2 estimate of the same
quantity that the vignette already computes perturbatively.

Two settings here are *not* inherited from the H10 work, because both were
measured to break at this size:

* **The Davidson subspace has to shrink as the space grows.** ``run_fci``
  defaults to ``max_space=200``, chosen when a CI vector was 0.5 MB. At 16
  orbitals a vector is 153 MB and 200 of them are 61 GB. See
  :func:`davidson_max_space` -- another instance of an absolute setting that
  does not transfer between systems, like the ``conv_tol`` that broke CASCI in
  :mod:`~tn_quantum_chemistry.vignette`.
* **Orbital ordering is a real variable at 26 sites.** The RHF canonical order
  is energy-ordered, which is not an MPS-friendly order. See
  :func:`fiedler_ordering`; the ordering used is recorded per calculation
  rather than assumed.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tn_quantum_chemistry.schema import (
    CalculationRecord,
    RecordStore,
    ReferenceType,
    Status,
)
from tn_quantum_chemistry.validation import (
    ActiveSpaceHamiltonian,
    DMRGSchedule,
    Geometry,
    HamiltonianSpec,
    active_space_hamiltonian,
    build_mean_field,
    diatomic,
    run_dmrg,
    run_fci,
)

#: The two N 1s orbitals, left inactive exactly as in ``CAS(10e,8o)`` so that
#: every rung of the ladder correlates the same ten valence electrons and the
#: energies are comparable across the whole ladder.
N2_FROZEN_CORE = 2

#: Ten valence electrons: 2s and 2p on both nitrogens.
VALENCE_ELECTRONS = 10

#: All fourteen electrons of N2, for the all-electron rung.
N2_TOTAL_ELECTRONS = 14

#: The rungs. 8 is the vignette's full-valence space; 26 is every orbital left
#: after freezing the core in cc-pVDZ. 20 and 26 have no exact reference.
ACTIVE_SPACE_LADDER = (8, 12, 16, 20, 26)

#: The all-electron rung: every cc-pVDZ orbital, nothing frozen. ``CAS(14e,28o)``
#: is *full CI* in this basis -- 1.4e12 determinants -- and it exists for one
#: reason. PySCF's NEVPT2 takes ``ncore`` from the CASSCF object and generates
#: excitations out of those core orbitals, so the vignette's NEVPT2 energy
#: contains core correlation. ``CAS(10e,26o)`` freezes the core and does not.
#: Measured with frozen-core against all-electron CCSD(T) in the same basis,
#: that mismatch is 3.9 mHa at 1.098 A and 3.1 mHa at 1.600 A -- far larger than
#: any agreement worth quoting, so scoring NEVPT2 against the 26-orbital number
#: would be comparing two different quantities. The 28-orbital rung is
#: comparable to it without correction.
ALL_ELECTRON_ORBITALS = 28

#: Rungs with an exact reference (Phase 1) and rungs without (Phase 2). The
#: split is not a convention: it is where :func:`davidson_max_space` runs out of
#: memory, and it is the whole point of the ladder.
PHASE_1_RUNGS = (8, 12, 16)
PHASE_2_RUNGS = (20, 26)

#: block2 stack memory, in GB. Four is enough for the sixteen-orbital rung and
#: is not enough above it, so the Phase 2 script raises it -- but not as far as
#: it first did. block2 pre-allocates this arena and grows into it, so the value
#: is an upper bound on a job's resident size, not a hint. Set to 24 GB it let
#: three concurrent 26/28-orbital jobs at chi = 1000 reach 60 of 61 GB, and the
#: kernel started killing things. The probe that sized it had been run at
#: chi = 500, where peak RSS was 5-6 GB; four times the bond dimension is
#: roughly four times the tensors, and that was not re-measured before
#: committing.
#:
#: The value used travels with every record, because an out-of-memory failure
#: and a converged run are not distinguishable from the energy alone -- and one
#: of these runs was SIGKILLed, which leaves no record and no traceback at all.
DEFAULT_STACK_MEM_GB = 4

#: Stack for the rungs above the exact wall, and the reason concurrency is
#: capped at two jobs rather than three: 2 x 16 GB fits in 61 GB with the MPS
#: scratch and leaves headroom, 3 x 24 GB does not.
LARGE_RUNG_STACK_MEM_GB = 16
MAX_CONCURRENT_LARGE_JOBS = 2

#: The vignette's geometries, reused so the new numbers sit beside the existing
#: RHF/CCSD/CASSCF/NEVPT2 table rather than beside nothing.
DEFAULT_BOND_LENGTHS = (1.098, 1.600, 2.400)

#: Memory the exact solver is allowed for its Davidson subspace. Deliberately
#: well under the machine's free memory: PySCF allocates several working arrays
#: besides the subspace, and a swapping FCI is worse than an absent one.
DAVIDSON_BUDGET_BYTES = 8 << 30

#: Threads for the SCF that *defines* the active space. One, and not as a
#: performance compromise: on 32 threads the RHF here returns a different set of
#: orbitals on almost every run. The energies agree to 3e-13, so the SCF looks
#: converged, but the orbitals differ enough that the active-space Hamiltonian
#: they define is a different Hamiltonian -- four runs gave four distinct
#: integral digests and a CASCI energy moving over 6.4e-10 Ha. CASCI is not
#: stationary with respect to orbital rotations, so that noise lands directly in
#: the reference.
#:
#: This first showed up as an apparent variational violation: DMRG landing
#: 6.4e-10 Ha *below* "the" exact energy, which is impossible for one
#: Hamiltonian and merely means the two numbers came from two. Same mechanism as
#: the NEVPT2 non-determinism recorded in :mod:`~tn_quantum_chemistry.vignette`
#: -- N2's degenerate pi pair is free to rotate -- but here it moves the
#: reference itself rather than a correction on top of it. Pinned to one thread
#: the digest is bit-identical run to run.
SCF_OMP_THREADS = 1

#: PySCF's own default, and the floor this module will not go below. If even
#: twelve vectors do not fit, the space is not affordable and the exact
#: reference is skipped rather than attempted with a subspace too small to
#: separate the near-degenerate roots that stretched N2 produces.
DAVIDSON_MIN_SPACE = 12

#: The project convention for small systems, from ``run_fci``.
DAVIDSON_MAX_SPACE = 200


def determinant_count(n_orbitals: int, n_electrons: int, spin: int = 0) -> int:
    """Number of determinants in ``CAS(n_electrons, n_orbitals)``.

    ``spin`` is 2S in PySCF's convention. This is the size of the vector the
    exact solver has to hold, and the quantity that decides which rungs of the
    ladder have a reference at all.
    """
    n_alpha = (n_electrons + spin) // 2
    n_beta = n_electrons - n_alpha
    if not 0 <= n_alpha <= n_orbitals or not 0 <= n_beta <= n_orbitals:
        raise ValueError(
            f"cannot place {n_electrons} electrons (2S={spin}) in {n_orbitals} orbitals"
        )
    return math.comb(n_orbitals, n_alpha) * math.comb(n_orbitals, n_beta)


def ci_vector_bytes(n_orbitals: int, n_electrons: int, spin: int = 0) -> int:
    """Bytes in one double-precision CI vector for that active space."""
    return 8 * determinant_count(n_orbitals, n_electrons, spin)


def davidson_max_space(
    n_orbitals: int,
    n_electrons: int,
    spin: int = 0,
    *,
    budget_bytes: int = DAVIDSON_BUDGET_BYTES,
) -> int | None:
    """Largest affordable Davidson subspace, or ``None`` if the space is too big.

    PySCF's Davidson holds the subspace vectors *and* their images under the
    Hamiltonian, so the cost is about ``2 * max_space`` CI vectors. The
    project's ``max_space=200`` was set on H10, where a vector is 0.5 MB and the
    whole subspace is 200 MB; the same setting at 16 orbitals asks for 61 GB.

    Shrinking the subspace is not free -- a wide subspace is what stopped the
    reference drifting with geometry on stretched systems, where the M_S = 0
    sector fills with near-degenerate triplets -- so the value actually used is
    recorded with every calculation instead of being left implicit.
    """
    per_vector = 2 * ci_vector_bytes(n_orbitals, n_electrons, spin)
    affordable = int(budget_bytes // per_vector)
    if affordable < DAVIDSON_MIN_SPACE:
        return None
    return min(DAVIDSON_MAX_SPACE, affordable)


def fiedler_ordering(asham: ActiveSpaceHamiltonian, *, n_threads: int = 4) -> list[int]:
    """block2's Fiedler-vector orbital ordering for this Hamiltonian.

    An MPS is a one-dimensional object laid over a set of orbitals that have no
    intrinsic order. The RHF canonical order is by orbital energy, which puts
    strongly coupled orbitals arbitrarily far apart on the chain; the Fiedler
    ordering minimises a weighted bandwidth of the exchange matrix and is the
    standard QC-DMRG choice (block2, ORCA and QCMaquis all implement it).

    On ten orbitals this hardly matters. On twenty-six it is a design variable,
    so it is applied explicitly and recorded, never assumed.
    """
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    driver = DMRGDriver(
        symm_type=SymmetryTypes.SU2, n_threads=n_threads, stack_mem=4 << 30
    )
    driver.initialize_system(
        n_sites=asham.n_orbitals,
        n_elec=asham.n_electrons,
        spin=asham.spin,
        orb_sym=asham.orb_sym,
    )
    # Copies: block2 writes into the integral arrays it is handed.
    order = driver.orbital_reordering(
        np.array(asham.h1e, copy=True), np.array(asham.g2e, copy=True)
    )
    return [int(i) for i in order]


def reorder_active_space(
    asham: ActiveSpaceHamiltonian, order: list[int] | np.ndarray
) -> ActiveSpaceHamiltonian:
    """Permute the orbitals of ``asham``, leaving the spectrum invariant.

    A relabelling of orbitals cannot change any eigenvalue, so the exact energy
    is identical before and after; only the MPS's representation cost changes.
    That invariance is the test in :mod:`tests.test_n2_ladder`, and it is what
    makes an ordering comparison meaningful rather than a change of problem.
    """
    index = np.asarray(order, dtype=int)
    if index.shape != (asham.n_orbitals,) or sorted(index.tolist()) != list(
        range(asham.n_orbitals)
    ):
        raise ValueError(f"order must be a permutation of 0..{asham.n_orbitals - 1}")
    return ActiveSpaceHamiltonian(
        n_orbitals=asham.n_orbitals,
        n_electrons=asham.n_electrons,
        spin=asham.spin,
        e_core_constant=asham.e_core_constant,
        e_nuclear_repulsion=asham.e_nuclear_repulsion,
        h1e=np.ascontiguousarray(asham.h1e[np.ix_(index, index)]),
        g2e=np.ascontiguousarray(asham.g2e[np.ix_(index, index, index, index)]),
        orb_sym=[asham.orb_sym[i] for i in index],
    )


@dataclass(frozen=True)
class LadderPoint:
    """One (bond length, basis, active space) rung.

    ``n_frozen_core`` is 2 everywhere on the valence ladder, so every rung
    correlates the same ten electrons and the energies are comparable along it.
    Setting it to 0 gives the all-electron rung; see
    :data:`ALL_ELECTRON_ORBITALS` for why that rung exists.
    """

    bond_length: float
    basis: str = "cc-pvdz"
    n_active_orbitals: int = 8
    orbital_order: str = "canonical"  # or "fiedler"
    n_frozen_core: int = N2_FROZEN_CORE
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_electrons(self) -> int:
        """Electrons correlated in this active space."""
        return N2_TOTAL_ELECTRONS - 2 * self.n_frozen_core

    @property
    def geometry(self) -> Geometry:
        return diatomic("N", self.bond_length, label=f"N2_R{self.bond_length:.3f}")

    @property
    def hamiltonian(self) -> HamiltonianSpec:
        return HamiltonianSpec(
            basis=self.basis,
            n_frozen_core=self.n_frozen_core,
            n_active_orbitals=self.n_active_orbitals,
            point_group_symmetry=False,
            orbital_source=f"RHF canonical, {self.orbital_order} ordering",
        )

    @property
    def n_determinants(self) -> int:
        return determinant_count(self.n_active_orbitals, self.n_electrons)

    @property
    def label(self) -> str:
        return f"{self.n_active_orbitals}o"


def build_active_space(
    point: LadderPoint,
    *,
    n_threads: int = 4,
    scf_threads: int = SCF_OMP_THREADS,
) -> tuple[Any, ActiveSpaceHamiltonian, list[int]]:
    """RHF, then the one integral set both solvers consume, in the chosen order.

    Returns ``(mean_field, hamiltonian, order)``. The order is returned even for
    the canonical case, where it is the identity, so callers never have to
    branch on which variant they asked for.

    The SCF runs at ``scf_threads``; see :data:`SCF_OMP_THREADS` for why that is
    1 and why it is not a performance decision. The integral digest is the
    check -- stable across runs at one thread, and not at 32.
    """
    from pyscf import lib

    ham = point.hamiltonian
    with lib.with_omp_threads(scf_threads):
        mf = build_mean_field(point.geometry, ham)
        asham = active_space_hamiltonian(mf, ham)
    order = list(range(asham.n_orbitals))
    if point.orbital_order == "fiedler":
        order = fiedler_ordering(asham, n_threads=n_threads)
        asham = reorder_active_space(asham, order)
    elif point.orbital_order != "canonical":
        raise ValueError(f"unknown orbital_order {point.orbital_order!r}")
    return mf, asham, order


def _base_fields(point: LadderPoint, mf, asham: ActiveSpaceHamiltonian) -> dict[str, Any]:
    geometry = point.geometry
    return {
        "system_label": geometry.label,
        "basis": point.basis,
        "atoms": [[symbol, *position] for symbol, position in geometry.atoms],
        "geometry_parameters": {
            "bond_length": point.bond_length,
            "n_active_orbitals": float(point.n_active_orbitals),
        },
        "coordinate_unit": geometry.unit,
        "charge": geometry.charge,
        "spin_two_s": geometry.spin,
        "n_frozen_core": point.n_frozen_core,
        "n_orbitals": asham.n_orbitals,
        "n_electrons": asham.n_electrons,
        "point_group_symmetry": False,
        "reference_type": ReferenceType.RHF,
        "e_nuclear_repulsion": asham.e_nuclear_repulsion,
    }


def _space_fields(point: LadderPoint, asham, order) -> dict[str, Any]:
    return {
        "active_electrons": point.n_electrons,
        "active_orbitals": point.n_active_orbitals,
        "n_determinants": point.n_determinants,
        "orbital_order": point.orbital_order,
        "orbital_permutation": list(order),
        # Two records sharing this digest were computed from the same
        # Hamiltonian; two that do not were not, however equal their inputs
        # look. See SCF_OMP_THREADS.
        "integral_digest": asham.digest(),
        "e_core_constant": asham.e_core_constant,
        "scf_omp_threads": SCF_OMP_THREADS,
    }


def record_key(point: LadderPoint, method: str) -> str:
    """The store key a record for ``(point, method)`` will have.

    Lets a generator consult its cache before paying for the calculation rather
    than after. It reproduces :attr:`CalculationRecord.key` by building the same
    record the run would build, minus the results, so there is one definition of
    the key and not two that can drift apart.
    """
    geometry = point.geometry
    return CalculationRecord(
        system_label=geometry.label,
        method=method,
        basis=point.basis,
        atoms=[[symbol, *position] for symbol, position in geometry.atoms],
        coordinate_unit=geometry.unit,
        charge=geometry.charge,
        spin_two_s=geometry.spin,
        n_frozen_core=point.n_frozen_core,
        status=Status.NOT_CONVERGED,
    ).key


def run_exact(
    point: LadderPoint,
    mf,
    asham: ActiveSpaceHamiltonian,
    order: list[int],
    *,
    scf_walltime: float = 0.0,
    budget_bytes: int = DAVIDSON_BUDGET_BYTES,
    max_space: int | None = None,
) -> list[CalculationRecord]:
    """Exact diagonalisation inside the active space, where it is affordable.

    This *is* CASCI on RHF canonical orbitals, computed through the same
    integral arrays that DMRG will consume, so "the same Hamiltonian" is
    evidenced by a shared digest rather than assumed from two libraries having
    been asked for the same thing.

    Returns an empty list when the space does not fit the memory budget -- the
    honest outcome above the wall, and not a failure record, because nothing
    was attempted.

    ``max_space`` overrides the budget calculation. It exists so the reference
    can be run at two subspace sizes and the *reference's own* uncertainty
    measured, rather than assumed to be zero: at sixteen orbitals the subspace
    is 28 rather than the project's usual 200, and two builds of the same rung
    have already been seen 3e-10 apart. An agreement quoted tighter than the
    reference's own spread is not an agreement. The subspace size enters the
    method label whenever it is not the budgeted one, so the two runs are
    different points and cannot share a cache key.
    """
    if max_space is None:
        max_space = davidson_max_space(
            asham.n_orbitals, asham.n_electrons, asham.spin, budget_bytes=budget_bytes
        )
        budgeted = True
    else:
        budgeted = False
    if max_space is None:
        return []

    method = f"CASCI({point.n_electrons}e,{point.n_active_orbitals}o)"
    if not budgeted:
        method += f"[max_space={max_space}]"
    common = {**_base_fields(point, mf, asham)}
    space = _space_fields(point, asham, order)
    start = time.perf_counter() - scf_walltime
    try:
        e_total, fci_walltime = run_fci(asham, max_space=max_space)
    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
        return [
            CalculationRecord(
                method=method,
                e_total=None,
                status=Status.FAILED,
                walltime_seconds=time.perf_counter() - start,
                error_message=f"{type(exc).__name__}: {exc}",
                extra={**space, "davidson_max_space": max_space},
                **common,
            )
        ]
    return [
        CalculationRecord(
            method=method,
            e_total=float(e_total),
            e_electronic=float(e_total) - asham.e_nuclear_repulsion,
            status=Status.CONVERGED,
            walltime_seconds=time.perf_counter() - start,
            extra={
                **space,
                "davidson_max_space": max_space,
                "solver_walltime_seconds": fci_walltime,
                "note": "exact within the active space; not exact for the basis",
            },
            **common,
        )
    ]


def dmrg_method_label(point: LadderPoint, schedule: DMRGSchedule) -> str:
    """Method label carrying everything that makes two runs different points.

    :attr:`CalculationRecord.key` is built from the system, method, basis,
    geometry and frozen core -- deliberately not from the bond dimension, seed
    or orbital ordering, which are not properties of the *system*. So they have
    to travel in the method label, or a chi sweep would collapse onto one cache
    key and silently keep only its first point. Same convention as
    ``scripts/chi_convergence.py``.
    """
    label = (
        f"DMRG({point.n_electrons}e,{point.n_active_orbitals}o,"
        f"{point.orbital_order},chi={max(schedule.bond_dims)},seed={schedule.seed}"
    )
    if schedule.cutoff > 1e-20:
        label += f",cut={schedule.cutoff:g}"
    return label + ")"


def run_dmrg_rung(
    point: LadderPoint,
    mf,
    asham: ActiveSpaceHamiltonian,
    order: list[int],
    schedule: DMRGSchedule,
    *,
    scf_walltime: float = 0.0,
    e_exact: float | None = None,
    profile: bool = True,
    stack_mem_gb: int = DEFAULT_STACK_MEM_GB,
) -> CalculationRecord:
    """One DMRG calculation on the rung, with its convergence diagnostics.

    ``e_exact`` fills ``fci_discrepancy`` where a reference exists. Above the
    wall it stays ``None``: recording a discrepancy against a reference that was
    never computed would be worse than leaving the field empty.

    With ``profile`` the converged MPS is inspected before its scratch space is
    torn down, which is the only way to learn the bond dimensions it *actually*
    carries -- block2's ``sweep_bond_dims`` echoes the schedule it was given and
    is therefore a copy of the input, not a measurement.
    """
    chi = max(schedule.bond_dims)
    method = dmrg_method_label(point, schedule)
    common = {**_base_fields(point, mf, asham)}
    space = _space_fields(point, asham, order)
    captured: dict[str, Any] = {}
    hook = None
    if profile:
        from tn_quantum_chemistry.entanglement import symmetry_resolved_profile

        def hook(driver, ket):  # noqa: ANN001
            captured["profile"] = symmetry_resolved_profile(
                driver, ket, n_orbitals=asham.n_orbitals
            )

    start = time.perf_counter() - scf_walltime
    try:
        e_total, stats, dmrg_walltime = run_dmrg(
            asham, schedule, post_process=hook, stack_mem_gb=stack_mem_gb
        )
    except Exception as exc:  # noqa: BLE001
        return CalculationRecord(
            method=method,
            e_total=None,
            status=Status.FAILED,
            walltime_seconds=time.perf_counter() - start,
            error_message=f"{type(exc).__name__}: {exc}",
            extra={
                **space,
                "requested_bond_dim": chi,
                "seed": schedule.seed,
                "stack_mem_gb": stack_mem_gb,
            },
            **common,
        )

    measured: dict[str, Any] = {}
    if "profile" in captured:
        summary = captured["profile"].to_dict()
        entanglement = summary["entanglement"]
        measured = {
            # Counted in *states*; block2's chi counts SU(2) multiplets, so this
            # runs roughly 2x the requested cap when the cap is what binds.
            "effective_bond_dim": max(summary["bond_dimensions"]),
            "bond_dimensions": summary["bond_dimensions"],
        }
        space = {
            **space,
            "bipartite_entanglement": entanglement,
            "max_bipartite_entanglement": max(entanglement),
        }

    record = CalculationRecord(
        method=method,
        e_total=float(e_total),
        e_electronic=float(e_total) - asham.e_nuclear_repulsion,
        status=Status.CONVERGED,
        walltime_seconds=time.perf_counter() - start,
        dmrg={
            # The requested cap and the schedule, kept separate from anything
            # measured: block2's sweep_bond_dims echoes the request, so it is
            # never evidence about what the MPS did.
            "requested_bond_dim": chi,
            "schedule_bond_dims": list(schedule.bond_dims),
            "n_sweeps": schedule.n_sweeps,
            "cutoff": schedule.cutoff,
            "noises": list(schedule.noises),
            "davidson_thresholds": list(schedule.davidson_thresholds),
            "energy_tol": schedule.energy_tol,
            "seed": schedule.seed,
            "n_threads": schedule.n_threads,
            **measured,
            "final_discarded_weight": stats["sweep_discarded_weights"][-1],
            "n_sweeps_run": len(stats["sweep_energies"]),
            "sweep_energies": stats["sweep_energies"],
            "sweep_discarded_weights": stats["sweep_discarded_weights"],
            "solver_walltime_seconds": dmrg_walltime,
        },
        extra={
            **space,
            "requested_bond_dim": chi,
            "seed": schedule.seed,
            "stack_mem_gb": stack_mem_gb,
        },
        **common,
    )
    if e_exact is not None and np.isfinite(e_exact):
        record.fci_discrepancy = float(e_total - e_exact)
    return record


# --------------------------------------------------------------------------
# Reading the ladder back, and putting an uncertainty on it
# --------------------------------------------------------------------------
#
# These live here rather than in the script that first needed them because two
# analyses now depend on them -- the out-of-active-space correlation study and
# the external-reference check -- and they must agree. A duplicated copy of the
# uncertainty recipe would let the two drift, which is the same failure that let
# the sector profiles ship with a different solver threshold from the curve they
# were compared against.

#: What each rung of the ladder actually *is*, in the basis. Not cosmetic: the
#: 26- and 28-orbital rungs are exact solutions of a Hamiltonian, and saying so
#: is what makes them comparable to a published number.
RUNG_MEANING = {
    (10, 20): "intermediate valence space",
    (10, 26): "frozen-core FCI in cc-pVDZ",
    (14, 28): "full CI in cc-pVDZ",
}


def load_ladder(paths: list[Path]) -> dict[tuple, list[dict]]:
    """DMRG runs keyed by (bond length, active electrons, active orbitals).

    Accepts several stores. Phase 2 is partitioned by geometry across concurrent
    jobs -- block2 gains only 1.75x from 4 to 16 threads, so three 8-thread jobs
    beat one wide one -- and each job owns its own file, because two writers on
    one append-only store is a hazard this project has already had a near miss
    with. Reading them together here removes any need to merge them.
    """
    runs: dict[tuple, list[dict]] = defaultdict(list)
    for record in (r for path in paths for r in RecordStore(path)):
        if record.status != Status.CONVERGED or not record.method.startswith("DMRG"):
            continue
        extra = record.extra
        key = (
            round(record.geometry_parameters["bond_length"], 6),
            int(extra["active_electrons"]),
            int(extra["active_orbitals"]),
        )
        runs[key].append(
            {
                "e_total": record.e_total,
                # How far the energy was still moving on the final sweep. At
                # these rungs the sweep count binds before the energy tolerance
                # does, so a run can stop while still descending; ignoring that
                # would charge residual sweeping to bond-dimension truncation.
                "sweep_drift": (
                    abs(record.dmrg["sweep_energies"][-1]
                        - record.dmrg["sweep_energies"][-2])
                    if len(record.dmrg.get("sweep_energies") or []) >= 2
                    else None
                ),
                "sweeps_run": record.dmrg.get("n_sweeps_run"),
                "sweeps_planned": record.dmrg.get("n_sweeps"),
                "chi": int(extra["requested_bond_dim"]),
                "seed": int(extra["seed"]),
                "ordering": extra["orbital_order"],
                "dw": record.dmrg["final_discarded_weight"],
                "walltime": record.walltime_seconds,
            }
        )
    return dict(runs)


def summarise_rung(runs: list[dict]) -> dict[str, Any]:
    """Best variational energy at a rung, and an honest uncertainty on it.

    The uncertainty is the largest of three things that can be measured without
    a reference, because each catches a different way of being wrong:

    * the last step in the chi sweep -- residual truncation;
    * the disagreement between orbital orderings at the largest chi -- the only
      diagnostic that caught the local-minimum failure in Phase 1, where the
      discarded weight and the seed spread both reported success on a state
      0.19 Ha too high;
    * the seed spread at the largest chi -- a sweep landing in different minima;
    * the residual sweep drift -- how far the energy was still falling when the
      schedule ran out, which at these rungs is not negligible.

    None of them bounds the true error. They bound the spread of what was tried,
    which is a weaker and honest claim.
    """
    best = min(runs, key=lambda r: r["e_total"])
    chis = sorted({r["chi"] for r in runs})
    top = [r for r in runs if r["chi"] == chis[-1]]

    chi_step = None
    if len(chis) >= 2:
        previous = [r for r in runs if r["chi"] == chis[-2]]
        chi_step = abs(
            min(r["e_total"] for r in top) - min(r["e_total"] for r in previous)
        )

    by_ordering = defaultdict(list)
    for run in top:
        by_ordering[run["ordering"]].append(run["e_total"])
    ordering_gap = None
    if len(by_ordering) >= 2:
        bests = [min(v) for v in by_ordering.values()]
        ordering_gap = max(bests) - min(bests)

    by_seed = defaultdict(list)
    for run in top:
        by_seed[(run["ordering"], run["seed"])].append(run["e_total"])
    seed_spread = None
    same_ordering = [v for k, v in by_seed.items() if k[0] == best["ordering"]]
    if len(same_ordering) >= 2:
        values = [min(v) for v in same_ordering]
        seed_spread = max(values) - min(values)

    drifts = [r["sweep_drift"] for r in top if r["sweep_drift"] is not None]
    sweep_drift = max(drifts) if drifts else None
    saturated = sum(
        1 for r in top
        if r["sweeps_run"] is not None and r["sweeps_run"] >= (r["sweeps_planned"] or 0)
    )

    candidates = [
        x for x in (chi_step, ordering_gap, seed_spread, sweep_drift) if x is not None
    ]
    return {
        "sweep_drift": sweep_drift,
        "sweeps_saturated_at_chi_max": saturated,
        "e_best": best["e_total"],
        "chi_best": best["chi"],
        "ordering_best": best["ordering"],
        "chi_max": chis[-1],
        "chi_grid": chis,
        "chi_step": chi_step,
        "ordering_gap": ordering_gap,
        "seed_spread": seed_spread,
        "uncertainty": max(candidates) if candidates else None,
        "n_runs": len(runs),
        "total_walltime_s": sum(r["walltime"] for r in runs),
    }
