"""The electronic-structure method ladder, as reusable code.

One entry point per theory tier, each returning a
:class:`~tn_quantum_chemistry.schema.CalculationRecord` carrying the energy, its
convergence status, and its own wall time. Convergence and cost are first-class
outputs rather than console side effects, because Experiment 2 needs to plot the
first and report the second separately from accuracy.

A method that fails to converge, or raises, still produces a record. On the
stretched end of the H10 curve this is not hypothetical: coupled cluster stops
converging around R = 2.8 A and its energy falls *below* FCI well before that,
so "did this converge" is part of the result and not a detail.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np

from tn_quantum_chemistry.schema import (
    CalculationRecord,
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
    run_dmrg,
    run_fci,
)


def _patch_pyscf_numpy2_diis() -> None:
    """Restore ``numpy.linalg.linalg`` for PySCF's DIIS error handler.

    ``pyscf/lib/diis.py`` contains ``except numpy.linalg.linalg.LinAlgError``.
    NumPy 2 removed that alias, so when the DIIS extrapolation genuinely does
    hit a singular matrix -- which happens exactly where coupled cluster
    struggles -- PySCF's own handler raises ``AttributeError: module
    'numpy.linalg' has no attribute 'linalg'`` instead of the error it meant to
    catch.

    The shim does not rescue the calculation: PySCF logs and re-raises, so the
    CC run still fails. What it fixes is the *diagnosis*. Without it, a genuine
    numerical breakdown is reported as a missing module, which reads like a
    broken installation rather than the physics it actually is.

    ``numpy.linalg.linalg.LinAlgError`` and ``numpy.linalg.LinAlgError`` were
    the same class in NumPy 1, so aliasing the module to itself is exact.
    """
    import numpy.linalg

    if not hasattr(numpy.linalg, "linalg"):
        numpy.linalg.linalg = numpy.linalg


_patch_pyscf_numpy2_diis()

#: Single-reference tiers, in increasing order of correlation treatment.
SINGLE_REFERENCE_LADDER = ("RHF", "MP2", "CCSD", "CCSD(T)")

#: Bond dimensions for the Experiment 2 convergence sequence. It starts at 64
#: rather than 32 deliberately: below about 64 the DMRG energy on this system is
#: not a function of the geometry at all, varying by more than an order of
#: magnitude with the random MPS seed, so a smaller value would put seed noise
#: on the plot. The sequence ends at the FCI-validated setting.
DMRG_SEQUENCE = (64, 128, 256, 500)


def _base_fields(geometry: Geometry, ham: HamiltonianSpec) -> dict[str, Any]:
    return {
        "system_label": geometry.label,
        "basis": ham.basis,
        "atoms": [[symbol, *position] for symbol, position in geometry.atoms],
        "geometry_parameters": geometry.parameters,
        "coordinate_unit": geometry.unit,
        "charge": geometry.charge,
        "spin_two_s": geometry.spin,
        "reference_type": ReferenceType.RHF,
        "n_frozen_core": ham.n_frozen_core,
    }


def _record(
    geometry: Geometry, ham: HamiltonianSpec, method: str, **kwargs: Any
) -> CalculationRecord:
    return CalculationRecord(method=method, **_base_fields(geometry, ham), **kwargs)


def _failed(
    geometry: Geometry, ham: HamiltonianSpec, method: str, exc: BaseException,
    walltime: float,
) -> CalculationRecord:
    return CalculationRecord(
        method=method,
        status=Status.FAILED,
        error_message=f"{type(exc).__name__}: {exc}",
        walltime_seconds=walltime,
        **_base_fields(geometry, ham),
    )


# --------------------------------------------------------------------------- #
# Single-reference tiers
# --------------------------------------------------------------------------- #


def run_single_reference_ladder(
    geometry: Geometry,
    ham: HamiltonianSpec,
    *,
    mf=None,
    walltime_scf: float | None = None,
) -> list[CalculationRecord]:
    """RHF, MP2, CCSD and CCSD(T) on one geometry.

    Every tier's wall time includes the reference it is built on, so the numbers
    answer "what does this tier cost" rather than "what does this step cost".
    CCSD(T) likewise reuses the CCSD amplitudes and reports the cumulative cost.

    A caller that has already built the reference must pass ``walltime_scf``
    along with ``mf``. Without it the RHF timing would measure nothing at all --
    which is exactly the bug this signature exists to prevent.
    """
    from pyscf import cc, mp

    records: list[CalculationRecord] = []
    if mf is None:
        start = time.perf_counter()
        try:
            mf = build_mean_field(geometry, ham)
        except Exception as exc:  # noqa: BLE001 - a failed reference is a result
            return [_failed(geometry, ham, m, exc, time.perf_counter() - start)
                    for m in SINGLE_REFERENCE_LADDER]
        walltime_scf = time.perf_counter() - start
    elif walltime_scf is None:
        raise ValueError(
            "pass walltime_scf together with a pre-built mf, otherwise the RHF "
            "wall time is measured as zero"
        )
    e_hf = float(mf.e_tot)
    common = {
        "n_orbitals": int(mf.mo_coeff.shape[1]),
        "n_electrons": int(mf.mol.nelectron),
        "e_nuclear_repulsion": float(mf.mol.energy_nuc()),
    }

    records.append(
        _record(geometry, ham, "RHF", e_total=e_hf, e_correlation=0.0,
                walltime_seconds=walltime_scf, **common)
    )

    start = time.perf_counter()
    try:
        mp2 = mp.MP2(mf).run()
        records.append(
            _record(geometry, ham, "MP2", e_total=float(mp2.e_tot),
                    e_correlation=float(mp2.e_corr),
                    walltime_seconds=walltime_scf + time.perf_counter() - start,
                    **common)
        )
    except Exception as exc:  # noqa: BLE001
        records.append(_failed(geometry, ham, "MP2", exc, time.perf_counter() - start))

    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ccsd = cc.CCSD(mf)
            ccsd.max_cycle = 200
            ccsd.run()
        walltime_ccsd = walltime_scf + time.perf_counter() - start
        converged = bool(ccsd.converged)
        records.append(
            _record(
                geometry, ham, "CCSD", e_total=float(ccsd.e_tot),
                e_correlation=float(ccsd.e_corr),
                status=Status.CONVERGED if converged else Status.NOT_CONVERGED,
                error_message=None if converged
                else "CCSD amplitude equations did not converge",
                walltime_seconds=walltime_ccsd, **common,
            )
        )
    except np.linalg.LinAlgError as exc:
        # A singular DIIS matrix is coupled cluster failing, not the software.
        # Recorded as non-convergence with the real cause rather than as a
        # crash, so the surface keeps an honest row for the geometry.
        walltime_ccsd = walltime_scf + time.perf_counter() - start
        message = f"CCSD DIIS extrapolation became singular: {exc}"
        records.append(
            _record(geometry, ham, "CCSD", status=Status.NOT_CONVERGED,
                    error_message=message, walltime_seconds=walltime_ccsd, **common)
        )
        records.append(
            _record(geometry, ham, "CCSD(T)", status=Status.NOT_CONVERGED,
                    error_message=f"not attempted: {message}",
                    walltime_seconds=walltime_ccsd, **common)
        )
        return records
    except Exception as exc:  # noqa: BLE001
        records.append(_failed(geometry, ham, "CCSD", exc, time.perf_counter() - start))
        records.append(
            _failed(geometry, ham, "CCSD(T)", exc, time.perf_counter() - start)
        )
        return records

    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e_triples = float(ccsd.ccsd_t())
        records.append(
            _record(
                geometry, ham, "CCSD(T)", e_total=float(ccsd.e_tot) + e_triples,
                e_correlation=float(ccsd.e_corr) + e_triples,
                # The triples correction inherits the amplitudes' status: a
                # perturbative correction on unconverged amplitudes is not a
                # converged result no matter how cleanly it evaluates.
                status=Status.CONVERGED if converged else Status.NOT_CONVERGED,
                error_message=None if converged
                else "built on unconverged CCSD amplitudes",
                walltime_seconds=walltime_ccsd + time.perf_counter() - start,
                extra={"e_triples_correction": e_triples}, **common,
            )
        )
    except Exception as exc:  # noqa: BLE001
        records.append(
            _failed(geometry, ham, "CCSD(T)", exc, time.perf_counter() - start)
        )
    return records


# --------------------------------------------------------------------------- #
# Exact and near-exact tiers
# --------------------------------------------------------------------------- #


def run_fci_record(
    geometry: Geometry, ham: HamiltonianSpec, asham: ActiveSpaceHamiltonian
) -> CalculationRecord:
    start = time.perf_counter()
    try:
        e_fci, walltime = run_fci(asham)
    except Exception as exc:  # noqa: BLE001
        return _failed(geometry, ham, "FCI", exc, time.perf_counter() - start)
    return _record(
        geometry, ham, "FCI", e_total=e_fci, fci_discrepancy=0.0,
        walltime_seconds=walltime, n_orbitals=asham.n_orbitals,
        n_electrons=asham.n_electrons,
        e_nuclear_repulsion=asham.e_nuclear_repulsion,
        extra={"integral_digest": asham.digest()},
    )


def run_dmrg_sequence(
    geometry: Geometry,
    ham: HamiltonianSpec,
    asham: ActiveSpaceHamiltonian,
    *,
    bond_dims: tuple[int, ...] = DMRG_SEQUENCE,
    seed: int = 1234,
    n_threads: int = 4,
    with_profile: bool = True,
) -> list[CalculationRecord]:
    """DMRG at a sequence of bond dimensions, ending at the validated setting.

    Each entry is a separate record labelled ``DMRG(chi=...)``, so the sequence
    shows convergence with bond dimension rather than collapsing to a single
    number whose accuracy the reader has to take on trust.
    """
    from tn_quantum_chemistry.entanglement import (
        profile_invariants,
        spin_square_expectation,
        symmetry_resolved_profile,
    )

    records: list[CalculationRecord] = []
    for chi in bond_dims:
        n_sweeps = 20
        schedule = DMRGSchedule(
            bond_dims=[chi] * n_sweeps,
            noises=[1e-4] * 6 + [1e-6] * 4 + [0.0] * (n_sweeps - 10),
            # 1e-14 rather than 1e-12 on both the local eigensolver and the
            # sweep stopping rule. At R = 3.4 A the looser pair left chi = 500
            # 8.5e-10 Ha above the converged energy -- a systematic drift with
            # spacing, not noise, because the nearly dissociated chain has a
            # near-degenerate spectrum that a loosely converged Davidson stops
            # short of. Tightening costs 0.7 s per point (6.0 -> 6.7 s) and
            # brings the same chi to within 7e-13 Ha. The flat schedule stays
            # flat: each chi is published as its own record, so warming it up
            # from a larger bond dimension would make the low-chi tier a
            # statement about the ramp instead of about chi.
            davidson_thresholds=[1e-14] * n_sweeps,
            n_sweeps=n_sweeps,
            energy_tol=1e-14,
            seed=seed,
            n_threads=n_threads,
        )
        captured: dict[str, Any] = {}

        def capture(driver: Any, ket: Any, store: dict[str, Any] = captured) -> None:
            # ``store`` is bound as a default argument rather than captured from
            # the enclosing scope, so each iteration writes to its own dict.
            store["spin_squared"] = spin_square_expectation(driver, ket)
            profile = symmetry_resolved_profile(
                driver, ket, n_orbitals=asham.n_orbitals
            )
            store["profile"] = profile
            store["invariants"] = profile_invariants(profile)

        start = time.perf_counter()
        try:
            energy, stats, walltime = run_dmrg(
                asham, schedule, post_process=capture if with_profile else None
            )
        except Exception as exc:  # noqa: BLE001
            records.append(
                _failed(geometry, ham, f"DMRG(chi={chi})", exc,
                        time.perf_counter() - start)
            )
            continue

        profile = captured.get("profile")
        records.append(
            _record(
                geometry, ham, f"DMRG(chi={chi})", e_total=energy,
                walltime_seconds=walltime, n_orbitals=asham.n_orbitals,
                n_electrons=asham.n_electrons,
                e_nuclear_repulsion=asham.e_nuclear_repulsion,
                dmrg={
                    "requested_bond_dim": chi,
                    "max_bond_dim": max(stats["sweep_bond_dims"]),
                    "final_discarded_weight": stats["sweep_discarded_weights"][-1],
                    "n_sweeps_run": len(stats["sweep_energies"]),
                    "seed": seed,
                    "n_threads": n_threads,
                    **stats,
                },
                sector_profile=None if profile is None else profile.to_dict(),
                extra={
                    "integral_digest": asham.digest(),
                    "spin_squared": captured.get("spin_squared"),
                    **captured.get("invariants", {}),
                },
            )
        )
    return records


# --------------------------------------------------------------------------- #
# The full ladder
# --------------------------------------------------------------------------- #


def run_all_methods(
    geometry: Geometry,
    ham: HamiltonianSpec,
    *,
    dmrg_bond_dims: tuple[int, ...] = DMRG_SEQUENCE,
    with_fci: bool = True,
    with_dmrg: bool = True,
    seed: int = 1234,
    n_threads: int = 4,
) -> list[CalculationRecord]:
    """Every theory tier at one geometry, with FCI discrepancies filled in."""
    planned = [*SINGLE_REFERENCE_LADDER]
    if with_fci:
        planned.append("FCI")
    if with_dmrg:
        planned += [f"DMRG(chi={chi})" for chi in dmrg_bond_dims]

    # The reference is built once and shared by every tier. If it fails, no tier
    # can proceed, so all of them are recorded as failed with the same cause
    # rather than the geometry vanishing from the table.
    start = time.perf_counter()
    try:
        mf = build_mean_field(geometry, ham)
    except Exception as exc:  # noqa: BLE001
        walltime = time.perf_counter() - start
        return [_failed(geometry, ham, method, exc, walltime) for method in planned]
    walltime_scf = time.perf_counter() - start

    records = run_single_reference_ladder(
        geometry, ham, mf=mf, walltime_scf=walltime_scf
    )
    asham = active_space_hamiltonian(mf, ham)

    e_fci: float | None = None
    if with_fci:
        fci_record = run_fci_record(geometry, ham, asham)
        records.append(fci_record)
        e_fci = fci_record.e_total

    if with_dmrg:
        records.extend(
            run_dmrg_sequence(
                geometry, ham, asham, bond_dims=dmrg_bond_dims,
                seed=seed, n_threads=n_threads,
            )
        )

    if e_fci is not None:
        for record in records:
            if record.e_total is not None and record.method != "FCI":
                record.fci_discrepancy = record.e_total - e_fci
    return records


def energy_ladder(records: list[CalculationRecord]) -> dict[str, float]:
    """Convenience view: method -> total energy, converged records only."""
    return {
        record.method: record.e_total
        for record in records
        if record.converged and record.e_total is not None
    }


def ladder_is_monotone(records: list[CalculationRecord]) -> bool:
    """Whether RHF > MP2 > CCSD > CCSD(T) in energy.

    Note what this does *not* detect. The ladder stays monotone deep into the
    coupled-cluster breakdown -- on H10 at R = 2.8 A the energies still decrease
    in order while CCSD sits 0.18 Ha *below* FCI. Monotonicity is a sanity check
    on the wiring of the tiers, not a diagnostic of correctness. Use
    :func:`methods_below_fci` for that.
    """
    energies = energy_ladder(records)
    present = [m for m in SINGLE_REFERENCE_LADDER if m in energies]
    values = [energies[m] for m in present]
    return bool(np.all(np.greater(values[:-1], values[1:])))


#: Methods that are variational for the exact ground state of the same finite
#: Hamiltonian. They can never lie genuinely below FCI, so when they do, the
#: reference is the thing that has not converged.
VARIATIONAL_PREFIXES = ("DMRG",)


def is_variational(method: str) -> bool:
    return method.startswith(VARIATIONAL_PREFIXES)


def reference_convergence_floor(
    records: list[CalculationRecord], tolerance: float = 1e-12
) -> float:
    """Empirical lower bound on the FCI reference's own convergence error.

    DMRG is variational for the exact ground state, so ``E_DMRG >= E_FCI`` must
    hold for the same Hamiltonian. Any amount by which a converged DMRG run sits
    *below* FCI is therefore not a DMRG error but a measure of how far the FCI
    Davidson is from the true eigenvalue.

    This matters for the error panel of the dissociation curve: on H10/STO-3G it
    is 3e-12 Ha at worst, at R = 3.4 A, where the near-degenerate manifold of a
    nearly dissociated chain makes the reference hardest to converge. No error
    smaller than this can be resolved against FCI at that geometry, whatever the
    method.

    It used to be ~2e-9 Ha there, and the cause was solver settings on *both*
    sides rather than anything physical. The floor grew smoothly with spacing --
    a systematic drift, not scatter -- because PySCF's Davidson ran with a
    12-vector subspace and the DMRG local eigensolver with a 1e-12 threshold,
    and both stop short in a near-degenerate spectrum. With FCI at conv_tol
    1e-14/max_space 30 and DMRG at 1e-14, the discrepancy at R = 3.4 A fell from
    1.8e-9 to 1.7e-12 Ha, and both energies moved *down* -- confirming that each
    had been above the true ground state rather than disagreeing with each
    other. Bond dimension was never the constraint: chi = 500, 1000 and 1500
    agree to 1e-14 there.
    """
    e_fci = next(
        (r.e_total for r in records if r.method == "FCI" and r.converged), None
    )
    if e_fci is None:
        return 0.0
    violations = [
        e_fci - record.e_total
        for record in records
        if is_variational(record.method)
        and record.converged
        and record.e_total is not None
        and record.e_total < e_fci - tolerance
    ]
    return float(max(violations)) if violations else 0.0


def methods_below_fci(
    records: list[CalculationRecord], tolerance: float = 1e-9
) -> dict[str, float]:
    """Methods whose energy lies below FCI for the same finite Hamiltonian.

    FCI is exact in this space, so any method beneath it has failed rather than
    improved. This is the signature of the coupled-cluster breakdown on the
    stretched end of the H10 curve, and it is the diagnostic that actually
    fires: the ladder remains monotone throughout.
    """
    e_fci = next(
        (r.e_total for r in records if r.method == "FCI" and r.converged), None
    )
    if e_fci is None:
        return {}
    # Deliberately not restricted to converged records: a method that both fails
    # to converge and lands below FCI is the clearest form of the pathology, and
    # filtering on convergence would hide exactly those points.
    #
    # Variational methods are excluded. When DMRG dips below FCI the cause is
    # the reference's own convergence, not a breakdown of the method, and
    # reporting it here would conflate a 1e-9 numerical floor with a 1e-1 Ha
    # failure of coupled cluster. See :func:`reference_convergence_floor`.
    return {
        record.method: record.e_total - e_fci
        for record in records
        if record.method != "FCI"
        and not is_variational(record.method)
        and record.e_total is not None
        and record.e_total < e_fci - tolerance
    }
