"""Day-1 smoke test: every required package executes once and produces a versioned result.

Runs the whole stack on a single system -- H2O in STO-3G -- so the checks are
directly comparable rather than a scatter of unrelated toy problems:

    RHF -> MP2 -> CCSD -> CCSD(T) -> FCI -> DMRG        (PySCF + block2)
    GFN2-xTB on the same geometry                        (tblite)
    -grad(E) against central differences                 (JAX)

MP2 and CCSD(T) are not in the Day-1 list but are required by Experiment 2 and
cost nothing here, so they are exercised now rather than discovered broken on
Day 3.

Every electronic-structure check is written as a
:class:`~tn_quantum_chemistry.schema.CalculationRecord`. A check that raises is
recorded with ``status=failed`` and its error message; it is never dropped.

Run with::

    uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from tn_quantum_chemistry.methods import run_all_methods
from tn_quantum_chemistry.schema import (
    CalculationRecord,
    RecordStore,
    ReferenceType,
    hardware_id,
    software_provenance,
)
from tn_quantum_chemistry.validation import (
    DMRGSchedule,
    Geometry,
    HamiltonianSpec,
    active_space_hamiltonian,
    build_mean_field,
    run_dmrg,
    water,
)


def _record(
    geometry: Geometry, method: str, basis: str, **kwargs: Any
) -> CalculationRecord:
    return CalculationRecord(
        system_label=geometry.label,
        method=method,
        basis=basis,
        atoms=[[s, *p] for s, p in geometry.atoms],
        geometry_parameters=geometry.parameters,
        coordinate_unit=geometry.unit,
        charge=geometry.charge,
        spin_two_s=geometry.spin,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# PySCF: RHF, MP2, CCSD, CCSD(T), FCI
# --------------------------------------------------------------------------- #


def smoke_pyscf(geometry: Geometry, ham: HamiltonianSpec) -> list[CalculationRecord]:
    """Run the single-reference ladder plus FCI on one Hamiltonian.

    Delegates to :mod:`tn_quantum_chemistry.methods` so the smoke test and the
    production curve exercise exactly the same code.
    """
    return run_all_methods(geometry, ham, with_dmrg=False)


# --------------------------------------------------------------------------- #
# block2: DMRG
# --------------------------------------------------------------------------- #


def smoke_dmrg(
    geometry: Geometry, ham: HamiltonianSpec, schedule: DMRGSchedule, e_fci: float | None
) -> CalculationRecord:
    mf = build_mean_field(geometry, ham)
    asham = active_space_hamiltonian(mf, ham)
    e_dmrg, stats, walltime = run_dmrg(asham, schedule)
    return _record(
        geometry, "DMRG", basis=ham.basis, reference_type=ReferenceType.RHF,
        n_frozen_core=ham.n_frozen_core, n_orbitals=asham.n_orbitals,
        n_electrons=asham.n_electrons, e_nuclear_repulsion=asham.e_nuclear_repulsion,
        e_total=e_dmrg,
        fci_discrepancy=None if e_fci is None else e_dmrg - e_fci,
        walltime_seconds=walltime,
        dmrg={
            "max_bond_dim": max(stats["sweep_bond_dims"]),
            "final_discarded_weight": stats["sweep_discarded_weights"][-1],
            "n_sweeps_run": len(stats["sweep_energies"]),
            "seed": schedule.seed,
            "n_threads": schedule.n_threads,
            **stats,
        },
        extra={"integral_digest": asham.digest()},
    )


# --------------------------------------------------------------------------- #
# tblite: GFN2-xTB
# --------------------------------------------------------------------------- #


def smoke_xtb(geometry: Geometry) -> CalculationRecord:
    """GFN2-xTB single point on the same geometry.

    The energy is a semiempirical total energy with its own reference and is
    NOT comparable to the ab initio numbers above. It is recorded here only to
    prove the binding runs and to pin the method, version and units.
    """
    from tblite.interface import Calculator

    from tn_quantum_chemistry.schema import BOHR_TO_ANGSTROM

    symbol_to_z = {"H": 1, "O": 8, "N": 7, "C": 6}
    numbers = np.array([symbol_to_z[s] for s, _ in geometry.atoms])
    # tblite takes Bohr; the geometry is stored in Angstrom.
    positions = np.array([p for _, p in geometry.atoms]) / BOHR_TO_ANGSTROM

    start = time.perf_counter()
    calc = Calculator("GFN2-xTB", numbers, positions)
    calc.set("verbosity", 0)
    result = calc.singlepoint()
    energy = float(result.get("energy"))

    return _record(
        geometry, "GFN2-xTB", basis="none (semiempirical, tblite parameterisation)",
        reference_type=ReferenceType.NONE, e_total=energy,
        walltime_seconds=time.perf_counter() - start,
        extra={
            "comparable_to_ab_initio": False,
            "note": "semiempirical total energy; different reference, not comparable "
                    "to RHF/CCSD/FCI totals on this system",
            "input_position_unit": "Bohr",
        },
    )


# --------------------------------------------------------------------------- #
# JAX: reverse-mode gradient against central differences
# --------------------------------------------------------------------------- #


def smoke_jax(tolerance: float = 1e-7) -> dict[str, Any]:
    """Check ``-jax.grad(E)`` against central differences on a Morse potential.

    This is the exact pattern Experiment 5 uses for generalized forces, so the
    smoke test exercises the real capability rather than a toy identity.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    D_e, alpha, r_e = 0.1745, 1.0271, 1.4011  # H2-like, atomic units

    def energy(r):
        return D_e * (1.0 - jnp.exp(-alpha * (r - r_e))) ** 2

    force = jax.jit(jax.grad(lambda r: -energy(r)))

    h = 1e-5
    points = jnp.array([1.0, 1.4011, 2.0, 3.0])
    analytic = np.array([float(force(r)) for r in points])
    numeric = np.array(
        [float(-(energy(r + h) - energy(r - h)) / (2 * h)) for r in points]
    )
    max_abs_err = float(np.max(np.abs(analytic - numeric)))

    return {
        "check": "jax_grad_vs_central_difference",
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "test_function": "Morse potential, D_e=0.1745, alpha=1.0271, r_e=1.4011 (a.u.)",
        "finite_difference_step": h,
        "points": [float(r) for r in points],
        "analytic_force": analytic.tolist(),
        "numeric_force": numeric.tolist(),
        "max_abs_error": max_abs_err,
        "tolerance": tolerance,
        "passed": bool(max_abs_err < tolerance),
    }


# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Day-1 environment smoke test.")
    parser.add_argument("--outdir", type=Path, default=Path("data/raw/smoke"))
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)

    geometry = water()
    ham = HamiltonianSpec(basis=args.basis)
    schedule = DMRGSchedule.default(seed=args.seed, n_threads=args.threads)

    args.outdir.mkdir(parents=True, exist_ok=True)
    store = RecordStore(args.outdir / "records.jsonl")
    checks: dict[str, Any] = {}
    records: list[CalculationRecord] = []

    def guard(name: str, fn):
        print(f"--- {name}", flush=True)
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - a broken package must be recorded, not fatal
            checks[name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            return None

    pyscf_records = guard("pyscf", lambda: smoke_pyscf(geometry, ham)) or []
    records.extend(pyscf_records)
    if pyscf_records:
        checks["pyscf"] = {"passed": all(r.converged for r in pyscf_records),
                           "methods": [r.method for r in pyscf_records]}

    e_fci = next((r.e_total for r in pyscf_records if r.method == "FCI"), None)
    dmrg_record = guard("block2", lambda: smoke_dmrg(geometry, ham, schedule, e_fci))
    if dmrg_record is not None:
        records.append(dmrg_record)
        checks["block2"] = {
            "passed": dmrg_record.converged,
            "fci_discrepancy": dmrg_record.fci_discrepancy,
        }

    xtb_record = guard("tblite", lambda: smoke_xtb(geometry))
    if xtb_record is not None:
        records.append(xtb_record)
        checks["tblite"] = {"passed": True, "method": "GFN2-xTB",
                            "energy_hartree": xtb_record.e_total}

    jax_check = guard("jax", smoke_jax)
    if jax_check is not None:
        checks["jax"] = jax_check

    for record in records:
        store.append(record, replace=True)

    all_passed = bool(checks) and all(c.get("passed") for c in checks.values())
    report = {
        "report": "day1_smoke_test",
        "system": geometry.label,
        "basis": args.basis,
        "all_passed": all_passed,
        "checks": checks,
        "hardware_id": hardware_id(),
        "software": software_provenance(),
    }
    (args.outdir / "environment.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 68)
    for record in records:
        marker = "ok  " if record.converged else "FAIL"
        energy = "n/a" if record.e_total is None else f"{record.e_total:18.12f}"
        disc = "" if record.fci_discrepancy is None else f"  dFCI {record.fci_discrepancy:+.3e}"
        print(f" [{marker}] {record.method:<10} {energy} Ha"
              f"  {record.walltime_seconds:6.2f}s{disc}")
    if "jax" in checks and "max_abs_error" in checks["jax"]:
        j = checks["jax"]
        print(f" [{'ok  ' if j['passed'] else 'FAIL'}] {'JAX grad':<10} "
              f"max |analytic - FD| = {j['max_abs_error']:.3e} "
              f"(tol {j['tolerance']:.0e}, {j['backend']})")
    print("=" * 68)
    print(f"records     -> {args.outdir / 'records.jsonl'}")
    print(f"environment -> {args.outdir / 'environment.json'}")
    print(f"\nDAY-1 SMOKE TEST: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
