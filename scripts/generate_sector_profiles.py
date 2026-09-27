"""Generate H10 energies together with their symmetry-resolved MPS structure.

Writes one :class:`CalculationRecord` per geometry carrying the DMRG energy, the
usual convergence metadata, and the full ``(N, 2S, pg)``-labelled Schmidt
spectrum of the converged MPS. The spectrum is free once DMRG has run and cannot
be recovered afterwards without re-running the solver, so it is persisted rather
than summarised away.

``--bond-dim`` selects the fidelity tier and is recorded in the method label, so
a cheap and an expensive sweep over the same geometries coexist in one store
without colliding:

    uv run python scripts/generate_sector_profiles.py --bond-dim 50
    uv run python scripts/generate_sector_profiles.py --bond-dim 500 --with-fci

Completed geometries are skipped on re-run. A geometry that fails is written
with its status and error message rather than dropped.

Choosing the low-fidelity bond dimension
----------------------------------------
Too small a ``chi`` makes the low-fidelity tier *non-reproducible* rather than
merely inaccurate: DMRG lands in different local minima depending on the random
MPS initialisation. Measured on H10/STO-3G at R = 2.6 A, the geometry where this
is worst, over five MPS seeds:

    chi =  32   error 2.0e-2 .. 2.6e-1   (12.8x spread)   0.7 s
    chi =  64   error 1.4e-2 .. 1.7e-2   ( 1.2x spread)   2.5 s
    chi = 128   error 1.0e-4             ( 1.0x spread)   9.2 s

Below chi = 64 the energy is not a function of the geometry at all, so any
correction learned from it is partly fitting seed noise. chi = 64 is the useful
tier here: the error is large enough to be worth correcting and reproducible
across seeds. chi = 32 records are retained in the store as a record of the
measurement, but should not be used as a fidelity tier.
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from tn_quantum_chemistry.entanglement import symmetry_resolved_profile
from tn_quantum_chemistry.schema import (
    CalculationRecord,
    RecordStore,
    ReferenceType,
)
from tn_quantum_chemistry.validation import (
    DMRGSchedule,
    Geometry,
    HamiltonianSpec,
    active_space_hamiltonian,
    build_mean_field,
    linear_hydrogen_chain,
    run_dmrg,
    run_fci,
)


def geometry_grid(
    n_atoms: int, spacings: list[float], dimerisations: list[float]
) -> list[Geometry]:
    return [
        linear_hydrogen_chain(
            n_atoms, R=R, delta=delta, label=f"H{n_atoms}_R{R:g}_d{delta:g}"
        )
        for R in spacings
        for delta in dimerisations
    ]


def schedule_for(bond_dim: int, seed: int, threads: int) -> DMRGSchedule:
    """The project's fixed-chi schedule; see :meth:`DMRGSchedule.fixed_bond_dimension`."""
    return DMRGSchedule.fixed_bond_dimension(bond_dim, seed=seed, n_threads=threads)


def build_record(
    geometry: Geometry, ham: HamiltonianSpec, bond_dim: int,
    schedule: DMRGSchedule, *, with_fci: bool,
) -> CalculationRecord:
    method = f"DMRG(chi={bond_dim})"
    atoms = [[symbol, *position] for symbol, position in geometry.atoms]
    common = {
        "system_label": geometry.label,
        "method": method,
        "basis": ham.basis,
        "atoms": atoms,
        "geometry_parameters": geometry.parameters,
        "coordinate_unit": geometry.unit,
        "charge": geometry.charge,
        "spin_two_s": geometry.spin,
        "reference_type": ReferenceType.RHF,
        "n_frozen_core": ham.n_frozen_core,
    }

    try:
        mf = build_mean_field(geometry, ham)
        asham = active_space_hamiltonian(mf, ham)

        e_fci = None
        if with_fci:
            e_fci, _ = run_fci(asham)

        captured: dict = {}
        start = time.perf_counter()
        e_dmrg, stats, walltime = run_dmrg(
            asham,
            schedule,
            post_process=lambda driver, ket: captured.update(
                profile=symmetry_resolved_profile(
                    driver, ket, n_orbitals=asham.n_orbitals
                )
            ),
        )
        total_time = time.perf_counter() - start
        profile = captured["profile"]

        return CalculationRecord(
            n_orbitals=asham.n_orbitals,
            n_electrons=asham.n_electrons,
            e_total=e_dmrg,
            e_nuclear_repulsion=asham.e_nuclear_repulsion,
            e_correlation=e_dmrg - float(mf.e_tot),
            fci_discrepancy=None if e_fci is None else e_dmrg - e_fci,
            walltime_seconds=total_time,
            dmrg={
                "requested_bond_dim": bond_dim,
                "max_bond_dim": max(stats["sweep_bond_dims"]),
                "final_discarded_weight": stats["sweep_discarded_weights"][-1],
                "n_sweeps_run": len(stats["sweep_energies"]),
                "seed": schedule.seed,
                "n_threads": schedule.n_threads,
                "dmrg_walltime_seconds": walltime,
                **stats,
            },
            sector_profile=profile.to_dict(),
            extra={
                "integral_digest": asham.digest(),
                "e_hf": float(mf.e_tot),
                "e_fci": e_fci,
            },
            **common,
        )
    except Exception as exc:  # noqa: BLE001 - a failed point stays in the table
        traceback.print_exc()
        return CalculationRecord.failed(
            error_message=f"{type(exc).__name__}: {exc}", **common
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("data/raw/sector_profiles"))
    parser.add_argument("--n-atoms", type=int, default=10)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--bond-dim", type=int, default=500)
    parser.add_argument(
        "--spacings", type=float, nargs="+",
        default=[1.0, 1.4, 1.8, 2.2, 2.6, 3.0],
        help="nearest-neighbour spacings R in Angstrom",
    )
    parser.add_argument(
        "--dimerisations", type=float, nargs="+", default=[0.0],
        help="alternating displacements delta in this project's R +/- delta/2 convention",
    )
    parser.add_argument("--with-fci", action="store_true",
                        help="also compute FCI and record the DMRG discrepancy")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--replace", action="store_true",
                        help="recompute and overwrite cached geometries")
    args = parser.parse_args(argv)

    ham = HamiltonianSpec(basis=args.basis)
    schedule = schedule_for(args.bond_dim, args.seed, args.threads)
    store = RecordStore(args.outdir / f"h{args.n_atoms}_{args.basis}.jsonl")
    cached = store.completed_keys()

    geometries = geometry_grid(args.n_atoms, args.spacings, args.dimerisations)
    print(f"{len(geometries)} geometries at chi={args.bond_dim} -> {store.path}\n")

    header = f"{'geometry':<22} {'E_total':>16} {'dFCI':>10} {'S_mid':>7} {'wall':>7}"
    print(header)
    print("-" * len(header))

    n_written = n_skipped = n_failed = 0
    for geometry in geometries:
        record = build_record(
            geometry, ham, args.bond_dim, schedule, with_fci=args.with_fci
        )
        if record.key in cached and not args.replace:
            n_skipped += 1
            print(f"{geometry.label:<22} {'(cached)':>16}")
            continue

        store.append(record, replace=args.replace)
        if record.converged:
            n_written += 1
            profile = record.sector_profile
            middle = profile["entanglement"][len(profile["entanglement"]) // 2]
            discrepancy = (
                "n/a" if record.fci_discrepancy is None
                else f"{record.fci_discrepancy:.2e}"
            )
            print(f"{geometry.label:<22} {record.e_total:>16.12f} {discrepancy:>10} "
                  f"{middle:>7.4f} {record.walltime_seconds:>6.1f}s")
        else:
            n_failed += 1
            print(f"{geometry.label:<22} {'FAILED':>16}  {record.error_message}")

    floats = sum(
        len(values)
        for record in store
        if record.sector_profile
        for cut in record.sector_profile["sector_spectra"]
        for values in cut.values()
    )
    print(f"\n{n_written} written, {n_skipped} cached, {n_failed} failed")
    print(f"{store.path}: {store.path.stat().st_size / 1e6:.2f} MB, "
          f"{floats} stored Schmidt values")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
