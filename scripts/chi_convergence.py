"""How the DMRG energy error decays with bond dimension, along the H10 curve.

Sweeps ``chi`` at each chain spacing and records the error against the FCI
reference for the same integrals, together with the bipartite entanglement of
the converged MPS. Three MPS seeds are run at every point.

The question is whether the variational error is a smooth function of ``chi``
with a geometry-dependent rate -- the finite-system analogue of the condensed
matter expectation that the energy gradient falls off with bond dimension -- or
whether low-``chi`` runs are simply landing in different local minima. Those two
explanations predict different things and the seed spread separates them:

* an *entanglement* effect is reproducible across seeds, and the fitted decay
  rate should track the bipartite entropy of the geometry;
* a *trapping* effect shows up as a large seed spread that does not shrink
  smoothly with ``chi``.

The existing sector-profile store cannot distinguish them: it holds one seed.

FCI is computed once per geometry and reused across every ``chi`` and seed, so
the cost of the sweep is the DMRG runs alone.

    uv run python scripts/chi_convergence.py
    uv run python scripts/chi_convergence.py --chis 16 32 64 --seeds 1234
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from tn_quantum_chemistry.entanglement import symmetry_resolved_profile
from tn_quantum_chemistry.schema import CalculationRecord, RecordStore, ReferenceType
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

CHI_GRID = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
SPACINGS = (1.0, 1.4, 1.8, 2.2, 2.6, 3.0)
SEEDS = (1234, 5678, 9012)


def build_record(
    geometry: Geometry,
    ham: HamiltonianSpec,
    asham,
    e_hf: float,
    e_fci: float,
    bond_dim: int,
    seed: int,
    threads: int,
    cutoff: float,
) -> CalculationRecord:
    """One (geometry, chi, seed) point, reusing the geometry's FCI reference.

    The cutoff enters the method label whenever it is active, because it changes
    the calculation: two runs at the same nominal ``chi`` with and without it are
    different points and must not share a cache key.
    """
    method = f"DMRG(chi={bond_dim},seed={seed})"
    if cutoff > 1e-20:
        method = f"DMRG(chi={bond_dim},cut={cutoff:g},seed={seed})"
    common = {
        "system_label": geometry.label,
        "method": method,
        "basis": ham.basis,
        "atoms": [[symbol, *position] for symbol, position in geometry.atoms],
        "geometry_parameters": geometry.parameters,
        "coordinate_unit": geometry.unit,
        "charge": geometry.charge,
        "spin_two_s": geometry.spin,
        "reference_type": ReferenceType.RHF,
        "n_frozen_core": ham.n_frozen_core,
    }
    try:
        schedule = DMRGSchedule.fixed_bond_dimension(
            bond_dim, seed=seed, n_threads=threads, cutoff=cutoff
        )
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

        # The full sector spectra are ~18 kB per record and this sweep has
        # hundreds of points; only the bipartite entropy per cut is kept.
        profile = captured["profile"].to_dict()
        entanglement = profile["entanglement"]
        # The bond dimensions the MPS actually carries. ``sweep_bond_dims`` only
        # echoes the requested schedule, so it is identical with and without a
        # cutoff and cannot report which limit bound. Note these count states
        # while block2's chi counts SU(2) multiplets, so they run roughly 2x the
        # requested value when the cap is what binds.
        effective = max(profile["bond_dimensions"])

        return CalculationRecord(
            n_orbitals=asham.n_orbitals,
            n_electrons=asham.n_electrons,
            e_total=e_dmrg,
            e_nuclear_repulsion=asham.e_nuclear_repulsion,
            e_correlation=e_dmrg - e_hf,
            fci_discrepancy=e_dmrg - e_fci,
            walltime_seconds=total_time,
            dmrg={
                "requested_bond_dim": bond_dim,
                "max_bond_dim": max(stats["sweep_bond_dims"]),
                "effective_bond_dim": effective,
                "cutoff": cutoff,
                "final_discarded_weight": stats["sweep_discarded_weights"][-1],
                "n_sweeps_run": len(stats["sweep_energies"]),
                "seed": seed,
                "n_threads": threads,
                "dmrg_walltime_seconds": walltime,
                **stats,
            },
            extra={
                "integral_digest": asham.digest(),
                "e_hf": e_hf,
                "e_fci": e_fci,
                "bipartite_entanglement": entanglement,
                "max_bipartite_entanglement": max(entanglement),
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
    parser.add_argument("--outdir", type=Path, default=Path("data/raw/chi_convergence"))
    parser.add_argument("--n-atoms", type=int, default=10)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--chis", type=int, nargs="+", default=list(CHI_GRID))
    parser.add_argument("--spacings", type=float, nargs="+", default=list(SPACINGS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--cutoff", type=float, default=1e-20,
        help="singular-value cutoff; the effective bond dimension is set by "
             "whichever of this and --chis binds first (default: block2's 1e-20, "
             "which never binds here and leaves --chis in sole control)",
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)

    ham = HamiltonianSpec(basis=args.basis)
    store = RecordStore(args.outdir / f"h{args.n_atoms}_{args.basis}.jsonl")
    cached = store.completed_keys()

    n_points = len(args.spacings) * len(args.chis) * len(args.seeds)
    print(f"{n_points} points ({len(args.spacings)} R x {len(args.chis)} chi "
          f"x {len(args.seeds)} seeds) -> {store.path}\n")

    n_written = n_skipped = n_failed = 0
    for R in args.spacings:
        geometry = linear_hydrogen_chain(
            args.n_atoms, R=R, delta=0.0, label=f"H{args.n_atoms}_R{R:g}_d0"
        )
        mf = build_mean_field(geometry, ham)
        asham = active_space_hamiltonian(mf, ham)
        t0 = time.perf_counter()
        e_fci, _ = run_fci(asham)
        print(f"R = {R:.1f}  E_FCI = {e_fci:.12f}  ({time.perf_counter() - t0:.1f}s)")
        header = (f"  {'chi':>5} {'eff':>5} {'seed':>6} {'|dFCI| (Ha)':>13} "
                  f"{'S_max':>7} {'dw':>10} {'wall':>7}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for bond_dim in args.chis:
            for seed in args.seeds:
                record = build_record(
                    geometry, ham, asham, float(mf.e_tot), e_fci,
                    bond_dim, seed, args.threads, args.cutoff,
                )
                if record.key in cached and not args.replace:
                    n_skipped += 1
                    continue
                store.append(record, replace=args.replace)
                if record.converged:
                    n_written += 1
                    print(f"  {bond_dim:>5} {record.dmrg['effective_bond_dim']:>5} {seed:>6} "
                          f"{abs(record.fci_discrepancy):>13.3e} "
                          f"{record.extra['max_bipartite_entanglement']:>7.4f} "
                          f"{record.dmrg['final_discarded_weight']:>10.2e} "
                          f"{record.walltime_seconds:>6.1f}s")
                else:
                    n_failed += 1
                    print(f"  {bond_dim:>5} {'-':>5} {seed:>6} {'FAILED':>13}  "
                          f"{record.error_message}")
        print()

    print(f"{n_written} written, {n_skipped} cached, {n_failed} failed")
    print(f"{store.path}: {store.path.stat().st_size / 1e6:.2f} MB")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
