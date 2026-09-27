"""How much bond dimension a hydrogen chain actually needs, as it gets longer.

The question behind this is whether tensor networks earn their place on this
system class. At H10 they do not: converged DMRG costs 5.4 s against FCI's
2.4 s, so the tensor network certifies labels it could not have produced more
cheaply. The usual argument is that this reverses with chain length, because a
gapped one-dimensional system obeys an area law -- its entanglement saturates,
so the bond dimension needed for fixed accuracy stops growing while the
determinant count explodes combinatorially.

That argument is geometry-dependent and it is measured here rather than assumed.
A first attempt at fixed chi = 512 and R = 1.8 A found the opposite -- DMRG got
45x slower and stopped converging between H10 and H12 -- but R = 1.8 A sits near
the metal-insulator crossover where entanglement peaks, which is precisely where
no saturation should be expected. The spacing is therefore a controlled variable
here, not a fixed choice:

* ``R = 1.0`` compressed and metallic-like, where correlation is dynamic and the
  Schmidt spectrum has a long tail (measured at H10: the *least* entangled
  geometry and the most expensive);
* ``R = 1.8`` the crossover, where the entanglement is largest;
* ``R = 3.0`` stretched toward a gapped spin chain, where the area law should
  bite and the bond dimension should saturate with length.

**The instrument is a singular-value cutoff, not a chi scan.** With a large cap
and an active cutoff, the effective bond dimension is chosen by the state rather
than imposed on it, so ``chi_eff(N, R)`` is read off directly instead of being
bisected for. That is the quantity the area-law question is about. The cap is
still recorded, because a run that hits it has not converged and its chi_eff is
a floor rather than a measurement.

FCI is computed where it fits in memory (N <= 14; H16 needs 40 GB at
``max_space`` 30 and more than this machine has at the 200 the reference
actually requires) so the DMRG numbers are anchored to an exact answer over part
of the range.

    uv run python scripts/chain_length_scaling.py
    uv run python scripts/chain_length_scaling.py --lengths 10 14 --spacings 3.0
"""

from __future__ import annotations

import argparse
import time
import traceback
from math import comb
from pathlib import Path

from tn_quantum_chemistry.entanglement import symmetry_resolved_profile
from tn_quantum_chemistry.schema import CalculationRecord, RecordStore, ReferenceType
from tn_quantum_chemistry.validation import (
    DMRGSchedule,
    HamiltonianSpec,
    active_space_hamiltonian,
    build_mean_field,
    linear_hydrogen_chain,
    run_dmrg,
    run_fci,
)

LENGTHS = (10, 14, 20, 26, 32)
SPACINGS = (1.0, 1.8, 3.0)
DIMERISATIONS = (0.0, 0.4)
CUTOFFS = (1e-6, 1e-8, 1e-10)
BOND_CAP = 1500
N_SWEEPS = 20

#: Largest chain whose FCI fits in memory here. H16 needs 1.3 GB per Davidson
#: vector and the reference needs max_space=200 to converge at stretched
#: spacings; H18 needs 19 GB per vector.
FCI_MAX_LENGTH = 14
FCI_MAX_SPACE = 200


def ramped(cap: int, cutoff: float, seed: int, threads: int) -> DMRGSchedule:
    """The project's ramped schedule; see :meth:`DMRGSchedule.ramped`."""
    return DMRGSchedule.ramped(
        cap, cutoff=cutoff, seed=seed, n_threads=threads, n_sweeps=N_SWEEPS
    )


def build_record(n_atoms, R, delta, ham, cutoff, cap, seed, threads, e_fci):
    geometry = linear_hydrogen_chain(
        n_atoms, R=R, delta=delta, label=f"H{n_atoms}_R{R:g}_d{delta:g}"
    )
    common = {
        "system_label": geometry.label,
        "method": f"DMRG(cap={cap},cut={cutoff:g})",
        "basis": ham.basis,
        "atoms": [[s, *p] for s, p in geometry.atoms],
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
        captured: dict = {}
        start = time.perf_counter()
        e_dmrg, stats, walltime = run_dmrg(
            asham,
            ramped(cap, cutoff, seed, threads),
            post_process=lambda driver, ket: captured.update(
                profile=symmetry_resolved_profile(
                    driver, ket, n_orbitals=asham.n_orbitals
                )
            ),
        )
        total = time.perf_counter() - start
        profile = captured["profile"].to_dict()
        entanglement = profile["entanglement"]
        chi_eff = max(profile["bond_dimensions"])
        return CalculationRecord(
            n_orbitals=asham.n_orbitals,
            n_electrons=asham.n_electrons,
            e_total=e_dmrg,
            e_nuclear_repulsion=asham.e_nuclear_repulsion,
            e_correlation=e_dmrg - float(mf.e_tot),
            fci_discrepancy=None if e_fci is None else e_dmrg - e_fci,
            walltime_seconds=total,
            dmrg={
                "requested_bond_dim": cap,
                "cutoff": cutoff,
                "effective_bond_dim": chi_eff,
                # Which limit bound. Not readable from sweep_bond_dims, which
                # echoes the requested schedule and therefore always equals the
                # cap; the honest test is where the discarded weight landed. A
                # weight at the cutoff means the cutoff chose the basis and the
                # run is converged to that tolerance; a weight orders of
                # magnitude above it means the cap ran out first and chi_eff is
                # a floor, not a measurement.
                "cap_bound": bool(stats["sweep_discarded_weights"][-1] > 100 * cutoff),
                "max_bond_dim": max(stats["sweep_bond_dims"]),
                "final_discarded_weight": stats["sweep_discarded_weights"][-1],
                "n_sweeps_run": len(stats["sweep_energies"]),
                "seed": seed,
                "n_threads": threads,
                "dmrg_walltime_seconds": walltime,
                **stats,
            },
            extra={
                "integral_digest": asham.digest(),
                "e_hf": float(mf.e_tot),
                "e_fci": e_fci,
                "n_atoms": n_atoms,
                "n_determinants": comb(asham.n_orbitals, asham.n_electrons // 2) ** 2,
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
    parser.add_argument("--outdir", type=Path, default=Path("data/raw/chain_scaling"))
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    parser.add_argument("--spacings", type=float, nargs="+", default=list(SPACINGS))
    parser.add_argument(
        "--dimerisations", type=float, nargs="+", default=list(DIMERISATIONS),
        help="delta opens a spin-Peierls gap; the uniform chain (delta=0) is "
             "critical at large R and should NOT show a saturating bond dimension",
    )
    parser.add_argument("--cutoffs", type=float, nargs="+", default=list(CUTOFFS))
    parser.add_argument("--cap", type=int, default=BOND_CAP)
    parser.add_argument(
        "--fci-max-length", type=int, default=FCI_MAX_LENGTH,
        help="anchor to FCI up to this chain length (memory-bound; see module docstring)",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)

    ham = HamiltonianSpec(basis=args.basis)
    store = RecordStore(args.outdir / f"h_chain_{args.basis}.jsonl")
    cached = store.completed_keys()

    print(f"{len(args.lengths)} lengths x {len(args.spacings)} spacings x "
          f"{len(args.cutoffs)} cutoffs -> {store.path}\n")
    header = (f"{'N':>4} {'R':>5} {'delta':>5} {'cutoff':>8} {'chi_eff':>8} {'bound':>5} "
              f"{'S_max':>7} {'dw':>9} {'wall s':>8} {'dFCI':>10}")
    print(header); print("-" * len(header))

    n_written = n_skipped = n_failed = 0
    for n_atoms in args.lengths:
      for delta in args.dimerisations:
        for R in args.spacings:
            e_fci = None
            if n_atoms <= args.fci_max_length:
                geometry = linear_hydrogen_chain(n_atoms, R=R, delta=delta)
                asham = active_space_hamiltonian(build_mean_field(geometry, ham), ham)
                t0 = time.perf_counter()
                e_fci, _ = run_fci(asham, max_space=FCI_MAX_SPACE)
                print(f"  [FCI H{n_atoms} R={R:g} d={delta:g}: {e_fci:.12f} in "
                      f"{time.perf_counter() - t0:.1f}s]", flush=True)
            for cutoff in args.cutoffs:
                record = build_record(
                    n_atoms, R, delta, ham, cutoff, args.cap, args.seed,
                    args.threads, e_fci
                )
                if record.key in cached and not args.replace:
                    n_skipped += 1
                    continue
                store.append(record, replace=args.replace)
                if record.converged:
                    n_written += 1
                    d = record.dmrg
                    dfci = ("n/a" if record.fci_discrepancy is None
                            else f"{abs(record.fci_discrepancy):.1e}")
                    print(f"{n_atoms:>4} {R:>5.1f} {delta:>5.1f} {cutoff:>8.0e} "
                          f"{d['effective_bond_dim']:>8} {'CAP' if d['cap_bound'] else 'cut':>5} "
                          f"{record.extra['max_bipartite_entanglement']:>7.3f} "
                          f"{d['final_discarded_weight']:>9.1e} "
                          f"{record.walltime_seconds:>8.1f} {dfci:>10}", flush=True)
                else:
                    n_failed += 1
                    print(f"{n_atoms:>4} {R:>5.1f} {delta:>5.1f} {cutoff:>8.0e}  FAILED  "
                          f"{record.error_message}", flush=True)

    print(f"\n{n_written} written, {n_skipped} cached, {n_failed} failed")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
