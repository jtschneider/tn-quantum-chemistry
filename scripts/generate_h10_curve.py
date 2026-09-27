"""Experiment 2: the H10 dissociation curve at delta = 0, every theory tier.

Runs RHF, MP2, CCSD, CCSD(T), FCI and a DMRG bond-dimension sequence over a
range of nearest-neighbour spacings, writing one record per (geometry, method)
to a cached JSON-lines store. Figures are produced separately from this data;
this script performs the calculations and nothing else.

Two properties of the curve drive the design:

* Coupled cluster fails, and fails in two distinct ways. Past roughly
  R = 1.6 A both CCSD and CCSD(T) fall *below* FCI -- for a fixed finite
  Hamiltonian FCI is exact, so a lower energy is a failure and not an
  improvement -- and past roughly R = 2.8 A the amplitude equations stop
  converging as well. Both are recorded per point; neither is dropped, and the
  figure must not plot an unconverged point as converged.
* The DMRG sequence starts at chi = 64. Below that the energy on this system is
  not a function of the geometry, varying more than tenfold with the random MPS
  seed, so smaller bond dimensions would put seed noise on the plot.

Run with::

    uv run python scripts/generate_h10_curve.py
    uv run python scripts/generate_h10_curve.py --spacings 1.0 1.2 --replace
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tn_quantum_chemistry.methods import (
    DMRG_SEQUENCE,
    methods_below_fci,
    reference_convergence_floor,
    run_all_methods,
)
from tn_quantum_chemistry.schema import RecordStore, Status, hardware_id
from tn_quantum_chemistry.validation import HamiltonianSpec, linear_hydrogen_chain

#: Default spacings in Angstrom. Chosen from a pilot: the range spans the
#: weakly correlated minimum through the coupled-cluster breakdown to the
#: near-dissociated chain, and both failure modes are clearly visible inside it.
DEFAULT_SPACINGS = (
    0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("data/raw/h10_curve"))
    parser.add_argument("--n-atoms", type=int, default=10)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--spacings", type=float, nargs="+", default=list(DEFAULT_SPACINGS))
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument(
        "--bond-dims", type=int, nargs="+", default=list(DMRG_SEQUENCE),
        help="DMRG convergence sequence; the last entry should be the "
             "FCI-validated setting",
    )
    parser.add_argument("--no-dmrg", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)

    ham = HamiltonianSpec(basis=args.basis)
    store = RecordStore(args.outdir / f"h{args.n_atoms}_{args.basis}_curve.jsonl")
    cached = store.completed_keys()

    print(f"H{args.n_atoms}/{args.basis}, delta = {args.delta:g}, "
          f"{len(args.spacings)} spacings -> {store.path}")
    print(f"DMRG sequence: {'disabled' if args.no_dmrg else args.bond_dims}")
    print(f"hardware: {hardware_id()}\n")

    header = (f"{'R':>5} {'E_FCI':>15} {'HF':>10} {'MP2':>10} {'CCSD':>10} "
              f"{'CCSD(T)':>10} {'DMRG*':>10} {'flags':>22} {'wall':>7}")
    print(header)
    print("-" * len(header))

    n_written = n_skipped = n_failed = 0
    floors: dict[float, float] = {}
    for spacing in args.spacings:
        geometry = linear_hydrogen_chain(
            args.n_atoms, R=spacing, delta=args.delta,
            label=f"H{args.n_atoms}_R{spacing:g}_d{args.delta:g}",
        )
        start = time.perf_counter()
        records = run_all_methods(
            geometry, ham, dmrg_bond_dims=tuple(args.bond_dims),
            with_dmrg=not args.no_dmrg, seed=args.seed, n_threads=args.threads,
        )
        elapsed = time.perf_counter() - start

        new = [r for r in records if r.key not in cached or args.replace]
        for record in new:
            store.append(record, replace=args.replace)
        cached |= {r.key for r in records}

        by_method = {r.method: r for r in records}
        e_fci = by_method["FCI"].e_total if "FCI" in by_method else None

        def show(name: str, methods=by_method, reference=e_fci) -> str:
            record = methods.get(name)
            if record is None or record.e_total is None or reference is None:
                return "     n/a"
            marker = "" if record.converged else "!"
            return f"{record.e_total - reference:9.2e}{marker}"

        best_dmrg = f"DMRG(chi={args.bond_dims[-1]})"
        flags = []
        below = methods_below_fci(records)
        if below:
            flags.append(f"below-FCI:{len(below)}")
        unconverged = [r.method for r in records if r.status == Status.NOT_CONVERGED]
        if unconverged:
            flags.append(f"unconverged:{len(unconverged)}")
        failed = [r for r in records if r.status == Status.FAILED]
        n_failed += len(failed)
        if failed:
            flags.append(f"FAILED:{len(failed)}")
        floor = reference_convergence_floor(records)
        if floor > 0.0:
            flags.append(f"fci-floor:{floor:.0e}")
            floors[spacing] = floor
        n_written += len(new)
        n_skipped += len(records) - len(new)

        fci_text = "n/a" if e_fci is None else f"{e_fci:15.10f}"
        print(f"{spacing:>5g} {fci_text:>15} {show('RHF'):>10} {show('MP2'):>10} "
              f"{show('CCSD'):>10} {show('CCSD(T)'):>10} {show(best_dmrg):>10} "
              f"{','.join(flags) or '-':>22} {elapsed:>6.1f}s")

    print(f"\n{n_written} records written, {n_skipped} cached, {n_failed} failed")
    print("'!' marks a record whose method did not converge; "
          "a negative value lies below FCI and is a failure, not an improvement.")

    if floors:
        worst = max(floors.values())
        print(
            f"\nFCI reference convergence floor: up to {worst:.1e} Ha "
            f"(at R = {max(floors, key=floors.__getitem__):g} A).\n"
            "  DMRG is variational, so where it sits below FCI the reference is\n"
            "  what has not converged. No error smaller than this can be resolved\n"
            "  against FCI at those geometries."
        )

    energies = [r.e_total for r in store if r.method == "FCI" and r.converged]
    if energies:
        print(f"\nFCI curve: {len(energies)} points, "
              f"minimum {np.min(energies):.10f} Ha")
    print(f"{store.path}: {store.path.stat().st_size / 1e6:.2f} MB")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
