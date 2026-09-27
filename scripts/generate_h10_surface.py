"""Experiment 4: the two-dimensional H10 ``(R, delta)`` dataset.

A rectangular grid over nearest-neighbour spacing ``R`` and alternating
displacement ``delta``, with every theory tier at each point, written to a
cached record store. This is the dataset the surrogate study consumes; it is
deliberately modest, because controlled evaluation matters more than density.

Grid choices, both settled by the pilot rather than assumed:

**Cost.** The full ladder plus DMRG at chi = 64 and 500 costs about 11 s per
geometry, so a grid of order a hundred points is a quarter of an hour. FCI
dominates at large ``R`` (about 10 s at R = 2.6 A) while DMRG stays near 5 s.

**delta must span both signs.** H10 has ten atoms and therefore *nine* bonds --
an odd number -- so ``+delta`` and ``-delta`` are not the same system. With
``delta < 0`` the chain has five short bonds and forms five H2 molecules; with
``delta > 0`` it has four short bonds, giving four H2 and two unpaired terminal
atoms. The fully paired arrangement is markedly lower in energy: 210 mHa at
R = 1.0 A, 50 mHa at 1.8 A, 4 mHa at 2.6 A. Sampling only ``delta >= 0`` would
cover half the surface and omit the more stable half. The symmetry that would
excuse it holds for rings, and for chains with an even number of bonds, but not
here.

Run with::

    uv run python scripts/generate_h10_surface.py
    uv run python scripts/generate_h10_surface.py --dry-run
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tn_quantum_chemistry.methods import (
    methods_below_fci,
    reference_convergence_floor,
    run_all_methods,
)
from tn_quantum_chemistry.schema import RecordStore, Status, hardware_id
from tn_quantum_chemistry.validation import HamiltonianSpec, linear_hydrogen_chain

#: Spacings in Angstrom: from the compressed, weakly correlated chain through
#: the coupled-cluster breakdown to the near-dissociated limit.
DEFAULT_SPACINGS = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0)

#: Alternating displacements in Angstrom, this project's ``R +/- delta/2``
#: convention. Both signs, for the reason in the module docstring.
DEFAULT_DIMERISATIONS = (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6)

#: DMRG sequence: a cheap tier that is still a function of the geometry, and the
#: FCI-validated setting. Below chi = 64 the energy varies more than tenfold
#: with the random MPS seed on this system.
DEFAULT_BOND_DIMS = (64, 500)


def valid_geometry(spacing: float, dimerisation: float) -> bool:
    """Both alternating bond lengths must stay positive."""
    return abs(dimerisation) < 2.0 * spacing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("data/raw/h10_surface"))
    parser.add_argument("--n-atoms", type=int, default=10)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--spacings", type=float, nargs="+", default=list(DEFAULT_SPACINGS))
    parser.add_argument(
        "--dimerisations", type=float, nargs="+", default=list(DEFAULT_DIMERISATIONS)
    )
    parser.add_argument("--bond-dims", type=int, nargs="+", default=list(DEFAULT_BOND_DIMS))
    parser.add_argument("--no-dmrg", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report the grid and estimated cost without calculating",
    )
    parser.add_argument(
        "--seconds-per-point", type=float, default=11.0,
        help="pilot-measured cost, used only for the estimate",
    )
    args = parser.parse_args(argv)

    points = [
        (R, d)
        for R in args.spacings
        for d in args.dimerisations
        if valid_geometry(R, d)
    ]
    rejected = len(args.spacings) * len(args.dimerisations) - len(points)

    print(f"H{args.n_atoms}/{args.basis} surface")
    print(f"  R      : {len(args.spacings)} values, "
          f"{min(args.spacings):g} to {max(args.spacings):g} A")
    print(f"  delta  : {len(args.dimerisations)} values, "
          f"{min(args.dimerisations):+g} to {max(args.dimerisations):+g} A")
    print(f"  points : {len(points)}" + (f"  ({rejected} rejected as unphysical)"
                                          if rejected else ""))
    print("  methods: ladder + FCI"
          + ("" if args.no_dmrg else f" + DMRG {tuple(args.bond_dims)}"))
    estimate = len(points) * args.seconds_per_point
    print(f"  cost   : ~{estimate / 60:.0f} min at {args.seconds_per_point:g} s/point")
    print(f"  hardware: {hardware_id()}")
    if args.dry_run:
        return 0

    ham = HamiltonianSpec(basis=args.basis)
    store = RecordStore(args.outdir / f"h{args.n_atoms}_{args.basis}_surface.jsonl")
    cached = store.completed_keys()
    print(f"  store  : {store.path}\n")

    header = (f"{'R':>5} {'delta':>6} {'E_FCI':>15} {'E-E(d=0)':>10} "
              f"{'flags':>24} {'wall':>7} {'eta':>6}")
    print(header)
    print("-" * len(header))

    n_written = n_skipped = n_failed = 0
    baseline: dict[float, float] = {}
    started = time.perf_counter()
    for index, (spacing, dimerisation) in enumerate(points, start=1):
        geometry = linear_hydrogen_chain(
            args.n_atoms, R=spacing, delta=dimerisation,
            label=f"H{args.n_atoms}_R{spacing:g}_d{dimerisation:+g}",
        )
        point_start = time.perf_counter()
        records = run_all_methods(
            geometry, ham, dmrg_bond_dims=tuple(args.bond_dims),
            with_dmrg=not args.no_dmrg, seed=args.seed, n_threads=args.threads,
        )
        elapsed = time.perf_counter() - point_start

        new = [r for r in records if r.key not in cached or args.replace]
        for record in new:
            store.append(record, replace=args.replace)
        cached |= {r.key for r in records}
        n_written += len(new)
        n_skipped += len(records) - len(new)

        by_method = {r.method: r for r in records}
        e_fci = by_method["FCI"].e_total if "FCI" in by_method else None
        if dimerisation == 0.0 and e_fci is not None:
            baseline[spacing] = e_fci

        flags = []
        if methods_below_fci(records):
            flags.append(f"below-FCI:{len(methods_below_fci(records))}")
        if any(r.status == Status.NOT_CONVERGED for r in records):
            flags.append("unconverged")
        failed = [r for r in records if r.status == Status.FAILED]
        n_failed += len(failed)
        if failed:
            flags.append(f"FAILED:{len(failed)}")
        floor = reference_convergence_floor(records)
        if floor > 0.0:
            flags.append(f"fci-floor:{floor:.0e}")

        relative = (
            "" if e_fci is None or spacing not in baseline
            else f"{e_fci - baseline[spacing]:10.2e}"
        )
        remaining = (time.perf_counter() - started) / index * (len(points) - index)
        print(f"{spacing:>5g} {dimerisation:>+6.2f} "
              f"{'n/a' if e_fci is None else f'{e_fci:15.9f}'} {relative:>10} "
              f"{','.join(flags) or '-':>24} {elapsed:>6.1f}s {remaining / 60:>5.1f}m")

    print(f"\n{n_written} records written, {n_skipped} cached, {n_failed} failed")
    print(f"elapsed {(time.perf_counter() - started) / 60:.1f} min")

    fci = [
        (r.geometry_parameters["R"], r.geometry_parameters["delta"], r.e_total)
        for r in store
        if r.method == "FCI" and r.converged
    ]
    if fci:
        energies = np.array([e for _, _, e in fci])
        print(f"FCI surface: {len(fci)} points, "
              f"{energies.min():.9f} to {energies.max():.9f} Ha")
    print(f"{store.path}: {store.path.stat().st_size / 1e6:.2f} MB")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
