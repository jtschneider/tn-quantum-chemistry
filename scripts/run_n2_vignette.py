"""Day 6: the N2 chemistry vignette, three bond lengths in two bases.

Runs RHF (with its stability analysis), a stabilised UHF, MP2/CCSD/CCSD(T)
with the core frozen, CASCI and CASSCF in the full-valence CAS(10e,8o),
NEVPT2 on top of the converged CASSCF, and -- where the basis is small enough
to afford it -- all-electron FCI.

Results are cached per (geometry, method, basis) in a record store, so a
re-run recomputes nothing and a failure keeps its row.

Run with::

    uv run python scripts/run_n2_vignette.py
    uv run python scripts/run_n2_vignette.py --dry-run
    uv run python scripts/run_n2_vignette.py --basis sto-3g
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tn_quantum_chemistry.schema import RecordStore, Status
from tn_quantum_chemistry.vignette import (
    DEFAULT_BASES,
    DEFAULT_BOND_LENGTHS,
    VignetteSpec,
    run_vignette_point,
)

DEFAULT_STORE = Path("data/raw/n2_vignette/n2_vignette.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--basis", action="append", default=None)
    parser.add_argument("--bond-length", type=float, action="append", default=None)
    parser.add_argument("--no-nevpt2", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bases = args.basis or list(DEFAULT_BASES)
    lengths = args.bond_length or list(DEFAULT_BOND_LENGTHS)
    specs = [
        VignetteSpec(bond_length=r, basis=b, run_nevpt2=not args.no_nevpt2)
        for b in bases
        for r in lengths
    ]
    print(f"{len(specs)} points: {bases} x {lengths}")
    if args.dry_run:
        return 0

    args.store.parent.mkdir(parents=True, exist_ok=True)
    store = RecordStore(args.store)
    already = store.completed_keys()
    print(f"{len(already)} records already cached")

    started = time.perf_counter()
    for index, spec in enumerate(specs, start=1):
        point_start = time.perf_counter()
        records = run_vignette_point(spec)
        written = sum(store.append(record, replace=True) for record in records)
        elapsed = time.perf_counter() - point_start
        statuses = {
            record.method: record.status for record in records
        }
        unconverged = [m for m, s in statuses.items() if s != Status.CONVERGED]
        print(
            f"[{index}/{len(specs)}] N2 R={spec.bond_length:.3f} A / {spec.basis}: "
            f"{written} records, {elapsed:.1f} s"
            + (f"  [not converged: {', '.join(unconverged)}]" if unconverged else ""),
            flush=True,
        )

    print(f"\ntotal {(time.perf_counter() - started) / 60:.1f} min -> {args.store}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
