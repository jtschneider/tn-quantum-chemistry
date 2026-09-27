"""The N2 vignette table, built only from cached records.

Writes ``figures/n2_vignette.csv`` and prints the same numbers as Markdown for
the README. Runs no calculation.

Run with::

    uv run python scripts/make_vignette_table.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from tn_quantum_chemistry.schema import HARTREE_TO_KCAL_PER_MOL, RecordStore, Status

DEFAULT_STORE = Path("data/raw/n2_vignette/n2_vignette.jsonl")
DEFAULT_OUT = Path("figures/n2_vignette.csv")

METHOD_ORDER = (
    "RHF", "UHF", "MP2", "CCSD", "CCSD(T)",
    "CASCI(10e,8o)", "CASSCF(10e,8o)", "NEVPT2/CASSCF(10e,8o)", "FCI",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    by_point: dict[tuple[str, float], dict[str, object]] = defaultdict(dict)
    for record in RecordStore(args.store):
        key = (record.basis, float(record.geometry_parameters["bond_length"]))
        by_point[key][record.method] = record

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "basis", "bond_length_angstrom", "method", "reference_type",
            "n_frozen_core", "status", "e_total_hartree",
            "error_vs_fci_kcal_per_mol", "spin_squared",
            "nevpt2_correction_hartree", "active_natural_occupations",
            "walltime_seconds",
        ])
        for (basis, length) in sorted(by_point):
            for method in METHOD_ORDER:
                record = by_point[(basis, length)].get(method)
                if record is None:
                    continue
                extra = record.extra or {}
                writer.writerow([
                    basis, f"{length:.3f}", method, str(record.reference_type),
                    record.n_frozen_core, str(record.status),
                    f"{record.e_total:.8f}",
                    "" if record.fci_discrepancy is None
                    else f"{record.fci_discrepancy * HARTREE_TO_KCAL_PER_MOL:.3f}",
                    "" if "spin_squared" not in extra
                    else f"{extra['spin_squared']:.6f}",
                    "" if "nevpt2_correction_hartree" not in extra
                    else f"{extra['nevpt2_correction_hartree']:.6f}",
                    "" if "active_natural_occupations" not in extra
                    else " ".join(f"{o:.3f}" for o in extra["active_natural_occupations"]),
                    f"{record.walltime_seconds:.2f}",
                ])
    print(f"wrote {args.out}")

    for basis in sorted({key[0] for key in by_point}):
        lengths = sorted(length for b, length in by_point if b == basis)
        print(f"\n### N2 / {basis}\n")
        header = " | ".join(f"{length:.3f} A" for length in lengths)
        print(f"| method | {header} |")
        print("| --- | " + " --- |" * len(lengths))
        for method in METHOD_ORDER:
            cells = []
            for length in lengths:
                record = by_point[(basis, length)].get(method)
                if record is None:
                    cells.append("--")
                    continue
                mark = "" if record.status == Status.CONVERGED else " ‡"
                cells.append(f"{record.e_total:.6f}{mark}")
            if any(cell != "--" for cell in cells):
                print(f"| {method} | " + " | ".join(cells) + " |")
        print("\n‡ amplitude equations did not converge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
