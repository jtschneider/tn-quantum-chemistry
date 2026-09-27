"""Phase 1 summary: what it costs DMRG to reproduce the exact answer, per rung.

Reads ``data/raw/n2_ladder/`` and writes ``figures/n2_ladder.csv`` plus the same
numbers as Markdown. The headline quantity is ``chi*`` -- the smallest bond
dimension whose energy is within ``--tolerance`` of exact diagonalisation in the
same active space -- reported separately for each orbital ordering, because the
ordering is the variable that turned out to matter.

    uv run python scripts/make_n2_ladder_table.py
    uv run python scripts/make_n2_ladder_table.py --tolerance 1e-10
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from tn_quantum_chemistry.schema import RecordStore, Status

#: Chemical accuracy is 1.6e-3 Ha; this is six orders tighter. The point of
#: Phase 1 is not chemistry, it is establishing that the two solvers agree to
#: the numerical floor, so the threshold is set where "agreement" stops being
#: interesting rather than where it stops mattering.
DEFAULT_TOLERANCE = 1e-9


def load(path: Path) -> tuple[dict, dict]:
    """Exact energies and DMRG runs, keyed by (basis, bond length, orbitals)."""
    exact: dict[tuple, dict] = {}
    dmrg: dict[tuple, list[dict]] = defaultdict(list)
    for record in RecordStore(path):
        if record.status != Status.CONVERGED:
            continue
        key = (
            record.basis,
            round(record.geometry_parameters["bond_length"], 6),
            int(record.extra["active_orbitals"]),
        )
        if record.method.startswith("CASCI"):
            exact[key] = {
                "e_total": record.e_total,
                "walltime": record.walltime_seconds,
                "max_space": record.extra.get("davidson_max_space"),
                "digest": record.extra.get("integral_digest"),
                "n_determinants": record.extra.get("n_determinants"),
            }
        elif record.method.startswith("DMRG"):
            dmrg[key].append(
                {
                    "n_determinants": record.extra.get("n_determinants"),
                    "ordering": record.extra["orbital_order"],
                    "chi": int(record.extra["requested_bond_dim"]),
                    "seed": int(record.extra["seed"]),
                    "e_total": record.e_total,
                    "error": record.fci_discrepancy,
                    "walltime": record.walltime_seconds,
                    "dw": record.dmrg["final_discarded_weight"],
                    "effective_bond_dim": record.dmrg.get("effective_bond_dim"),
                    "s_max": record.extra.get("max_bipartite_entanglement"),
                    "digest": record.extra.get("integral_digest"),
                }
            )
    return exact, dict(dmrg)


def chi_star(runs: list[dict], tolerance: float) -> tuple[int | None, float | None]:
    """Smallest chi within ``tolerance`` of exact, and its wall time.

    Uses the *worst* seed at each chi. A bond dimension that reaches the target
    on one seed and not another has not reached it: the project measured on H10
    that low-chi DMRG can land on different plateaus from different starts, and
    picking the lucky one would report a convergence that does not reproduce.
    """
    by_chi: dict[int, list[dict]] = defaultdict(list)
    for run in runs:
        if run["error"] is not None:
            by_chi[run["chi"]].append(run)
    for chi in sorted(by_chi):
        errors = [abs(r["error"]) for r in by_chi[chi]]
        if max(errors) <= tolerance:
            return chi, max(r["walltime"] for r in by_chi[chi])
    return None, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path,
                        default=Path("data/raw/n2_ladder/n2_active_space_ladder.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("figures/n2_ladder.csv"))
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args(argv)

    exact, dmrg = load(args.store)
    orderings = sorted({run["ordering"] for runs in dmrg.values() for run in runs})

    rows = []
    for key in sorted(set(exact) | set(dmrg)):
        basis, bond_length, n_orb = key
        reference = exact.get(key)
        runs = dmrg.get(key, [])
        row = {
            "basis": basis,
            "bond_length_angstrom": bond_length,
            "active_orbitals": n_orb,
            # Above the exact wall there is no reference record, so the count
            # comes from the DMRG runs instead. It is a property of the active
            # space, not of whichever solver happened to be affordable.
            "n_determinants": (reference or (runs[0] if runs else {})).get(
                "n_determinants"
            ),
            "e_exact_hartree": (reference or {}).get("e_total"),
            "exact_walltime_s": (reference or {}).get("walltime"),
            "davidson_max_space": (reference or {}).get("max_space"),
        }
        for ordering in orderings:
            subset = [r for r in runs if r["ordering"] == ordering]
            chi, walltime = chi_star(subset, args.tolerance)
            best = min(subset, key=lambda r: abs(r["error"]) if r["error"] is not None
                       else float("inf"), default=None)
            row[f"chi_star_{ordering}"] = chi
            row[f"chi_star_walltime_s_{ordering}"] = walltime
            row[f"best_error_{ordering}"] = None if best is None else best["error"]
            row[f"s_max_{ordering}"] = None if best is None else best["s_max"]
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value, spec=""):
        return "—" if value is None else format(value, spec)

    header = ["R (Å)", "CAS", "dets", "E_exact (Ha)"]
    for ordering in orderings:
        header += [f"χ* {ordering}", f"t(χ*) {ordering}"]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        cells = [
            fmt(row["bond_length_angstrom"], ".3f"),
            f"({10}e,{row['active_orbitals']}o)",
            fmt(row["n_determinants"], ","),
            fmt(row["e_exact_hartree"], ".12f"),
        ]
        for ordering in orderings:
            cells.append(fmt(row[f"chi_star_{ordering}"]))
            cells.append(fmt(row[f"chi_star_walltime_s_{ordering}"], ".1f"))
        print("| " + " | ".join(cells) + " |")
    print(f"\nχ* = smallest bond dimension within {args.tolerance:g} Ha of exact "
          f"diagonalisation in the same active space, worst seed.")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
