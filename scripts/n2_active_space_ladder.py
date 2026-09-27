"""Phase 1 of the N2 experiment: DMRG against exact diagonalisation, rung by rung.

Runs DMRG and exact diagonalisation in ``CAS(10e,No)`` for N = 8, 12, 16 at
three bond lengths, in both orbital orderings and across a bond-dimension grid.

The point is not the energies. It is to establish, on a molecule with a frozen
core, a real basis and a genuinely multireference geometry, that

1. the active-space plumbing is right -- both solvers consume one set of
   integrals, evidenced by a shared digest, and agree to the numerical floor;
2. the *cost* of reaching that agreement is known as a function of the active
   space, the orbital ordering and the geometry.

(2) is what licenses Phase 2, where the space grows past exact diagonalisation
and there is no reference left to check against. A convergence protocol that
has not been calibrated where the answer is known is not a protocol.

    uv run python scripts/n2_active_space_ladder.py
    uv run python scripts/n2_active_space_ladder.py --orbitals 8 12 --chis 250
    uv run python scripts/n2_active_space_ladder.py --orderings fiedler --no-exact
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from tn_quantum_chemistry.n2_ladder import (
    DEFAULT_BOND_LENGTHS,
    DEFAULT_STACK_MEM_GB,
    LARGE_RUNG_STACK_MEM_GB,
    N2_FROZEN_CORE,
    N2_TOTAL_ELECTRONS,
    LadderPoint,
    build_active_space,
    determinant_count,
    dmrg_method_label,
    record_key,
    run_dmrg_rung,
    run_exact,
)
from tn_quantum_chemistry.schema import RecordStore, Status
from tn_quantum_chemistry.validation import DMRGSchedule

#: Phase 1 stays at or below the exact wall. 20 and 26 are Phase 2 and are not
#: run here, because nothing in this script would be able to check them.
DEFAULT_ORBITALS = (8, 12, 16)

#: Both orderings, because the comparison *is* one of the results. Canonical is
#: RHF energy order, which is what any naive active-space DMRG gets by default.
DEFAULT_ORDERINGS = ("canonical", "fiedler")

#: A grid rather than a single value: two neighbouring bond dimensions agreeing
#: is not convergence -- measured on H10, where the error sits on plateaus flat
#: to 1e-13 across a factor of two in chi and then moves by 1e-4.
DEFAULT_CHIS = (250, 500, 1000)

#: One seed everywhere, plus a second at the largest bond dimension, where a
#: seed difference would mean the sweep is landing in different minima rather
#: than truncating.
DEFAULT_SEEDS = (1234,)
CHECK_SEED = 5678

DEFAULT_BASES = ("cc-pvdz",)


def orbitals_available(basis: str, bond_length: float, n_frozen_core: int) -> int:
    """Active orbitals left after freezing ``n_frozen_core`` cores, for this basis."""
    from pyscf import gto

    from tn_quantum_chemistry.validation import diatomic

    geometry = diatomic("N", bond_length)
    mol = gto.M(
        atom=geometry.atom_spec, unit=geometry.unit, basis=basis,
        charge=geometry.charge, spin=geometry.spin, symmetry=False, verbose=0,
    )
    return int(mol.nao_nr()) - n_frozen_core


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-lengths", type=float, nargs="+", default=list(DEFAULT_BOND_LENGTHS))
    parser.add_argument("--bases", nargs="+", default=list(DEFAULT_BASES))
    parser.add_argument("--orbitals", type=int, nargs="+", default=list(DEFAULT_ORBITALS))
    parser.add_argument("--frozen-core", type=int, default=None,
                        help="inactive N 1s pairs; 2 (default) keeps every rung on the "
                             "same ten valence electrons, 0 gives the all-electron "
                             "CAS(14e,28o) rung that is full CI in cc-pVDZ")
    parser.add_argument("--orderings", nargs="+", default=list(DEFAULT_ORDERINGS))
    parser.add_argument("--chis", type=int, nargs="+", default=list(DEFAULT_CHIS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--check-seed", type=int, default=CHECK_SEED,
                        help="extra seed, run only at the largest chi; 0 disables")
    parser.add_argument("--sweeps", type=int, default=24)
    parser.add_argument("--stack-mem-gb", type=int, default=None,
                        help="block2 stack memory; defaults to 4 GB at or below "
                             "16 orbitals and 24 GB above, where a 4 GB stack "
                             "will not hold the MPS")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--exact-max-space", type=int, default=None,
                        help="override the Davidson subspace for the exact reference; "
                             "run twice with different values to measure the "
                             "reference's own uncertainty")
    parser.add_argument("--no-exact", action="store_true",
                        help="skip exact diagonalisation even where it is affordable")
    parser.add_argument("--no-profile", action="store_true",
                        help="skip the converged-MPS inspection (entropies, real bond dims)")
    parser.add_argument("--outdir", type=Path, default=Path("data/raw/n2_ladder"))
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)

    store = RecordStore(args.outdir / "n2_active_space_ladder.jsonl")
    done = store.completed_keys()
    # Exact energies already in the store, so a re-run reuses the reference
    # instead of paying for it again -- 150 s at sixteen orbitals, and the
    # reference is what every discrepancy in the run is measured against.
    cached_exact = {
        record.key: record.e_total
        for record in store
        if record.method.startswith("CASCI") and record.status == Status.CONVERGED
    }
    chis = sorted(args.chis)
    n_written = n_skipped = 0

    for basis in args.bases:
        for bond_length in args.bond_lengths:
            n_frozen = (
                N2_FROZEN_CORE if args.frozen_core is None else args.frozen_core
            )
            available = orbitals_available(basis, bond_length, n_frozen)
            rungs = [n for n in sorted(args.orbitals) if n <= available]
            skipped = [n for n in sorted(args.orbitals) if n > available]
            if skipped:
                # Not a failure: STO-3G simply has no 12- or 16-orbital space to
                # correlate. Saying so beats an empty row.
                print(f"[{basis} R={bond_length}] {available} active orbitals available; "
                      f"skipping {skipped}", flush=True)

            for n_orb in rungs:
                stack_mem_gb = args.stack_mem_gb
                if stack_mem_gb is None:
                    stack_mem_gb = (
                        DEFAULT_STACK_MEM_GB if n_orb <= 16
                        else LARGE_RUNG_STACK_MEM_GB
                    )
                header = f"[{basis} R={bond_length:.3f} CAS(10e,{n_orb}o)]"
                n_elec = N2_TOTAL_ELECTRONS - 2 * n_frozen
                print(f"{header} CAS({n_elec}e,{n_orb}o), "
                      f"{determinant_count(n_orb, n_elec):,d} determinants", flush=True)

                e_exact: float | None = None
                for ordering in args.orderings:
                    point = LadderPoint(
                        bond_length=bond_length, basis=basis,
                        n_active_orbitals=n_orb, orbital_order=ordering,
                        n_frozen_core=n_frozen,
                    )
                    try:
                        start = time.perf_counter()
                        mf, asham, order = build_active_space(point, n_threads=args.threads)
                        scf_walltime = time.perf_counter() - start
                    except Exception:  # noqa: BLE001
                        print(f"{header} {ordering}: active space failed\n"
                              f"{traceback.format_exc()}", flush=True)
                        continue

                    # The exact energy is invariant under orbital relabelling, so
                    # it is computed once and reused. Running it per ordering
                    # would cost the same answer twice and, at 16 orbitals, an
                    # extra 9 GB of Davidson subspace.
                    exact_method = f"CASCI({n_elec}e,{n_orb}o)"
                    if args.exact_max_space is not None:
                        exact_method += f"[max_space={args.exact_max_space}]"
                    exact_key = record_key(point, exact_method)
                    # Deliberately not gated on --replace: the exact reference
                    # does not depend on the DMRG schedule, so re-running the
                    # sweep is no reason to spend another 13 minutes on the
                    # sixteen-orbital Davidson. --exact-max-space is the way to
                    # compute a second reference, and it carries its own label.
                    if not args.no_exact and e_exact is None and exact_key in cached_exact:
                        e_exact = cached_exact[exact_key]
                        n_skipped += 1
                        print(f"{header} exact {e_exact:.12f} (cached)", flush=True)

                    if not args.no_exact and e_exact is None:
                        for record in run_exact(
                            point, mf, asham, order, scf_walltime=scf_walltime,
                            max_space=args.exact_max_space,
                        ):
                            wrote = store.append(record, replace=args.replace)
                            n_written += wrote
                            n_skipped += not wrote
                            if record.status == Status.CONVERGED:
                                e_exact = record.e_total
                                print(f"{header} exact {record.e_total:.12f} "
                                      f"({record.walltime_seconds:.1f} s, "
                                      f"max_space={record.extra['davidson_max_space']})",
                                      flush=True)
                            else:
                                print(f"{header} exact FAILED: {record.error_message}",
                                      flush=True)

                    for chi in chis:
                        seeds = list(args.seeds)
                        if args.check_seed and chi == chis[-1]:
                            seeds.append(args.check_seed)
                        for seed in seeds:
                            schedule = DMRGSchedule.ramped(
                                chi, seed=seed, n_threads=args.threads,
                                n_sweeps=args.sweeps,
                            )
                            label = dmrg_method_label(point, schedule)
                            # Check the cache *before* paying for the sweep.
                            if record_key(point, label) in done and not args.replace:
                                n_skipped += 1
                                print(f"{header} {label}: cached", flush=True)
                                continue
                            try:
                                record = run_dmrg_rung(
                                    point, mf, asham, order, schedule,
                                    scf_walltime=scf_walltime, e_exact=e_exact,
                                    profile=not args.no_profile,
                                    stack_mem_gb=stack_mem_gb,
                                )
                            except Exception:  # noqa: BLE001
                                print(f"{header} {label} raised\n{traceback.format_exc()}",
                                      flush=True)
                                continue
                            wrote = store.append(record, replace=args.replace)
                            n_written += wrote
                            n_skipped += not wrote
                            # A failed run has no energy and no sweep
                            # statistics. Formatting them unconditionally
                            # crashed the whole sweep on the first failure --
                            # losing every later point in the job -- when
                            # block2 ran out of scratch space at 28 orbitals.
                            # The record itself was stored correctly; only the
                            # progress line died, which is the worst way to
                            # lose a twelve-hour run.
                            if record.status == Status.FAILED:
                                print(f"{header} {ordering:9s} chi={chi:5d} "
                                      f"seed={seed:5d} FAILED: "
                                      f"{record.error_message}", flush=True)
                                continue
                            delta = ("      -" if record.fci_discrepancy is None
                                     else f"{record.fci_discrepancy:+.3e}")
                            print(f"{header} {ordering:9s} chi={chi:5d} seed={seed:5d} "
                                  f"E {record.e_total:.12f}  dE {delta}  "
                                  f"dw {record.dmrg['final_discarded_weight']:.2e}  "
                                  f"{record.walltime_seconds:7.1f} s", flush=True)

    print(f"\nwrote {n_written}, skipped {n_skipped} (already cached)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
