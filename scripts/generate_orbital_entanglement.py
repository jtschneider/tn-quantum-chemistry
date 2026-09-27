"""Experiment 3: orbital entropies and mutual information at three geometries.

Computes ``s_i`` and ``I_ij`` for the H10 chain at a weakly, a moderately and a
strongly correlated spacing, and caches the result with its validation record.
Figures are produced separately from this data.

Each point runs twice: the validated SU2 calculation supplies the energy, the
reduced density matrices and ``s_i``; an SZ calculation supplies ``s_ij``,
because ``get_orbital_entropies`` raises in SU2 mode in block2 0.5.4rc16. The
SZ state's energy, ``<S^2>`` and ``s_i`` are all checked against the SU2 route
before its two-orbital entropies are used, and the script fails loudly rather
than record entanglement for a state that is not the one validated.

Run with::

    uv run python scripts/generate_orbital_entanglement.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tn_quantum_chemistry.orbital_entanglement import (
    compute_orbital_entanglement,
    lowdin_hamiltonian,
)
from tn_quantum_chemistry.schema import hardware_id, software_provenance
from tn_quantum_chemistry.validation import (
    HamiltonianSpec,
    active_space_hamiltonian,
    build_mean_field,
    linear_hydrogen_chain,
    run_fci,
)

#: Weak, moderate and strong correlation on the validated delta = 0 curve.
DEFAULT_SPACINGS = (1.0, 1.8, 2.6)

REGIME = {1.0: "weakly correlated", 1.8: "moderately correlated",
          2.6: "strongly correlated"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("data/raw/entanglement"))
    parser.add_argument("--n-atoms", type=int, default=10)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--spacings", type=float, nargs="+", default=list(DEFAULT_SPACINGS))
    parser.add_argument("--sz-bond-dim", type=int, default=1500)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    ham = HamiltonianSpec(basis=args.basis)
    results = {}
    print(f"H{args.n_atoms}/{args.basis}, delta = 0, SZ bond dimension "
          f"{args.sz_bond_dim}\nhardware: {hardware_id()}\n")

    for spacing in args.spacings:
        geometry = linear_hydrogen_chain(
            args.n_atoms, R=spacing, delta=0.0,
            label=f"H{args.n_atoms}_R{spacing:g}_d0",
        )
        print(f"--- R = {spacing:g} A ({REGIME.get(spacing, 'unclassified')})",
              flush=True)
        start = time.perf_counter()
        mf = build_mean_field(geometry, ham)
        asham = active_space_hamiltonian(mf, ham)
        e_fci, _ = run_fci(asham)
        entanglement = compute_orbital_entanglement(
            asham,
            localised_asham=lowdin_hamiltonian(mf, ham),
            sz_bond_dim=args.sz_bond_dim,
            n_threads=args.threads,
        )
        elapsed = time.perf_counter() - start

        check = entanglement.cross_check
        print(f"    E_FCI = {e_fci:.10f} Ha")
        print(f"    SU2 - FCI = {check['su2_energy'] - e_fci:+.2e}, "
              f"SZ - SU2 = {check['energy_difference']:.2e}, "
              f"<S^2> = {check['spin_squared']:.2e}")
        print(f"    s_i routes agree to {check['single_orbital_entropy_difference']:.2e}")
        localised = entanglement.localised_single_orbital_entropy
        print(f"    sum s_i = {entanglement.total_single_orbital_entropy:.4f}, "
              f"max I_ij = {entanglement.invariants['max_mutual_information']:.4f}")
        print(f"    mean s_i: {np.mean(entanglement.single_orbital_entropy):.4f} "
              f"canonical MO vs {np.mean(localised):.4f} Löwdin AO "
              f"(ln4 = {np.log(4):.4f}, ln2 = {np.log(2):.4f})")
        worst = max(
            v for k, v in entanglement.invariants.items() if k.endswith("violation")
        )
        print(f"    worst invariant violation: {worst:.1e}   ({elapsed:.0f}s)",
              flush=True)

        results[f"{spacing:g}"] = {
            "geometry": {
                "label": geometry.label,
                "R": spacing,
                "delta": 0.0,
                "unit": geometry.unit,
                "atoms": [[s, *p] for s, p in geometry.atoms],
                "regime": REGIME.get(spacing, "unclassified"),
            },
            "basis": args.basis,
            "n_orbitals": asham.n_orbitals,
            "n_electrons": asham.n_electrons,
            "e_fci": e_fci,
            "walltime_seconds": elapsed,
            **entanglement.to_dict(),
        }

    payload = {
        "experiment": "orbital entanglement (single-orbital entropy and mutual "
                      "information)",
        "orbital_basis": "canonical RHF, no frozen core, PySCF ordering",
        "localised_orbital_basis": (
            "symmetrically orthogonalised (Löwdin) AOs, S^(-1/2); one orbital per "
            "hydrogen atom. Reported alongside the canonical values because s_i "
            "describes the orbital partition as much as the state: the identical "
            "state gives mean s_i near ln 4 in canonical orbitals and near ln 2 "
            "in these, the latter being the one-electron-per-site Mott signature."
        ),
        "conventions": {
            "mutual_information": "I_ij = (s_i + s_j - s_ij) / 2, zero on the diagonal",
            "note": "s_i and I_ij depend on the orbital basis, the active space "
                    "and the ordering; they are not invariants of the molecule.",
        },
        "citations": [
            "O. Legeza and J. Solyom, Phys. Rev. B 68, 195116 (2003)",
            "J. Rissler, R. M. Noack, S. R. White, Chem. Phys. 323, 519 (2006)",
            "K. Boguslawski, P. Tecmer, Int. J. Quantum Chem. 115, 1289 (2015)",
        ],
        "hardware_id": hardware_id(),
        "software": software_provenance(),
        "results": results,
    }
    path = args.outdir / f"h{args.n_atoms}_{args.basis}_orbital_entanglement.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwritten to {path} ({path.stat().st_size / 1024:.0f} kB)")

    worst = max(
        v for r in results.values() for k, v in r["invariants"].items()
        if k.endswith("violation")
    )
    print(f"worst invariant violation across all geometries: {worst:.2e}")
    return 0 if worst < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
