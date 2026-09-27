"""Score the N2 ladder against published near-exact energies.

Every other check in this project compares the ladder to something the project
computed itself. Above sixteen orbitals that stops being possible: CAS(10e,26o)
is 4.3e9 determinants and CAS(14e,28o) is 1.4e12, so there is no exact reference
to compare against and the uncertainty is built out of what varies -- the last
step in the chi sweep, the gap between two orbital orderings, the seed spread,
the residual sweep drift. That recipe has never been tested against a known
answer, because until now there was no known answer.

There is one in the literature. Chan, Kallay and Gauss computed the all-electron
N2/cc-pVDZ binding curve by DMRG, and Wang, Li and Lu later reproduced its
equilibrium point by coordinate-descent FCI; the two agree to 1.6e-5 Ha, which is
seventy times finer than this project's own uncertainty at that rung. Sharma and
co-workers give a frozen-core value by semistochastic heat-bath CI.

**Neither is at a geometry this project already runs**, and the two sources use
different units -- 2.118 Bohr for the all-electron value, 1.0977 Angstrom for the
frozen-core one. So the all-electron comparison is made by *running the ladder at
their bond length*, which is what ``external_check_*`` in ``data/raw/n2_ladder/``
holds. The frozen-core comparison is made at our own 1.098 A against their
1.0977 A, and the script reports that offset rather than hiding it: measured by
CCSD(T) it is worth 3.5e-5 Ha, fifteen times below the uncertainty we quote on
that rung, so it does not decide the comparison. A dedicated run at 1.0977 A
would not sharpen anything, because the published frozen-core value is itself a
stochastic quantity resolving only at 1e-4.

What this can establish is narrow and worth having: whether the ladder's
*uncertainty estimator* is honest where it cannot be checked -- whether the true
error falls inside the band the project quotes.

    uv run python scripts/n2_external_check.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from tn_quantum_chemistry.n2_ladder import (
    DEFAULT_BOND_LENGTHS,
    load_ladder,
    summarise_rung,
)
from tn_quantum_chemistry.reference_data import (
    n2_best_estimate,
    n2_bond_length_angstrom,
    n2_curve_energy,
    n2_reference,
)

#: Which stored rung answers which published value, and at which geometry.
#:
#: ``bond_length`` is the geometry *in our stores*. For the all-electron rung it
#: is the published one, because the ladder was run there on purpose; for the
#: frozen-core rung it is our own grid point, 0.0003 A away from theirs.
COMPARISONS = {
    "all_electron": {
        "bond_length": n2_bond_length_angstrom("all_electron"),
        "n_electrons": 14,
        "n_orbitals": 28,
        "geometry_matches": True,
    },
    "frozen_core": {
        "bond_length": 1.098,
        "n_electrons": 10,
        "n_orbitals": 26,
        "geometry_matches": False,
    },
}

#: CCSD(T)-measured cost of the 1.098 A / 1.0977 A offset, in Hartree. Asserted
#: in ``tests/test_n2_external_reference.py``, which recomputes it rather than
#: trusting this constant; it is carried here only so the printed table can say
#: how large the offset is without launching a calculation.
FROZEN_CORE_GEOMETRY_OFFSET = -3.5e-5

#: A verdict needs an uncertainty, and the uncertainty needs something to vary.
#: With one bond dimension there is no chi step, and with one ordering there is
#: no ordering gap -- which is the term that carries the most weight, and the
#: only one that has ever caught a converged-looking wrong answer. Scoring a
#: partial sweep against a published number would manufacture a disagreement out
#: of an unfinished calculation, so an incomplete rung is reported as incomplete.
MIN_CHI_POINTS = 2
MIN_ORDERINGS = 2

#: The all-electron rung this project already ran, at its own three geometries.
#: The published curve is dense enough (0.05 a0 near equilibrium) to be
#: interpolated to two of them, which checks stored results at no compute cost.
CURVE_RUNG = (14, 28)


def rung_is_complete(runs: list[dict]) -> bool:
    """Whether this rung has enough spread for its uncertainty to mean anything."""
    return (
        len({r["chi"] for r in runs}) >= MIN_CHI_POINTS
        and len({r["ordering"] for r in runs}) >= MIN_ORDERINGS
    )


def fmt(value: float | None, spec: str = ".6f") -> str:
    return "—" if value is None else format(value, spec)


def ratio(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}×"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ladder", type=Path, nargs="+",
        default=sorted(Path().glob("data/raw/n2_ladder/**/n2_active_space_ladder.jsonl")),
        help="record stores to read; defaults to every store under data/raw/n2_ladder",
    )
    parser.add_argument("--out", type=Path, default=Path("figures/n2_external_check.csv"))
    args = parser.parse_args(argv)

    ladder = load_ladder(args.ladder)
    rows: list[dict[str, Any]] = []

    for space, spec in COMPARISONS.items():
        published_block = n2_reference(space)
        published, published_uncertainty = n2_best_estimate(space)
        key = (
            round(spec["bond_length"], 6),
            spec["n_electrons"],
            spec["n_orbitals"],
        )
        runs = ladder.get(key)
        row: dict[str, Any] = {
            "space": space,
            "n_electrons": spec["n_electrons"],
            "n_orbitals": spec["n_orbitals"],
            "meaning": (
                "full CI in cc-pVDZ" if space == "all_electron"
                else "frozen-core FCI in cc-pVDZ"
            ),
            "our_bond_length_angstrom": spec["bond_length"],
            "published_bond_length": published_block["bond_length"],
            "published_bond_length_unit": published_block["bond_length_unit"],
            "geometry_matches": spec["geometry_matches"],
            "e_published": published,
            "uncertainty_published": published_uncertainty,
            "method_published": published_block["best_estimate_method"],
        }
        if runs is None:
            row.update(
                complete=False, e_ours=None, uncertainty_ours=None, chi_max=None,
                n_runs=None, difference=None, within_combined_uncertainty=None,
            )
            rows.append(row)
            continue

        stats = summarise_rung(runs)
        complete = rung_is_complete(runs)
        difference = stats["e_best"] - published
        combined = (stats["uncertainty"] or 0.0) + published_uncertainty
        row.update(
            complete=complete,
            e_ours=stats["e_best"],
            uncertainty_ours=stats["uncertainty"],
            ordering_gap=stats["ordering_gap"],
            chi_step=stats["chi_step"],
            sweep_drift=stats["sweep_drift"],
            chi_max=stats["chi_max"],
            n_runs=stats["n_runs"],
            total_walltime_s=stats["total_walltime_s"],
            difference=difference,
            within_combined_uncertainty=(
                abs(difference) <= combined if complete else None
            ),
            sigma_over_error=(
                (stats["uncertainty"] / abs(difference))
                if stats["uncertainty"] and difference else None
            ),
            # The ordering gap on its own. It is the term this project trusts
            # most -- the only diagnostic that ever caught a converged-looking
            # wrong answer -- so whether it is also the right *size* for the
            # error is worth separating from the max that gets quoted.
            ordering_gap_over_error=(
                (stats["ordering_gap"] / abs(difference))
                if stats["ordering_gap"] and difference else None
            ),
        )
        rows.append(row)

    # Second comparison: our existing geometries against the published curve.
    # Nothing is recomputed here -- the curve is dense enough to be interpolated
    # to two of the three bond lengths the ladder already ran, which turns
    # stored results into external checks for free. The third is outside the
    # published range and is refused rather than extrapolated to.
    from pyscf.data.nist import BOHR

    n_elec, n_orb = CURVE_RUNG
    for bond_length in DEFAULT_BOND_LENGTHS:
        runs = ladder.get((round(bond_length, 6), n_elec, n_orb))
        row = {
            "space": "all_electron_curve",
            "n_electrons": n_elec,
            "n_orbitals": n_orb,
            "meaning": "full CI in cc-pVDZ",
            "our_bond_length_angstrom": bond_length,
            "published_bond_length": bond_length / BOHR,
            "published_bond_length_unit": "Bohr",
            "geometry_matches": True,
            "method_published": "CDFCI (interpolated)",
        }
        try:
            published, published_uncertainty = n2_curve_energy(bond_length / BOHR)
        except ValueError as exc:
            row.update(
                complete=False, e_published=None, uncertainty_published=None,
                e_ours=None if runs is None else summarise_rung(runs)["e_best"],
                uncertainty_ours=None, difference=None,
                within_combined_uncertainty=None, note=str(exc),
            )
            rows.append(row)
            continue

        row["e_published"] = published
        row["uncertainty_published"] = published_uncertainty
        if runs is None:
            row.update(complete=False, e_ours=None, uncertainty_ours=None,
                       difference=None, within_combined_uncertainty=None)
            rows.append(row)
            continue

        stats = summarise_rung(runs)
        complete = rung_is_complete(runs)
        difference = stats["e_best"] - published
        combined = (stats["uncertainty"] or 0.0) + published_uncertainty
        row.update(
            complete=complete,
            e_ours=stats["e_best"],
            uncertainty_ours=stats["uncertainty"],
            ordering_gap=stats["ordering_gap"],
            chi_max=stats["chi_max"],
            n_runs=stats["n_runs"],
            difference=difference,
            within_combined_uncertainty=(
                abs(difference) <= combined if complete else None
            ),
            # DMRG is variational, so a converged run must sit ABOVE the exact
            # answer. A negative difference larger than the reference's own
            # uncertainty would mean one of the two numbers is wrong, and it
            # would not be detectable from anything inside this project.
            above_reference=difference > -published_uncertainty,
            # The point of the whole exercise. sigma is what this project
            # quotes where it cannot check itself; |difference| is the error it
            # was actually making. Their ratio says whether the estimator is
            # honest, and by how much it over- or under-covers.
            sigma_over_error=(
                (stats["uncertainty"] / abs(difference))
                if stats["uncertainty"] and difference else None
            ),
            # The ordering gap on its own. It is the term this project trusts
            # most -- the only diagnostic that ever caught a converged-looking
            # wrong answer -- so whether it is also the right *size* for the
            # error is worth separating from the max that gets quoted.
            ordering_gap_over_error=(
                (stats["ordering_gap"] / abs(difference))
                if stats["ordering_gap"] and difference else None
            ),
        )
        rows.append(row)

    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    direct = [r for r in rows if r["space"] != "all_electron_curve"]
    curve = [r for r in rows if r["space"] == "all_electron_curve"]

    def verdict_of(row: dict[str, Any]) -> str:
        if row.get("note"):
            return "outside published range"
        if row["e_ours"] is None:
            return "not yet run"
        if not row["complete"]:
            return "sweep incomplete"
        return "yes" if row["within_combined_uncertainty"] else "NO"

    print("### At the published geometry\n")
    print("| rung | χ (runs) | our E | our σ | published E | ours − published "
          "| σ / |err| | gap / |err| | inside σ? |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in direct:
        print(
            f"| {row['n_electrons']}e{row['n_orbitals']}o "
            f"| {row['chi_max'] or '—'} ({row['n_runs'] or 0}) "
            f"| {fmt(row['e_ours'])} | {fmt(row['uncertainty_ours'], '.1e')} "
            f"| {fmt(row['e_published'])} "
            f"| {fmt(row['difference'], '+.2e')} "
            f"| {ratio(row.get('sigma_over_error'))} "
            f"| {ratio(row.get('ordering_gap_over_error'))} | {verdict_of(row)} |"
        )

    print("\n### Our own geometries, against the published curve\n")
    print("These are stored results. Nothing was recomputed: the published curve "
          "is dense enough near equilibrium to interpolate onto two of the three "
          "bond lengths this project already ran.\n")
    print("| R (Å) | R (a₀) | our E | our σ | published E | ours − published "
          "| σ / |err| | gap / |err| | above ref? | inside σ? |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in curve:
        above = row.get("above_reference")
        print(
            f"| {row['our_bond_length_angstrom']:.3f} "
            f"| {row['published_bond_length']:.4f} "
            f"| {fmt(row['e_ours'])} | {fmt(row['uncertainty_ours'], '.1e')} "
            f"| {fmt(row['e_published'])} "
            f"| {fmt(row['difference'], '+.2e')} "
            f"| {ratio(row.get('sigma_over_error'))} "
            f"| {ratio(row.get('ordering_gap_over_error'))} "
            f"| {'—' if above is None else ('yes' if above else 'NO')} "
            f"| {verdict_of(row)} |"
        )

    print()
    for row in direct:
        if row["e_ours"] is None:
            continue
        if not row["complete"]:
            print(f"- {row['n_electrons']}e{row['n_orbitals']}o: the chi sweep is "
                  f"still running ({row['n_runs']} runs, largest χ = {row['chi_max']}). "
                  f"No verdict: the difference above is against an unconverged energy.")
            continue
        if row["geometry_matches"]:
            print(f"- {row['n_electrons']}e{row['n_orbitals']}o: same geometry, "
                  f"{row['published_bond_length']} {row['published_bond_length_unit']}, "
                  f"and the same Hamiltonian — the published RHF energy is reproduced "
                  f"to 2.1e-8 Ha (tests/test_n2_external_reference.py).")
        else:
            print(f"- {row['n_electrons']}e{row['n_orbitals']}o: compared across a "
                  f"{abs(row['our_bond_length_angstrom'] - n2_bond_length_angstrom(row['space'])):.4f} Å "
                  f"geometry offset, worth {FROZEN_CORE_GEOMETRY_OFFSET:+.1e} Ha by CCSD(T) — "
                  f"below our own {fmt(row['uncertainty_ours'], '.1e')} Ha uncertainty, so it "
                  f"does not decide the comparison.")
    for row in curve:
        if row.get("note"):
            print(f"- {row['our_bond_length_angstrom']:.3f} Å = "
                  f"{row['published_bond_length']:.4f} a₀: {row['note']}.")

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
