"""Vendor the Simons Collaboration H10 benchmark values into the repository.

Source: Motta et al., "Towards the Solution of the Many-Electron Problem in Real
Materials: Equation of State of the Hydrogen Chain with State-of-the-Art
Many-Body Methods", Phys. Rev. X 7, 031059 (2017), data repository
https://github.com/simonsfoundation/hydrogen-benchmark-PRX (CC BY 4.0).

The values are vendored rather than fetched at test time so the regression
suite does not depend on the network or on a third-party repository staying
put. Re-run this script to refresh them.

Conventions, established by reproduction rather than from documentation, which
the repository does not provide: the directory name ``R_x`` is the
nearest-neighbour spacing **in Bohr**, and ``basis-STO`` is **STO-6G**. Computing
H10 at those geometries in STO-6G reproduces the published FCI energies to a few
times 1e-9 Ha, while STO-3G is off by 3.6-6.6e-2 Ha -- which is what fixes the
basis rather than leaving it a guess.

Run with::

    uv run python scripts/fetch_motta_reference.py
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

RAW_BASE = (
    "https://raw.githubusercontent.com/simonsfoundation/"
    "hydrogen-benchmark-PRX/master"
)
SYSTEM = "N_10_OBC"

#: Spacings published for H10 with open boundary conditions, in Bohr.
SPACINGS = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 3.6)

#: Published method -> the name this project uses for the same calculation.
#: Only methods this project can reproduce are vendored; UHF is kept as context
#: for the spin-symmetry-breaking discussion even though nothing here runs it.
METHODS = {
    "RHF_basis-STO": "RHF",
    "UHF_basis-STO": "UHF",
    "RCCSD_basis-STO": "CCSD",
    "RCCSD(T)_basis-STO": "CCSD(T)",
    "FCI_basis-STO": "FCI",
    "DMRG_basis-STO": "DMRG",
}

CITATION = (
    "M. Motta et al. (Simons Collaboration on the Many-Electron Problem), "
    "Phys. Rev. X 7, 031059 (2017), doi:10.1103/PhysRevX.7.031059. "
    "Data: https://github.com/simonsfoundation/hydrogen-benchmark-PRX, CC BY 4.0."
)


def fetch(spacing: float, filename: str, *, timeout: int = 60) -> str | None:
    # One decimal, always: the directories are R_1.0 and R_2.0, so "%g"
    # formatting silently drops those two spacings.
    url = f"{RAW_BASE}/{SYSTEM}/R_{spacing:.1f}/{urllib.parse.quote(filename)}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode().strip()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/reference/motta_h10_obc_sto6g.json"),
    )
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    energies: dict[str, dict[str, float]] = {}
    decimals: dict[str, int] = {}
    for spacing in SPACINGS:
        row: dict[str, float] = {}
        for filename, method in METHODS.items():
            raw = fetch(spacing, filename)
            if raw is None:
                continue
            row[method] = float(raw.split()[0])
            # Published precision differs by method (FCI to 8 decimals, DMRG to
            # 6). Recording it lets each comparison use a tolerance the source
            # can actually support instead of one global number.
            digits = len(raw.split()[0].split(".")[-1])
            decimals[method] = min(decimals.get(method, digits), digits)
        if not row:
            raise RuntimeError(f"no data retrieved for R = {spacing} Bohr")
        energies[f"{spacing:.1f}"] = row
        print(f"  R = {spacing:>4.1f} Bohr: {len(row)} methods — " + ", ".join(sorted(row)))

    payload = {
        "citation": CITATION,
        "license": "CC BY 4.0",
        "source_repository": "simonsfoundation/hydrogen-benchmark-PRX",
        "system": "H10 linear chain, open boundary conditions",
        "basis": "STO-6G",
        "basis_note": (
            "The repository labels this 'basis-STO'. It is STO-6G: recomputing "
            "these geometries in STO-6G reproduces the published FCI energies "
            "to a few times 1e-9 Ha, while STO-3G differs by 3.6-6.6e-2 Ha."
        ),
        "spacing_unit": "Bohr",
        "spacing_note": (
            "The repository does not state units. Bohr is established by the "
            "same reproduction; interpreting R as Angstrom does not reproduce "
            "any published value."
        ),
        "energy_unit": "Hartree",
        "published_decimals": decimals,
        "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "energies": energies,
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(f"\nwritten to {args.out} ({args.out.stat().st_size / 1024:.1f} kB)")
    print(f"published decimals per method: {decimals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
