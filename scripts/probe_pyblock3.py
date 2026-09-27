"""Day-1 probe: is pyblock3 a usable DMRG fallback if block2 fails?

The risk register treats pyblock3 as a *candidate* fallback, not an assumed
easier install. This script settles that question with evidence and records the
answer so the decision is documented rather than hypothetical.

What it does:

1. Builds the H8/STO-6G R=1.8 Bohr chain -- the system in pyblock3's own
   published DMRG example -- and writes an FCIDUMP.
2. Computes the PySCF FCI energy *from that same FCIDUMP*, so the comparison
   cannot be spoiled by an integral writer/reader convention mismatch.
3. Runs pyblock3's published DMRG recipe on the FCIDUMP in a disposable ``uv``
   environment, isolated from the project environment.
4. Reports whether the fallback reproduces the reference energy.

The disposable environment is required: pyblock3 does not build on the
project's Python 3.14, and the environment policy forbids downgrading the main
environment to rescue an optional fallback. Its preview channel does publish
cp313 wheels, so the probe runs on Python 3.13 -- a current interpreter, in a
throwaway environment that never touches ``uv.lock``.

Run with::

    uv run python scripts/probe_pyblock3.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from tn_quantum_chemistry.schema import software_provenance

PYBLOCK3_INDEX = "https://block-hczhai.github.io/pyblock3-preview/pypi/"
PROBE_PYTHON = "3.13"

# pyblock3's published small-H-chain example, adapted only to take the FCIDUMP
# path and emit JSON. The DMRG recipe itself is unchanged.
_PYBLOCK3_SNIPPET = r'''
import json, sys, time
import importlib.metadata as md

out = {"stage": "import"}
try:
    from pyblock3.algebra.mpe import MPE
    from pyblock3.hamiltonian import Hamiltonian
    from pyblock3.fcidump import FCIDUMP
    out["pyblock3_version"] = md.version("pyblock3")
    out["numpy_version"] = md.version("numpy")
    out["python"] = sys.version.split()[0]

    fd_path, pg = sys.argv[1], sys.argv[2]
    out["stage"] = "build_hamiltonian"
    start = time.perf_counter()
    hamil = Hamiltonian(FCIDUMP(pg=pg).read(fd_path), flat=True)

    out["stage"] = "build_mpo"
    mpo = hamil.build_qc_mpo()
    mpo, _ = mpo.compress(left=True, cutoff=1e-9, norm_cutoff=1e-9)

    out["stage"] = "dmrg"
    mps = hamil.build_mps(250)
    dmrg = MPE(mps, mpo, mps).dmrg(
        bdims=[250], noises=[1e-6, 0], dav_thrds=[1e-10], iprint=0, n_sweeps=20, tol=1e-12
    )
    out["energy"] = float(dmrg.energies[-1])
    out["n_sweeps_run"] = len(dmrg.energies)
    out["walltime_seconds"] = time.perf_counter() - start
    out["stage"] = "complete"
    out["ok"] = True
except Exception as exc:
    out["ok"] = False
    out["error"] = f"{type(exc).__name__}: {exc}"
print("@@JSON@@" + json.dumps(out))
'''


def build_h8_fcidump(path: Path, *, n_atoms: int = 8, spacing_bohr: float = 1.8,
                     basis: str = "sto-6g") -> dict[str, Any]:
    """Write the FCIDUMP for pyblock3's published example system."""
    from pyscf import gto, scf
    from pyscf.tools import fcidump as fcidump_tools

    z = np.arange(n_atoms) * spacing_bohr
    z = z - z.mean()
    mol = gto.M(
        atom=[["H", (0.0, 0.0, float(zi))] for zi in z],
        unit="Bohr",
        basis=basis,
        symmetry=False,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge for the H8 probe system")

    fcidump_tools.from_scf(mf, str(path), tol=1e-15)
    return {
        "n_atoms": n_atoms,
        "spacing_bohr": spacing_bohr,
        "basis": basis,
        "e_hf": float(mf.e_tot),
        "e_nuclear_repulsion": float(mol.energy_nuc()),
        "n_orbitals": int(mf.mo_coeff.shape[1]),
        "n_electrons": int(mol.nelectron),
    }


def reference_fci_from_fcidump(path: Path) -> tuple[float, float]:
    """FCI energy computed from the written FCIDUMP itself."""
    from pyscf import ao2mo, fci
    from pyscf.tools import fcidump as fcidump_tools

    data = fcidump_tools.read(str(path))
    norb = int(data["NORB"])
    h1e = data["H1"]
    g2e = ao2mo.restore(1, data["H2"], norb)
    nelec = int(data["NELEC"])
    ecore = float(data["ECORE"])

    start = time.perf_counter()
    e_total, _ = fci.direct_spin1.kernel(
        h1e, g2e, norb, nelec, ecore=ecore, conv_tol=1e-12, max_cycle=500
    )
    return float(e_total), time.perf_counter() - start


def run_pyblock3(fcidump_path: Path, *, pg: str = "c1", timeout: int = 900) -> dict[str, Any]:
    """Run the pyblock3 example in a disposable uv environment."""
    with tempfile.TemporaryDirectory() as tmp:
        snippet = Path(tmp) / "pyblock3_example.py"
        snippet.write_text(_PYBLOCK3_SNIPPET)
        command = [
            "uv", "run", "--no-project", "--isolated",
            "--python", PROBE_PYTHON,
            "--index", PYBLOCK3_INDEX,
            "--with", "pyblock3",
            "--with", "numpy>=2",
            "python", str(snippet), str(fcidump_path), pg,
        ]
        start = time.perf_counter()
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
        elapsed = time.perf_counter() - start

    result: dict[str, Any] = {
        "command": " ".join(command[:-3] + ["<snippet>", "<fcidump>", pg]),
        "returncode": proc.returncode,
        "total_walltime_seconds": elapsed,
    }
    for line in proc.stdout.splitlines():
        if line.startswith("@@JSON@@"):
            result.update(json.loads(line.removeprefix("@@JSON@@")))
            break
    else:
        # A native fatal error kills the interpreter before the snippet's own
        # try/except can report anything, so the diagnosis has to be recovered
        # from the process output.
        result["ok"] = False
        result["error"] = "process died before returning a JSON payload"
        result["stdout_tail"] = proc.stdout[-2000:]
        result["stderr_tail"] = proc.stderr[-2000:]
        combined = proc.stdout + proc.stderr
        if "MKL" in combined and "FATAL" in combined:
            result["failure_mode"] = "bundled_mkl_incomplete"
            result["diagnosis"] = (
                "The pyblock3 wheel bundles Intel MKL 1.x in pyblock3.libs/ but omits "
                "libmkl_def.so.1, the generic kernel library MKL's dispatcher always "
                "dlopens at initialisation. MKL resolves it by absolute path inside the "
                "wheel's own library directory, so supplying the missing file elsewhere "
                "on LD_LIBRARY_PATH does not help. The MPO build reaches site 0 and the "
                "process is then killed by MKL, which is a native fatal error rather "
                "than a Python exception."
            )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe pyblock3 as a DMRG fallback.")
    parser.add_argument("--outdir", type=Path, default=Path("data/raw/smoke"))
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="accepted |E_pyblock3 - E_FCI| in Hartree")
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        fcidump_path = Path(tmp) / "H8.STO6G.R1.8.FCIDUMP"
        print("building H8/STO-6G R=1.8 Bohr FCIDUMP ...", flush=True)
        system = build_h8_fcidump(fcidump_path)
        print(f"  {system['n_orbitals']} orbitals, {system['n_electrons']} electrons, "
              f"E_HF = {system['e_hf']:.12f} Ha", flush=True)

        print("computing reference FCI from the same FCIDUMP ...", flush=True)
        e_fci, t_fci = reference_fci_from_fcidump(fcidump_path)
        print(f"  E_FCI = {e_fci:.12f} Ha  ({t_fci:.2f}s)", flush=True)

        print(f"running pyblock3 example in a disposable Python {PROBE_PYTHON} env ...",
              flush=True)
        probe = run_pyblock3(fcidump_path)

    available = bool(probe.get("ok"))
    discrepancy = None
    reproduces = False
    if available and "energy" in probe:
        discrepancy = probe["energy"] - e_fci
        reproduces = abs(discrepancy) < args.tolerance

    report = {
        "report": "day1_pyblock3_fallback_probe",
        "verdict": (
            "available_and_validated" if reproduces
            else "available_but_disagrees" if available
            else "unavailable"
        ),
        "usable_as_fallback": reproduces,
        "probe_python": PROBE_PYTHON,
        "probe_index": PYBLOCK3_INDEX,
        "project_python": sys.version.split()[0],
        "installs_on_project_python": False,
        "workarounds_attempted": [
            {
                "workaround": "constrain the probe environment to mkl==2024.2.2",
                "outcome": "rejected: the constraint pushes resolution off the 0.2.9rc6 "
                           "wheel onto the 0.2.9rc5 sdist, whose cmake build fails",
            },
            {
                "workaround": "supply libmkl_def.so.1 from mkl==2021.4.0 via LD_LIBRARY_PATH",
                "outcome": "no effect: MKL dlopens the file by absolute path inside the "
                           "wheel's own pyblock3.libs/ directory",
            },
            {
                "workaround": "copy libmkl_def.so.1 into the cached wheel's pyblock3.libs/",
                "outcome": "not attempted: mutating uv's shared package cache and mixing "
                           "MKL builds is not a reproducible fallback",
            },
        ],
        "installs_on_project_python_note": (
            "pyblock3 publishes no cp314 wheel and its sdist build fails on Python 3.14; "
            "the preview channel publishes wheels up to cp313. Probed in a disposable "
            "environment per the environment policy; uv.lock is untouched."
        ),
        "system": system,
        "reference": {
            "method": "PySCF FCI from the same FCIDUMP",
            "e_fci": e_fci,
            "walltime_seconds": t_fci,
        },
        "pyblock3": probe,
        "discrepancy_hartree": discrepancy,
        "tolerance": args.tolerance,
        "software": software_provenance(),
    }
    path = args.outdir / "pyblock3_probe.json"
    path.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 68)
    if available:
        print(f"  pyblock3 {probe.get('pyblock3_version')} on Python {probe.get('python')}")
        print(f"  E_pyblock3 = {probe['energy']:.12f} Ha  "
              f"({probe.get('n_sweeps_run')} sweeps, {probe.get('walltime_seconds', 0):.2f}s)")
        print(f"  E_FCI      = {e_fci:.12f} Ha")
        print(f"  difference = {discrepancy:+.3e} Ha  (tol {args.tolerance:.0e})")
    else:
        print(f"  pyblock3 unavailable: {probe.get('error')}")
        print(f"  failure mode: {probe.get('failure_mode', 'unknown')}")
        if probe.get("diagnosis"):
            import textwrap
            print(textwrap.fill(probe["diagnosis"], width=66,
                                initial_indent="  ", subsequent_indent="  "))
    print(f"\n  FALLBACK STATUS: {report['verdict']}")
    print("=" * 68)
    print(f"written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
