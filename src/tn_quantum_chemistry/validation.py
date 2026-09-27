"""Experiment 1 — independent solver validation: PySCF FCI vs block2 DMRG.

This module is the project's validation gate. Every downstream DMRG result
(dissociation curves, entanglement measures, ML labels) is only publishable
once the comparison implemented here passes.

The claim under test is deliberately narrow:

    For one fixed finite Hamiltonian -- fixed geometry, basis, orbital source,
    electron count, spin sector, frozen-core choice and nuclear-repulsion
    convention -- an independently converged DMRG calculation reproduces the
    exact diagonalisation of that same Hamiltonian.

This validates the *solver*, not the chemistry. FCI in a minimal basis is exact
only for the finite Hamiltonian it diagonalises; it is not a chemically
converged reference.

Design decisions that matter for the gate to mean anything
----------------------------------------------------------
1. Both solvers consume the *same* ``h1e``/``g2e``/``ecore`` arrays, built once
   by :func:`active_space_hamiltonian`. The arrays are hashed and the digest is
   stored in the result record, so "same Hamiltonian" is evidence rather than
   an assumption.
2. Electronic and total energies are reported separately, and the nuclear
   repulsion is taken from ``mol.energy_nuc()`` rather than from the integral
   builder's ``ecore`` (the two differ as soon as a core is frozen).
3. DMRG is variational for the exact ground state, so ``E_DMRG >= E_FCI`` up to
   numerical noise. A DMRG energy meaningfully *below* FCI is a mismatched
   Hamiltonian, not a better calculation, and is recorded as such.
4. Point-group symmetry is off by default. C1 removes the irrep-ordering
   convention as a failure mode; symmetry is an optimisation, not part of the
   claim.

The geometry builders below are the minimum needed to run this experiment at
two geometries. They move to ``geometries.py`` when Experiment 2 starts.

Run with::

    uv run python -m tn_quantum_chemistry.validation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tn_quantum_chemistry.schema import hardware_id, software_provenance

# --------------------------------------------------------------------------- #
# Geometry (temporary home; moves to geometries.py for Experiment 2)
# --------------------------------------------------------------------------- #

Atom = tuple[str, tuple[float, float, float]]


@dataclass(frozen=True)
class Geometry:
    """A nuclear configuration with its units and electronic sector made explicit."""

    label: str
    atoms: list[Atom]
    unit: str = "Angstrom"
    charge: int = 0
    spin: int = 0  # PySCF convention: 2S = n_alpha - n_beta
    parameters: dict[str, float] = field(default_factory=dict)

    @property
    def atom_spec(self) -> list[list[Any]]:
        """Geometry in the list form PySCF's ``gto.M(atom=...)`` accepts."""
        return [[symbol, list(position)] for symbol, position in self.atoms]


def linear_hydrogen_chain(
    n_atoms: int = 10,
    R: float = 1.0,
    delta: float = 0.0,
    *,
    label: str | None = None,
) -> Geometry:
    """Linear H_n chain with uniform spacing ``R`` and alternating displacement ``delta``.

    Bond-length convention (this project): consecutive nearest-neighbour
    distances alternate as ``R + delta/2``, ``R - delta/2``, starting with the
    long bond. ``delta = 0`` is the uniformly spaced chain.

    Giner, Bendazzoli, Evangelisti and Monari, J. Chem. Phys. 138, 074315
    (2013) are reported to use ``R +/- 2 * delta`` for the same family of
    chains, which would make their dimerisation parameter four times smaller
    than this one; see :func:`giner_delta_to_project`. That paper is not open
    access and the convention has **not** been checked against its text.

    Distances are in Angstrom and the chain is centred on the origin along z.
    """
    if n_atoms < 2:
        raise ValueError(f"need at least two atoms, got {n_atoms}")
    if R <= 0.0:
        raise ValueError(f"spacing R must be positive, got {R}")

    bonds = [R + 0.5 * delta if k % 2 == 0 else R - 0.5 * delta for k in range(n_atoms - 1)]
    if min(bonds) <= 0.0:
        raise ValueError(f"delta={delta} produces a non-positive bond length for R={R}")

    z = np.concatenate([[0.0], np.cumsum(bonds)])
    z -= z.mean()
    atoms: list[Atom] = [("H", (0.0, 0.0, float(zi))) for zi in z]

    return Geometry(
        label=label or f"H{n_atoms}_R{R:g}_d{delta:g}",
        atoms=atoms,
        unit="Angstrom",
        charge=0,
        spin=0,
        parameters={"R": R, "delta": delta, "n_atoms": float(n_atoms)},
    )


def giner_delta_to_project(delta_giner: float) -> float:
    """Convert a Giner et al. dimerisation parameter to this project's convention.

    Giner bonds are taken to be ``R +/- 2*delta_giner``; ours are
    ``R +/- delta_project/2``. Equating the two gives
    ``delta_project = 4 * delta_giner``.

    **Unverified premise.** The factor rests on that reading of their
    convention, taken from the project plan rather than from the paper, which is
    paywalled. The accompanying test checks this function against *our* bond
    lengths -- that the conversion is self-consistent -- and cannot check that
    it matches theirs. Verify against the published text before any comparison
    at ``delta != 0`` is described as a reproduction. At ``delta = 0`` the two
    conventions coincide and nothing here applies.
    """
    return 4.0 * delta_giner


def project_delta_to_giner(delta_project: float) -> float:
    """Inverse of :func:`giner_delta_to_project`."""
    return delta_project / 4.0


#: Experimental gas-phase water: r(OH) = 0.9584 A, angle(HOH) = 104.45 deg.
_WATER_R_OH = 0.9584
_WATER_ANGLE = np.deg2rad(104.45)


def water(label: str = "H2O_equilibrium") -> Geometry:
    """H2O in the xz plane, oxygen at the origin, distances in Angstrom."""
    half = 0.5 * _WATER_ANGLE
    x, z = _WATER_R_OH * np.sin(half), _WATER_R_OH * np.cos(half)
    return Geometry(
        label=label,
        atoms=[
            ("O", (0.0, 0.0, 0.0)),
            ("H", (float(x), 0.0, float(z))),
            ("H", (float(-x), 0.0, float(z))),
        ],
        unit="Angstrom",
        parameters={"r_OH": _WATER_R_OH, "angle_HOH_deg": float(np.rad2deg(_WATER_ANGLE))},
    )


def diatomic(symbol: str, bond_length: float, *, label: str | None = None, **kwargs: Any) -> Geometry:
    """Homonuclear diatomic aligned along z, bond length in Angstrom."""
    half = 0.5 * bond_length
    return Geometry(
        label=label or f"{symbol}2_r{bond_length:g}",
        atoms=[(symbol, (0.0, 0.0, -half)), (symbol, (0.0, 0.0, half))],
        unit="Angstrom",
        parameters={"bond_length": bond_length},
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Hamiltonian and solver specifications
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HamiltonianSpec:
    """Everything besides the geometry that pins down the finite Hamiltonian."""

    basis: str = "sto-3g"
    n_frozen_core: int = 0
    n_active_orbitals: int | None = None  # None = all remaining orbitals
    point_group_symmetry: bool = False
    orbital_source: str = "RHF canonical"
    scf_conv_tol: float = 1e-12


@dataclass(frozen=True)
class DMRGSchedule:
    """Sweep schedule, stopping rule and seed for the block2 calculation."""

    bond_dims: list[int]
    noises: list[float]
    davidson_thresholds: list[float]
    n_sweeps: int
    energy_tol: float = 1e-12
    seed: int = 1234
    n_threads: int = 4
    #: Singular-value cutoff. The renormalised basis is truncated by whichever
    #: limit binds first, this or ``bond_dims``, so the *effective* bond
    #: dimension adapts to the geometry instead of being imposed on it. block2's
    #: own default, 1e-20, is far below the truncation any of these runs reach
    #: and therefore leaves ``bond_dims`` in sole control; it is kept as the
    #: default here so schedules built before this option existed are unchanged.
    cutoff: float = 1e-20

    @classmethod
    def default(cls, seed: int = 1234, n_threads: int = 4) -> DMRGSchedule:
        """A schedule that is heavily over-converged for <= ~16 orbitals.

        Small problems are cheap, so the validation sequence deliberately
        overshoots: the point is to remove bond-dimension truncation as an
        explanation for any disagreement, not to find the cheapest setting.
        """
        # Twenty-five entries, not twenty. The lists used to be five short of
        # ``n_sweeps``, leaving the last five sweeps to whatever block2 does with
        # a schedule that runs out -- settings nobody chose. In practice the
        # ambiguity was never reached, because this schedule meets its energy
        # tolerance around sweep 16, and making the tail explicit moves the
        # converged energy by 6e-14. It is spelled out so the question cannot
        # arise on a system that does run the full twenty-five.
        bond_dims = [150] * 4 + [250] * 4 + [500] * 6 + [600] * 11
        noises = [1e-4] * 8 + [1e-5] * 4 + [0.0] * 13
        return cls(
            bond_dims=bond_dims,
            noises=noises,
            davidson_thresholds=[1e-14] * 25,
            n_sweeps=25,
            energy_tol=1e-14,
            seed=seed,
            n_threads=n_threads,
        )

    @classmethod
    def fixed_bond_dimension(
        cls,
        bond_dim: int,
        seed: int = 1234,
        n_threads: int = 4,
        n_sweeps: int = 20,
        cutoff: float = 1e-20,
    ) -> DMRGSchedule:
        """A sweep schedule holding ``chi`` fixed, for a genuine fidelity tier.

        Unlike :meth:`default`, which deliberately over-converges to remove
        truncation as an explanation, this pins the bond dimension so that a
        low-fidelity tier is actually low fidelity.

        The solver thresholds nonetheless match the rest of the project (1e-14,
        not the 1e-12 used before): a low bond dimension is supposed to limit
        the *variational space*, and letting the local eigensolver stop early as
        well would confound truncation with solver convergence -- which is
        exactly the effect that made the DMRG/FCI agreement drift with chain
        spacing.

        The noise ramp matters more here than in :meth:`default`. At small
        ``chi`` the sweep can settle into a low-entanglement state that the MPS
        represents well but which is not the ground state; the perturbative
        noise in the early sweeps is what lets it escape. It is switched off for
        the second half so the reported energy is variational.
        """
        return cls(
            bond_dims=[bond_dim] * n_sweeps,
            noises=[1e-4] * 6 + [1e-6] * 4 + [0.0] * (n_sweeps - 10),
            davidson_thresholds=[1e-14] * n_sweeps,
            n_sweeps=n_sweeps,
            energy_tol=1e-14,
            seed=seed,
            n_threads=n_threads,
            cutoff=cutoff,
        )


    @classmethod
    def ramped(
        cls,
        bond_dim: int,
        *,
        cutoff: float = 1e-20,
        seed: int = 1234,
        n_threads: int = 4,
        n_sweeps: int = 20,
    ) -> DMRGSchedule:
        """Grow the bond dimension in stages toward ``bond_dim``.

        For systems large enough that a flat schedule wastes its early sweeps:
        the state reorganises most in the first few passes and those are far
        cheaper at small bond dimension. On H10 this changed nothing about the
        converged energy -- the plateaus there are variational, not dynamical --
        but it changes the cost substantially at length.

        Thresholds and the stopping rule come from the same convention as every
        other schedule here; see :meth:`fixed_bond_dimension`.
        """
        stages = [max(16, bond_dim // 16)] * 4
        stages += [max(16, bond_dim // 8)] * 4
        stages += [max(16, bond_dim // 4)] * 4
        stages += [max(16, bond_dim // 2)] * 4
        # At least four sweeps at the full bond dimension: two carrying noise
        # and two without, which is the minimum that both escapes a local
        # minimum and then reports a variational energy. A caller asking for
        # fewer gets a slightly longer schedule rather than a silently useless
        # one; the schedule stores its own length.
        n_full = max(4, n_sweeps - 16)
        stages += [bond_dim] * n_full
        n_sweeps = len(stages)

        # The noise must still be on once the full bond dimension is available,
        # and this is where the schedule was wrong until 2026-09-07. The ramp
        # spent its entire noise budget while chi was still a sixteenth of its
        # target, then arrived at the full bond dimension with noise already at
        # zero -- so the perturbative term that lets a sweep leave a local
        # minimum was gone before the MPS had the freedom to use it. On N2 in
        # CAS(10e,16o) that produced a *reproducible* convergence to a state
        # 0.19 Ha above the ground state, at every chi and both seeds, with a
        # final discarded weight of 5e-10 reporting success. The same geometry
        # and seed converge correctly under `fixed_bond_dimension`, which keeps
        # noise on at full chi; exact diagonalisation of the same integrals
        # confirmed the Hamiltonian was never at fault.
        n_noisy_full = max(2, n_full // 2)
        # Always leave at least two noise-free sweeps at the end, so the
        # reported energy is a variational expectation value and not one with a
        # perturbative term still folded into it.
        n_noise = min(n_sweeps - 2, (n_sweeps - n_full) + n_noisy_full)
        n_strong = min(8, n_noise)
        return cls(
            bond_dims=stages,
            noises=(
                [1e-4] * n_strong
                + [1e-6] * (n_noise - n_strong)
                + [0.0] * (n_sweeps - n_noise)
            ),
            davidson_thresholds=[1e-14] * n_sweeps,
            n_sweeps=n_sweeps,
            energy_tol=1e-14,
            seed=seed,
            n_threads=n_threads,
            cutoff=cutoff,
        )


@dataclass
class ActiveSpaceHamiltonian:
    """The one set of integrals that both solvers are required to consume."""

    n_orbitals: int
    n_electrons: int
    spin: int  # 2S
    e_core_constant: float  # nuclear repulsion + frozen-core electronic energy
    e_nuclear_repulsion: float  # mol.energy_nuc() alone
    h1e: np.ndarray
    g2e: np.ndarray
    orb_sym: list[int]

    @property
    def nelec_pair(self) -> tuple[int, int]:
        """(n_alpha, n_beta) for the requested spin sector."""
        n_alpha = (self.n_electrons + self.spin) // 2
        n_beta = self.n_electrons - n_alpha
        return n_alpha, n_beta

    def digest(self) -> str:
        """SHA-256 over the integrals and constants actually handed to the solvers."""
        h = hashlib.sha256()
        for array in (np.ascontiguousarray(self.h1e, dtype=np.float64),
                      np.ascontiguousarray(self.g2e, dtype=np.float64)):
            h.update(str(array.shape).encode())
            h.update(array.tobytes())
        h.update(np.float64(self.e_core_constant).tobytes())
        h.update(f"{self.n_orbitals}:{self.n_electrons}:{self.spin}".encode())
        return h.hexdigest()


# --------------------------------------------------------------------------- #
# Result record
# --------------------------------------------------------------------------- #


@dataclass
class SolverComparison:
    """Full provenance record for one FCI/DMRG comparison."""

    # Inputs
    geometry: dict[str, Any]
    hamiltonian: dict[str, Any]
    dmrg_schedule: dict[str, Any]

    # Hamiltonian identity
    n_orbitals: int
    n_electrons: int
    spin_two_s: int
    nelec_alpha_beta: tuple[int, int]
    orb_sym: list[int]
    integral_digest: str
    nuclear_repulsion_included: bool
    e_nuclear_repulsion: float
    e_core_constant: float

    # Energies (Hartree)
    e_hf_total: float
    e_fci_total: float
    e_fci_electronic: float
    e_dmrg_total: float
    e_dmrg_electronic: float

    # Agreement
    delta_total: float
    delta_electronic: float
    tolerance: float
    dmrg_minus_fci: float
    variational_bound_respected: bool
    passed: bool

    # External published reference, where one exists for this exact system
    reference_energy: float | None
    reference_source: str | None
    reference_discrepancy: float | None

    # DMRG convergence diagnostics
    dmrg_sweep_bond_dims: list[int]
    dmrg_sweep_discarded_weights: list[float]
    dmrg_sweep_energies: list[float]
    dmrg_max_bond_dim: int
    dmrg_final_discarded_weight: float
    dmrg_n_sweeps_run: int

    # Cost (seconds, wall clock)
    walltime_scf: float
    walltime_fci: float
    walltime_dmrg: float

    provenance: dict[str, Any]

    # Symmetry-resolved structure of the converged MPS, extracted from the same
    # run that produced the energy.
    spin_squared: float | None = None
    expected_spin_squared: float | None = None
    spin_sector_verified: bool = True
    profile_sum_rule_error: float | None = None
    profile_normalisation_error: float | None = None
    sector_profile: dict[str, Any] | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=False)

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = (
            f"[{verdict}] {self.geometry['label']} / {self.hamiltonian['basis']}  "
            f"({self.n_orbitals}o, {self.n_electrons}e, 2S={self.spin_two_s})\n"
            f"    E_HF   total = {self.e_hf_total:.12f} Ha\n"
            f"    E_FCI  total = {self.e_fci_total:.12f} Ha   "
            f"electronic = {self.e_fci_electronic:.12f} Ha\n"
            f"    E_DMRG total = {self.e_dmrg_total:.12f} Ha   "
            f"electronic = {self.e_dmrg_electronic:.12f} Ha\n"
            f"    |dE| total = {self.delta_total:.3e} Ha, "
            f"electronic = {self.delta_electronic:.3e} Ha  (tol {self.tolerance:.1e})\n"
            f"    E_DMRG - E_FCI = {self.dmrg_minus_fci:+.3e} Ha "
            f"(variational bound {'ok' if self.variational_bound_respected else 'VIOLATED'})\n"
            f"    max bond dim {self.dmrg_max_bond_dim}, "
            f"final discarded weight {self.dmrg_final_discarded_weight:.3e}, "
            f"{self.dmrg_n_sweeps_run} sweeps\n"
            f"    wall: SCF {self.walltime_scf:.2f}s, FCI {self.walltime_fci:.2f}s, "
            f"DMRG {self.walltime_dmrg:.2f}s"
        )
        if self.spin_squared is not None:
            lines += (
                f"\n    <S^2> = {self.spin_squared:.2e} "
                f"(target {self.expected_spin_squared:.1f}, "
                f"{'ok' if self.spin_sector_verified else 'WRONG SECTOR'})"
            )
        if self.profile_sum_rule_error is not None:
            lines += (
                f"\n    sector profile: sum rule {self.profile_sum_rule_error:.1e}, "
                f"normalisation {self.profile_normalisation_error:.1e}"
            )
        if self.reference_energy is not None:
            lines += (
                f"\n    published    = {self.reference_energy:.12f} Ha "
                f"({self.reference_source})\n"
                f"    E_FCI - published = {self.reference_discrepancy:+.3e} Ha"
            )
        return lines


# --------------------------------------------------------------------------- #
# Stage 1-2: mean field and integrals
# --------------------------------------------------------------------------- #


def build_mean_field(geometry: Geometry, ham: HamiltonianSpec, *, verbose: int = 0):
    """Build the molecule and converge the RHF reference that defines the orbitals."""
    from pyscf import gto, scf

    mol = gto.M(
        atom=geometry.atom_spec,
        unit=geometry.unit,
        basis=ham.basis,
        charge=geometry.charge,
        spin=geometry.spin,
        symmetry=ham.point_group_symmetry,
        verbose=verbose,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = ham.scf_conv_tol
    mf.kernel()
    mf.converged_via = "diis"

    if not mf.converged:
        # Stretched, dimerised hydrogen chains defeat DIIS at a handful of
        # geometries. Second-order SCF converges them, and losing the point
        # entirely would take FCI with it -- the surrogate's target -- for a
        # reason that has nothing to do with the correlated methods.
        newton = mf.newton()
        newton.conv_tol = ham.scf_conv_tol
        newton.kernel()
        if newton.converged:
            newton.converged_via = "soscf"
            return newton

    if not mf.converged:
        raise RuntimeError(
            f"RHF did not converge for {geometry.label} with DIIS or second-order SCF"
        )
    return mf


def active_space_hamiltonian(mf, ham: HamiltonianSpec) -> ActiveSpaceHamiltonian:
    """Build the single set of MO integrals both solvers will use.

    Uses block2's own PySCF bridge so that the orbital ordering and symmetry
    labels are exactly the ones block2 expects; the same arrays are then handed
    to PySCF's FCI solver.
    """
    from pyblock2._pyscf import ao2mo as b2ao2mo

    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = b2ao2mo.get_rhf_integrals(
        mf,
        ncore=ham.n_frozen_core,
        ncas=ham.n_active_orbitals,
        pg_symm=ham.point_group_symmetry,
    )
    e_nuc = float(mf.mol.energy_nuc())

    if ham.n_frozen_core == 0 and not np.isclose(ecore, e_nuc, atol=1e-10):
        raise RuntimeError(
            "With no frozen core the integral constant must equal the nuclear "
            f"repulsion, got ecore={ecore!r} vs energy_nuc={e_nuc!r}"
        )

    return ActiveSpaceHamiltonian(
        n_orbitals=int(ncas),
        n_electrons=int(n_elec),
        spin=int(spin),
        e_core_constant=float(ecore),
        e_nuclear_repulsion=e_nuc,
        h1e=np.asarray(h1e),
        g2e=np.asarray(g2e),
        orb_sym=[int(s) for s in orb_sym],
    )


# --------------------------------------------------------------------------- #
# Stage 3-4: the two independent solvers
# --------------------------------------------------------------------------- #


def run_fci(
    asham: ActiveSpaceHamiltonian,
    *,
    conv_tol: float = 1e-14,
    nroots: int | None = None,
    max_space: int = 200,
) -> tuple[float, float]:
    """Exact diagonalisation of ``asham``. Returns (total energy, wall time).

    ``nroots`` defaults to 1 for closed-shell sectors and 4 otherwise, and the
    lowest returned root is used.

    The default is not paranoia. PySCF's single-root Davidson follows the state
    seeded from the lowest-diagonal determinant, and for an open-shell sector
    whose true ground state is not dominated by that determinant it can lock
    onto an excited root and report it as converged. On N2/STO-3G at 1.1 A in
    the Ms=1 sector it returns -107.343458537272, the *third* root; the true
    sector ground state is the doubly degenerate -107.356943001688, which
    block2 finds directly. Asking for several roots and taking the lowest
    removes the trap. ``fci.addons.fix_spin_`` does not: it converges to the
    same wrong root, because the problem is root selection, not spin
    contamination -- every root here already has <S^2> = 2.

    ``max_space`` is the Davidson subspace, raised from PySCF's default of 12.
    That default is what limits this reference at stretched geometries, and the
    symptom is systematic rather than random: converged DMRG sits *below* FCI by
    an amount that grows smoothly with the chain spacing, from 1e-12 Ha at
    R = 0.8 A to 2.5e-9 Ha at R = 3.4 A. DMRG is variational, so a converged MPS
    below FCI means the FCI Davidson has not converged, not that DMRG is wrong --
    and at R = 3.4 A the MPS is demonstrably converged, since chi = 500, 1000 and
    1500 agree to 1e-14 with a discarded weight of 1e-18.

    The cause is the near-degenerate spectrum of a nearly dissociated chain: a
    12-vector subspace is repeatedly restarted and loses the directions it needs.
    Measured at R = 3.4 A, against the converged MPS::

        conv_tol 1e-12, max_space 12 (PySCF default)   +2.52e-09 Ha   14.9 s
        conv_tol 1e-14, max_space 12                   +2.42e-11 Ha   23.3 s
        conv_tol 1e-14, max_space 30                   +3.91e-12 Ha    7.0 s
        conv_tol 1e-14, max_space 120                  +3.59e-12 Ha    6.8 s
        conv_tol 1e-14, max_space 200                   0.00e+00 Ha    8.0 s
        conv_tol 1e-14, max_space 300                   0.00e+00 Ha    7.8 s

    200 and 300 agree to every digit, which is what convergence looks like; the
    zero is by definition, since 200 is the lowest energy any method reached at
    that geometry. 30 was the default here until 2026-09-07 and left 3.9e-12 on
    the table at R = 3.4 A. The progression is *not* monotone -- 120 is worse
    than it looks next to 200 -- so a subspace is not "large enough" because a
    smaller one nearly agreed with it.

    The wider subspace is also no slower, because it converges instead of
    restarting. It is not free in memory -- the subspace holds ``max_space`` CI
    vectors, which is 1.3 GB each at H16 -- which is why it is a parameter rather
    than an unconditional maximum, and why chains beyond H14 cannot use it.

    Two further knobs were measured at R = 3.0 A and do nothing: ``pspace_size``
    (400 vs 4000) changes no digit, and ``direct_spin0`` returns energies
    identical to ``direct_spin1`` while running ~40% slower, so the singlet
    adaptation buys nothing here.

    With this setting the roles reverse: FCI becomes the better of the two
    solvers, and the residual disagreement at stretched geometries is DMRG's own
    seed dependence (1.4-2.1e-12 at R = 3.4 A, with two seeds differing by
    0.7e-12). This is exactly why Experiment 1 uses two independent solvers.
    """
    from pyscf import fci

    if nroots is None:
        nroots = 1 if asham.spin == 0 else 4

    start = time.perf_counter()
    energies, _ = fci.direct_spin1.kernel(
        asham.h1e,
        asham.g2e,
        asham.n_orbitals,
        asham.nelec_pair,
        ecore=asham.e_core_constant,
        conv_tol=conv_tol,
        max_cycle=500,
        max_space=max_space,
        nroots=nroots,
    )
    e_total = float(energies) if nroots == 1 else float(np.min(energies))
    return e_total, time.perf_counter() - start


def run_dmrg(
    asham: ActiveSpaceHamiltonian,
    schedule: DMRGSchedule,
    *,
    scratch: str | os.PathLike[str] | None = None,
    stack_mem_gb: int = 4,
    verbose: int = 0,
    post_process: Callable[[Any, Any], None] | None = None,
) -> tuple[float, dict[str, Any], float]:
    """Converge DMRG on ``asham``. Returns (total energy, sweep statistics, wall time).

    ``post_process`` is called as ``post_process(driver, ket)`` after
    convergence and before the scratch directory is torn down. It is the only
    point at which the converged MPS is still readable, so anything derived
    from the state -- reduced density matrices, symmetry-resolved spectra --
    must be extracted there rather than afterwards.
    """
    import block2
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    digest_before = asham.digest()
    owns_scratch = scratch is None
    scratch_dir = Path(tempfile.mkdtemp(prefix="block2_")) if owns_scratch else Path(scratch)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    try:
        block2.Random.rand_seed(schedule.seed)
        driver = DMRGDriver(
            scratch=str(scratch_dir),
            symm_type=SymmetryTypes.SU2,
            n_threads=schedule.n_threads,
            stack_mem=stack_mem_gb << 30,
        )
        driver.initialize_system(
            n_sites=asham.n_orbitals,
            n_elec=asham.n_electrons,
            spin=asham.spin,
            orb_sym=asham.orb_sym,
        )
        # block2 writes into the g2e array it is handed (symmetrisation and
        # unpacking happen in place). Passing copies keeps the object shared
        # with the FCI solver pristine, so the recorded integral digest still
        # describes what both solvers consumed after the run -- and so a
        # roundoff-level perturbation cannot leak from one solver to the other.
        mpo = driver.get_qc_mpo(
            h1e=np.array(asham.h1e, copy=True),
            g2e=np.array(asham.g2e, copy=True),
            ecore=asham.e_core_constant,
            iprint=verbose,
        )
        ket = driver.get_random_mps(tag="KET", bond_dim=schedule.bond_dims[0], nroots=1)
        e_total = driver.dmrg(
            mpo,
            ket,
            n_sweeps=schedule.n_sweeps,
            bond_dims=schedule.bond_dims,
            noises=schedule.noises,
            thrds=schedule.davidson_thresholds,
            tol=schedule.energy_tol,
            cutoff=schedule.cutoff,
            iprint=verbose,
        )
        if post_process is not None:
            post_process(driver, ket)
        bond_dims, dws, energies = driver.get_dmrg_results()
        stats = {
            "sweep_bond_dims": [int(b) for b in bond_dims],
            "sweep_discarded_weights": [float(w) for w in dws],
            "sweep_energies": [float(np.ravel(e)[0]) for e in energies],
        }
    finally:
        walltime = time.perf_counter() - start
        if owns_scratch:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    if asham.digest() != digest_before:
        raise RuntimeError(
            "the DMRG backend modified the shared integrals in place; the FCI/DMRG "
            "comparison would no longer be on an identical Hamiltonian"
        )
    return float(e_total), stats, walltime


# --------------------------------------------------------------------------- #
# Stage 5: the comparison
# --------------------------------------------------------------------------- #


def validate_solvers(
    geometry: Geometry,
    ham: HamiltonianSpec | None = None,
    schedule: DMRGSchedule | None = None,
    *,
    tolerance: float = 1e-8,
    reference_energy: float | None = None,
    reference_source: str | None = None,
    with_profile: bool = True,
    spin_tolerance: float = 1e-8,
    verbose: int = 0,
) -> SolverComparison:
    """Run the full Experiment 1 comparison for one geometry and return the record.

    ``reference_energy`` is an externally published total energy for this exact
    system, recorded and compared but deliberately *not* part of the pass
    criterion: the gate tests our two solvers against each other, and an
    external number can legitimately differ in its last digits.
    """
    ham = ham or HamiltonianSpec()
    schedule = schedule or DMRGSchedule.default()

    scf_start = time.perf_counter()
    mf = build_mean_field(geometry, ham, verbose=verbose)
    walltime_scf = time.perf_counter() - scf_start

    asham = active_space_hamiltonian(mf, ham)
    digest = asham.digest()

    e_fci_total, walltime_fci = run_fci(asham)

    # The converged MPS is only readable inside the solver's lifetime, so the
    # symmetry-resolved structure is taken from the same run as the energy.
    captured: dict[str, Any] = {}

    def capture(driver: Any, ket: Any) -> None:
        from tn_quantum_chemistry.entanglement import (
            profile_invariants,
            spin_square_expectation,
            symmetry_resolved_profile,
        )

        captured["spin_squared"] = spin_square_expectation(driver, ket)
        profile = symmetry_resolved_profile(driver, ket, n_orbitals=asham.n_orbitals)
        captured["profile"] = profile
        captured["invariants"] = profile_invariants(profile)

    e_dmrg_total, stats, walltime_dmrg = run_dmrg(
        asham, schedule, verbose=verbose, post_process=capture if with_profile else None
    )

    profile = captured.get("profile")
    invariants = captured.get("invariants", {})
    spin_squared = captured.get("spin_squared")
    # SU2 targets total spin S; <S^2> must come back as S(S+1).
    expected_spin_squared = (asham.spin / 2.0) * (asham.spin / 2.0 + 1.0)
    spin_ok = spin_squared is None or abs(spin_squared - expected_spin_squared) < spin_tolerance
    profile_ok = not invariants or (
        invariants["sum_rule_error"] < 1e-9 and invariants["normalisation_error"] < 1e-9
    )

    e_nuc = asham.e_nuclear_repulsion
    e_fci_elec = e_fci_total - e_nuc
    e_dmrg_elec = e_dmrg_total - e_nuc

    delta_total = abs(e_dmrg_total - e_fci_total)
    delta_elec = abs(e_dmrg_elec - e_fci_elec)
    signed = e_dmrg_total - e_fci_total

    return SolverComparison(
        geometry=asdict(geometry),
        hamiltonian=asdict(ham),
        dmrg_schedule=asdict(schedule),
        n_orbitals=asham.n_orbitals,
        n_electrons=asham.n_electrons,
        spin_two_s=asham.spin,
        nelec_alpha_beta=asham.nelec_pair,
        orb_sym=asham.orb_sym,
        integral_digest=digest,
        nuclear_repulsion_included=True,
        e_nuclear_repulsion=e_nuc,
        e_core_constant=asham.e_core_constant,
        e_hf_total=float(mf.e_tot),
        e_fci_total=e_fci_total,
        e_fci_electronic=e_fci_elec,
        e_dmrg_total=e_dmrg_total,
        e_dmrg_electronic=e_dmrg_elec,
        delta_total=delta_total,
        delta_electronic=delta_elec,
        tolerance=tolerance,
        dmrg_minus_fci=signed,
        # DMRG is variational for the exact ground state; only numerical noise
        # may push it below FCI.
        variational_bound_respected=bool(signed > -tolerance),
        passed=bool(
            delta_total < tolerance
            and delta_elec < tolerance
            and spin_ok
            and profile_ok
        ),
        reference_energy=reference_energy,
        reference_source=reference_source,
        reference_discrepancy=(
            None if reference_energy is None else e_fci_total - reference_energy
        ),
        dmrg_sweep_bond_dims=stats["sweep_bond_dims"],
        dmrg_sweep_discarded_weights=stats["sweep_discarded_weights"],
        dmrg_sweep_energies=stats["sweep_energies"],
        dmrg_max_bond_dim=max(stats["sweep_bond_dims"]),
        dmrg_final_discarded_weight=stats["sweep_discarded_weights"][-1],
        dmrg_n_sweeps_run=len(stats["sweep_energies"]),
        walltime_scf=walltime_scf,
        walltime_fci=walltime_fci,
        walltime_dmrg=walltime_dmrg,
        provenance=software_provenance(
            {"dmrg_n_threads": schedule.n_threads, "hardware_id": hardware_id()}
        ),
        spin_squared=spin_squared,
        expected_spin_squared=expected_spin_squared,
        spin_sector_verified=bool(spin_ok),
        profile_sum_rule_error=invariants.get("sum_rule_error"),
        profile_normalisation_error=invariants.get("normalisation_error"),
        sector_profile=None if profile is None else profile.to_dict(),
    )


# --------------------------------------------------------------------------- #
# Default validation suite and CLI
# --------------------------------------------------------------------------- #


#: block2's own published N2/STO-3G result at 1.1 Angstrom, from the
#: "Quantum Chemistry Hamiltonians" tutorial. Their run uses D2h symmetry and
#: the SU2 (RHF) driver; reproducing it here in C1 also checks that the energy
#: is independent of whether point-group symmetry is exploited.
BLOCK2_N2_STO3G_REFERENCE = -107.654122447524415
BLOCK2_N2_STO3G_SOURCE = (
    "block2 documentation, Quantum Chemistry Hamiltonians tutorial, "
    "N2/STO-3G at 1.1 A, SU2 DMRG"
)


@dataclass(frozen=True)
class ValidationCase:
    """One Experiment 1 case, optionally carrying an external published value."""

    geometry: Geometry
    hamiltonian: HamiltonianSpec
    reference_energy: float | None = None
    reference_source: str | None = None


def default_cases() -> list[ValidationCase]:
    """The Experiment 1 cases, ordered simplest first.

    N2 first because it is the Day-2 convention reference and the one case with
    an external published number; then H10 at a weakly and a strongly
    correlated spacing, which is the regime the rest of the project runs in.
    """
    sto3g = HamiltonianSpec(basis="sto-3g")
    return [
        ValidationCase(
            diatomic("N", 1.1, label="N2_block2_reference"),
            sto3g,
            reference_energy=BLOCK2_N2_STO3G_REFERENCE,
            reference_source=BLOCK2_N2_STO3G_SOURCE,
        ),
        ValidationCase(linear_hydrogen_chain(10, R=1.0, delta=0.0, label="H10_R1.0_weak"), sto3g),
        ValidationCase(
            linear_hydrogen_chain(10, R=2.4, delta=0.0, label="H10_R2.4_strong"), sto3g
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 1: validate block2 DMRG against PySCF FCI "
        "on identical finite Hamiltonians."
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-8,
        help="maximum accepted |E_DMRG - E_FCI| in Hartree (default: 1e-8)",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("data/raw/validation"),
        help="directory for the JSON result records",
    )
    parser.add_argument("--seed", type=int, default=1234, help="block2 MPS initialisation seed")
    parser.add_argument("--threads", type=int, default=4, help="block2 OpenMP threads")
    parser.add_argument(
        "--no-fail-fast", action="store_true",
        help="run every case even after one fails (default: stop at the first failure)",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    schedule = DMRGSchedule.default(seed=args.seed, n_threads=args.threads)
    args.outdir.mkdir(parents=True, exist_ok=True)

    results: list[SolverComparison] = []
    for case in default_cases():
        geometry = case.geometry
        print(f"\n=== {geometry.label} ===", flush=True)
        result = validate_solvers(
            geometry,
            case.hamiltonian,
            schedule,
            tolerance=args.tolerance,
            reference_energy=case.reference_energy,
            reference_source=case.reference_source,
            verbose=args.verbose,
        )
        results.append(result)
        print(result.summary(), flush=True)

        path = args.outdir / f"validation_{geometry.label}.json"
        path.write_text(result.to_json())
        print(f"    written to {path}", flush=True)

        if not result.passed and not args.no_fail_fast:
            print("\nStopping: the validation gate failed. "
                  "No downstream DMRG result is publishable until this passes.")
            break

    n_passed = sum(r.passed for r in results)
    all_passed = n_passed == len(results) == len(default_cases())
    print(f"\nGATE: {n_passed}/{len(default_cases())} cases passed "
          f"-> {'OPEN' if all_passed else 'CLOSED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
