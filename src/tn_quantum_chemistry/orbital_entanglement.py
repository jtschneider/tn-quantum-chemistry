"""Experiment 3: orbital entropies and mutual information from the converged MPS.

The bridge between this project's tensor-network background and the molecular
calculation. For each spatial orbital ``i`` the single-orbital entropy ``s_i``
measures how entangled that orbital is with the rest; the two-orbital mutual
information ``I_ij`` measures how strongly a *pair* is correlated beyond what
each carries alone. Both are standard QC-DMRG diagnostics -- Legeza and Sólyom,
Phys. Rev. B 68, 195116 (2003); Rissler, Noack and White, Chem. Phys. 323, 519
(2006); Boguslawski and Tecmer, Int. J. Quantum Chem. 115, 1289 (2015) -- and
nothing here is presented as novel.

Two routes, deliberately
------------------------
``DMRGDriver.get_orbital_entropies`` **raises in SU2 mode** in block2 0.5.4rc16
and works in SZ. So:

* ``s_i`` is computed analytically from the spin-traced 1- and 2-RDM of the
  *validated SU2* run. For a spin-adapted singlet this is exact, because spin
  symmetry fixes ``<n_i,up> = <n_i,down> = <n_i> / 2``.
* ``s_ij``, which needs the full two-orbital reduced density matrix, comes from
  an SZ calculation via block2's own routine.

The SZ state is not trusted on faith: its energy and ``<S^2>`` are checked
against the SU2 reference before its entropies are used, because an SZ MPS is
only constrained to fixed ``S_z`` and can carry weight in the wrong total-spin
sector while still looking converged in energy. The two routes overlap on
``s_i``, which is what makes the pair a check rather than two guesses -- they
agree to about 1e-7 on H10/STO-3G, limited by the SZ run's convergence.

Interpretation
--------------
These quantities are **basis-, active-space- and ordering-dependent**. They
describe entanglement between the chosen orbitals, not an invariant property of
the molecule: the same state in localised orbitals gives a different ``s_i``
profile from the same state in canonical RHF orbitals, and the DMRG site
ordering does not change converged values but does change how expensive they
are to reach. Everything here uses canonical RHF orbitals in the given basis,
with no frozen core, in the order PySCF produces them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

#: Probabilities below this are treated as zero when taking logarithms.
PROBABILITY_FLOOR = 1e-16

#: A spatial orbital has four states: empty, singly occupied (two ways), doubly
#: occupied. So ``0 <= s_i <= ln 4`` and ``0 <= s_ij <= ln 16``.
MAX_SINGLE_ORBITAL_ENTROPY = float(np.log(4.0))
MAX_TWO_ORBITAL_ENTROPY = float(np.log(16.0))


@dataclass
class OrbitalEntanglement:
    """Orbital entropies and mutual information for one converged state."""

    n_orbitals: int
    single_orbital_entropy: list[float]
    two_orbital_entropy: list[list[float]]
    mutual_information: list[list[float]]
    occupations: list[list[float]]
    #: The same state's ``s_i`` in symmetrically orthogonalised AOs, when the
    #: caller supplies that Hamiltonian. Present so the basis dependence is a
    #: recorded number rather than a caveat in prose.
    localised_single_orbital_entropy: list[float] | None = None
    localised_occupations: list[list[float]] | None = None
    invariants: dict[str, Any] = field(default_factory=dict)
    cross_check: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def total_single_orbital_entropy(self) -> float:
        """``sum_i s_i``, an upper bound on the true multi-orbital entanglement."""
        return float(np.sum(self.single_orbital_entropy))

    def strongest_pairs(self, count: int = 5) -> list[tuple[int, int, float]]:
        """The most strongly correlated orbital pairs, by mutual information."""
        matrix = np.asarray(self.mutual_information)
        pairs = [
            (i, j, float(matrix[i, j]))
            for i in range(self.n_orbitals)
            for j in range(i + 1, self.n_orbitals)
        ]
        return sorted(pairs, key=lambda item: -item[2])[:count]


# --------------------------------------------------------------------------- #
# Single-orbital quantities from spin-traced RDMs
# --------------------------------------------------------------------------- #


def single_orbital_occupations(dm1: np.ndarray, dm2: np.ndarray) -> np.ndarray:
    """Occupation probabilities of each spatial orbital: (n_orbitals, 4).

    Columns are (empty, up, down, doubly occupied). For a spin-adapted singlet,
    spin symmetry gives ``<n_i,up> = <n_i,down> = <n_i>/2``, and the double
    occupancy follows from the spin-traced 2-RDM:

        <n_i> = dm1[i, i]
        D_i   = <n_i,up n_i,down> = dm2[i, i, i, i] / 2

    using block2's convention ``dm2[i,j,k,l] = <a+_i a+_j a_k a_l>``, under which
    ``sum_ij dm2[i,j,j,i] = N(N-1)``. Since ``n^2 = n`` for fermions,
    ``<n_i^2> = <n_i> + 2 D_i``, which is where the factor of two comes from.
    """
    occupation = np.diag(np.asarray(dm1)).astype(float)
    double = np.einsum("iiii->i", np.asarray(dm2)).astype(float) / 2.0
    probabilities = np.stack(
        [
            1.0 - occupation + double,
            occupation / 2.0 - double,
            occupation / 2.0 - double,
            double,
        ],
        axis=1,
    )
    return probabilities


def entropy_from_probabilities(probabilities: np.ndarray, axis: int = -1) -> np.ndarray:
    """von Neumann entropy of a probability vector along ``axis``."""
    clipped = np.clip(np.asarray(probabilities, dtype=float), PROBABILITY_FLOOR, None)
    return -np.sum(clipped * np.log(clipped), axis=axis)


def single_orbital_entropy(dm1: np.ndarray, dm2: np.ndarray) -> np.ndarray:
    """``s_i`` for every spatial orbital, from spin-traced RDMs."""
    return entropy_from_probabilities(single_orbital_occupations(dm1, dm2))


# --------------------------------------------------------------------------- #
# Mutual information
# --------------------------------------------------------------------------- #


def mutual_information(
    s_single: np.ndarray, s_two: np.ndarray, *, halved: bool = True
) -> np.ndarray:
    """``I_ij = (s_i + s_j - s_ij) / 2`` for ``i != j``, zero on the diagonal.

    The factor of one half follows Legeza and Boguslawski; part of the
    literature omits it. It is stated here rather than left implicit because the
    two conventions differ by exactly a factor of two and both appear in print.
    """
    s_single = np.asarray(s_single, dtype=float)
    s_two = np.asarray(s_two, dtype=float)
    matrix = s_single[:, None] + s_single[None, :] - s_two
    if halved:
        matrix = 0.5 * matrix
    np.fill_diagonal(matrix, 0.0)
    return matrix


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #


def rdm_invariants(
    dm1: np.ndarray, dm2: np.ndarray, n_electrons: int
) -> dict[str, Any]:
    """Residuals of the identities a valid pair of reduced density matrices obeys.

    Checks the properties the experiment description names: traces, Hermiticity,
    positivity and entropy bounds. Reported as residuals rather than booleans so
    a caller can see how close to the edge a result sits.
    """
    dm1 = np.asarray(dm1, dtype=float)
    dm2 = np.asarray(dm2, dtype=float)
    occupations = np.linalg.eigvalsh(dm1)
    probabilities = single_orbital_occupations(dm1, dm2)
    entropies = entropy_from_probabilities(probabilities)

    return {
        "trace_1rdm": float(np.trace(dm1)),
        "trace_1rdm_error": float(abs(np.trace(dm1) - n_electrons)),
        "hermiticity_error_1rdm": float(np.max(np.abs(dm1 - dm1.T))),
        # Spatial-orbital natural occupations lie in [0, 2].
        "min_natural_occupation": float(occupations.min()),
        "max_natural_occupation": float(occupations.max()),
        "occupation_bound_violation": float(
            max(0.0, -occupations.min(), occupations.max() - 2.0)
        ),
        "trace_2rdm": float(np.einsum("ijji->", dm2)),
        "trace_2rdm_error": float(
            abs(np.einsum("ijji->", dm2) - n_electrons * (n_electrons - 1))
        ),
        # Positivity of the one-orbital RDM: its eigenvalues are the four
        # occupation probabilities, so a negative entry is a genuine violation.
        "min_orbital_probability": float(probabilities.min()),
        "probability_normalisation_error": float(
            np.max(np.abs(probabilities.sum(axis=1) - 1.0))
        ),
        "max_single_orbital_entropy": float(entropies.max()),
        "entropy_bound_violation": float(
            max(0.0, -entropies.min(), entropies.max() - MAX_SINGLE_ORBITAL_ENTROPY)
        ),
    }


def entanglement_invariants(
    s_single: np.ndarray, s_two: np.ndarray, information: np.ndarray
) -> dict[str, Any]:
    """Residuals of the inequalities two-orbital entropies must satisfy."""
    s_single = np.asarray(s_single, dtype=float)
    s_two = np.asarray(s_two, dtype=float)
    information = np.asarray(information, dtype=float)
    offdiag = ~np.eye(len(s_single), dtype=bool)

    pairwise_sum = s_single[:, None] + s_single[None, :]
    pairwise_difference = np.abs(s_single[:, None] - s_single[None, :])
    return {
        # Subadditivity: s_ij <= s_i + s_j, equivalently I_ij >= 0.
        "subadditivity_violation": float(
            max(0.0, np.max((s_two - pairwise_sum)[offdiag]))
        ),
        # Araki-Lieb: s_ij >= |s_i - s_j|.
        "araki_lieb_violation": float(
            max(0.0, np.max((pairwise_difference - s_two)[offdiag]))
        ),
        "min_mutual_information": float(information[offdiag].min()),
        "max_mutual_information": float(information[offdiag].max()),
        "two_orbital_bound_violation": float(
            max(0.0, np.max(s_two[offdiag]) - MAX_TWO_ORBITAL_ENTROPY)
        ),
        "mutual_information_symmetry_error": float(
            np.max(np.abs(information - information.T))
        ),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def lowdin_hamiltonian(mf, ham):
    """The same Hamiltonian expressed in symmetrically orthogonalised AOs.

    ``X = S^(-1/2)`` turns the atomic orbitals into an orthonormal, maximally
    *localised* set: for a hydrogen chain each resulting orbital sits on one
    atom. Since this is a unitary change of orbital basis, the FCI energy is
    unchanged -- but ``s_i`` is not, which is the point.

    Orbital entropies are properties of a partition of the Hilbert space into
    orbitals, so they describe the orbitals as much as the state. Reporting them
    in one basis alone invites reading a feature of the representation as a
    feature of the molecule. On H10 at R = 2.6 A the mean ``s_i`` is 1.336 in
    canonical RHF orbitals (96% of the ln 4 ceiling) and 0.741 in this basis
    (7% above ln 2) for the identical state.

    Only the all-electron case is supported: a frozen core is defined by the
    canonical orbitals and does not survive the transformation.
    """
    import dataclasses

    from pyscf import ao2mo

    from tn_quantum_chemistry.validation import active_space_hamiltonian

    if ham.n_frozen_core != 0:
        raise ValueError(
            "Löwdin localisation is only defined here for the all-electron case; "
            "a frozen core is specified in terms of the canonical orbitals"
        )

    overlap = mf.mol.intor("int1e_ovlp")
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    x = eigenvectors @ np.diag(eigenvalues**-0.5) @ eigenvectors.T

    n_orbitals = x.shape[1]
    canonical = active_space_hamiltonian(mf, ham)
    return dataclasses.replace(
        canonical,
        h1e=x.T @ mf.get_hcore() @ x,
        g2e=ao2mo.restore(1, ao2mo.kernel(mf.mol, x), n_orbitals),
    )


def su2_reduced_density_matrices(asham, schedule=None):
    """Converge the validated SU2 MPS and return (energy, dm1, dm2)."""
    from tn_quantum_chemistry.validation import DMRGSchedule, run_dmrg

    captured: dict[str, Any] = {}

    def capture(driver: Any, ket: Any) -> None:
        captured["dm1"] = driver.get_1pdm(ket)
        captured["dm2"] = driver.get_2pdm(ket)

    energy, _, _ = run_dmrg(
        asham, schedule or DMRGSchedule.default(), post_process=capture
    )
    return energy, captured["dm1"], captured["dm2"]


def sz_orbital_entropies(
    asham, *, bond_dim: int = 1500, seed: int = 1234, n_threads: int = 4
):
    """Converge an SZ MPS and take block2's orbital entropies from it.

    SZ is used only because ``get_orbital_entropies`` raises in SU2 mode in this
    block2 release. The returned energy and ``<S^2>`` let the caller confirm the
    state is the intended singlet before trusting anything derived from it --
    an SZ MPS fixes only ``S_z`` and can converge into the wrong total-spin
    sector while its energy still looks plausible.
    """
    import tempfile

    import block2
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    block2.Random.rand_seed(seed)
    driver = DMRGDriver(
        scratch=tempfile.mkdtemp(prefix="block2_sz_"),
        symm_type=SymmetryTypes.SZ,
        n_threads=n_threads,
        stack_mem=8 << 30,
    )
    driver.initialize_system(
        n_sites=asham.n_orbitals,
        n_elec=asham.n_electrons,
        spin=asham.spin,
        orb_sym=asham.orb_sym,
    )
    mpo = driver.get_qc_mpo(
        h1e=(asham.h1e.copy(), asham.h1e.copy()),
        g2e=(asham.g2e.copy(), asham.g2e.copy(), asham.g2e.copy()),
        ecore=asham.e_core_constant,
        iprint=0,
    )
    n_sweeps = 30
    start = max(bond_dim // 3, 200)
    ket = driver.get_random_mps(tag="SZ-ORBENT", bond_dim=start, nroots=1)
    energy = driver.dmrg(
        mpo, ket, n_sweeps=n_sweeps,
        bond_dims=[start] * 10 + [bond_dim] * (n_sweeps - 10),
        noises=[1e-4] * 10 + [1e-6] * 6 + [0.0] * (n_sweeps - 16),
        thrds=[1e-12] * n_sweeps, tol=1e-12, iprint=0,
    )
    spin_squared = float(
        driver.expectation(ket, driver.get_spin_square_mpo(iprint=0), ket)
    )
    s_single = np.asarray(driver.get_orbital_entropies(ket, orb_type=1)).ravel()
    s_two = np.asarray(driver.get_orbital_entropies(ket, orb_type=2))
    return float(energy), s_single, s_two, spin_squared


def compute_orbital_entanglement(
    asham,
    *,
    localised_asham=None,
    schedule=None,
    sz_bond_dim: int = 1500,
    energy_tolerance: float = 1e-6,
    spin_tolerance: float = 1e-6,
    cross_check_tolerance: float = 1e-4,
    n_threads: int = 4,
) -> OrbitalEntanglement:
    """Orbital entropies and mutual information, with both routes cross-checked.

    Raises if the SZ state used for ``s_ij`` does not reproduce the SU2 energy,
    is not in the intended spin sector, or disagrees with the analytic ``s_i``.
    Failing loudly is the point: every quantity here is a property of a state,
    so a quietly different state gives quietly wrong entanglement.
    """
    su2_energy, dm1, dm2 = su2_reduced_density_matrices(asham, schedule)
    s_analytic = single_orbital_entropy(dm1, dm2)

    sz_energy, s_block2, s_two, spin_squared = sz_orbital_entropies(
        asham, bond_dim=sz_bond_dim, n_threads=n_threads
    )

    expected_spin_squared = (asham.spin / 2.0) * (asham.spin / 2.0 + 1.0)
    energy_gap = abs(sz_energy - su2_energy)
    spin_gap = abs(spin_squared - expected_spin_squared)
    entropy_gap = float(np.max(np.abs(s_analytic - s_block2)))

    if energy_gap > energy_tolerance:
        raise RuntimeError(
            f"the SZ state used for two-orbital entropies is not the SU2 state: "
            f"energies differ by {energy_gap:.2e} Ha"
        )
    if spin_gap > spin_tolerance:
        raise RuntimeError(
            f"the SZ state is not in the target spin sector: <S^2> = "
            f"{spin_squared:.3e}, expected {expected_spin_squared:.1f}"
        )
    if entropy_gap > cross_check_tolerance:
        raise RuntimeError(
            f"single-orbital entropies disagree between the analytic and block2 "
            f"routes by {entropy_gap:.2e}"
        )

    localised_entropy = localised_occupations = None
    if localised_asham is not None:
        localised_energy, localised_dm1, localised_dm2 = su2_reduced_density_matrices(
            localised_asham, schedule
        )
        # A unitary change of orbital basis cannot move the energy. If it has,
        # the localised Hamiltonian is not the same Hamiltonian.
        if abs(localised_energy - su2_energy) > energy_tolerance:
            raise RuntimeError(
                f"the localised basis changed the energy by "
                f"{abs(localised_energy - su2_energy):.2e} Ha; an orbital "
                f"rotation must leave it invariant"
            )
        localised_entropy = [
            float(x) for x in single_orbital_entropy(localised_dm1, localised_dm2)
        ]
        localised_occupations = [
            [float(x) for x in row]
            for row in single_orbital_occupations(localised_dm1, localised_dm2)
        ]

    information = mutual_information(s_analytic, s_two)
    return OrbitalEntanglement(
        n_orbitals=asham.n_orbitals,
        single_orbital_entropy=[float(x) for x in s_analytic],
        two_orbital_entropy=[[float(x) for x in row] for row in s_two],
        mutual_information=[[float(x) for x in row] for row in information],
        occupations=[
            [float(x) for x in row] for row in single_orbital_occupations(dm1, dm2)
        ],
        localised_single_orbital_entropy=localised_entropy,
        localised_occupations=localised_occupations,
        invariants={
            **rdm_invariants(dm1, dm2, asham.n_electrons),
            **entanglement_invariants(s_analytic, s_two, information),
        },
        cross_check={
            "su2_energy": su2_energy,
            "sz_energy": sz_energy,
            "energy_difference": energy_gap,
            "spin_squared": spin_squared,
            "expected_spin_squared": expected_spin_squared,
            "single_orbital_entropy_difference": entropy_gap,
            "sz_bond_dim": sz_bond_dim,
        },
    )
