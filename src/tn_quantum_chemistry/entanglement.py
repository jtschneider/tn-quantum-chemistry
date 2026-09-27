"""Symmetry-resolved bipartition data from a converged block2 MPS.

The DMRG solver already works in a symmetry-adapted basis: every virtual bond
carries a definite ``(N, 2S, pg)`` label and the Schmidt spectrum is block
diagonal in it. This module reads that structure back out, which costs nothing
beyond the run already performed and turns the solver's internal bookkeeping
into physical observables.

For a cut ``l`` between orbitals ``l`` and ``l+1``, let ``s[l, q, a]`` be the
Schmidt values in sector ``q``. The module returns

    p[l, q] = sum_a s[l, q, a]^2                with  sum_q p[l, q] = 1

together with the quantities derived from it: the bipartite entanglement, the
number entropy, and the mean and variance of the particle number on the left of
the cut.

The SU(2) multiplicity convention
---------------------------------
block2's SU(2) tensors hold *reduced* matrix elements: one entry per multiplet
rather than per state. The stored singular values are normalised so that
``sum_q sum_a s^2 = 1`` with no degeneracy factor, but the physical reduced
density matrix has eigenvalues ``s^2 / (2S+1)``, each appearing ``2S+1`` times.
The von Neumann entropy is therefore

    S = -sum_q (2S_q + 1) * w * log w      with  w = s^2 / (2S_q + 1)

which is *not* the Shannon entropy of the reduced spectrum. Getting this wrong
is a silent error: it produces a plausible, smooth, symmetric profile that is
simply too small.

``DMRGDriver.get_bipartite_entanglement`` applies the naive Shannon entropy to
the flattened reduced spectrum and so under-reports SU(2) entanglement -- on
H10/STO-3G at R = 2.4 A it returns 4.46886 at the central cut where the true
value is 4.59413. The convention implemented here was fixed by reproducing an
independent abelian ``SZ`` calculation of the same state, where no multiplicity
factor arises; :func:`verify_against_driver` re-runs that comparison.

In ``SZ`` mode every sector is one state and the multiplicity is 1, so the same
code path serves both symmetry types.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np

#: Numerical floor below which a Schmidt weight is treated as absent.
WEIGHT_FLOOR = 1e-14

#: Renyi orders retained as features. ``1`` is the von Neumann entropy and
#: ``inf`` is ``-log`` of the largest Schmidt weight. Together they describe the
#: *shape* of the spectral decay rather than a single moment of it, while
#: remaining symmetric functions of the spectrum and therefore smooth in the
#: geometry -- which individual sorted Schmidt values are not.
RENYI_ALPHAS: tuple[float, ...] = (0.5, 1.0, 2.0, float("inf"))


def renyi_entropy(weights: np.ndarray, alpha: float) -> float:
    """Renyi entropy of order ``alpha`` for a normalised probability vector."""
    w = np.asarray(weights, dtype=float)
    w = w[w > WEIGHT_FLOOR]
    if w.size == 0:
        return 0.0
    w = w / w.sum()
    if np.isinf(alpha):
        return float(-np.log(w.max()))
    if abs(alpha - 1.0) < 1e-12:
        return float(-np.sum(w * np.log(w)))
    return float(np.log(np.sum(w**alpha)) / (1.0 - alpha))


def expand_multiplets(
    cut: dict[Sector, np.ndarray], *, is_su2: bool = True
) -> np.ndarray:
    """Physical reduced-density-matrix eigenvalues for one cut, sorted descending.

    Each SU(2) multiplet contributes ``2S+1`` copies of ``s^2 / (2S+1)``. The
    result is a genuine probability vector over physical states, which is what
    every spectral functional below expects.
    """
    if not cut:
        return np.zeros(0)
    parts = [
        np.repeat(values / (sector.multiplicity if is_su2 else 1),
                  sector.multiplicity if is_su2 else 1)
        for sector, values in cut.items()
    ]
    eigenvalues = np.concatenate(parts)
    total = eigenvalues.sum()
    if total > 0:
        eigenvalues = eigenvalues / total
    return np.sort(eigenvalues)[::-1]


def encode_sector_key(key: tuple[int, int, int]) -> str:
    """``(N, 2S, pg)`` -> ``"N,2S,pg"``, because JSON cannot key on tuples."""
    return ",".join(str(int(part)) for part in key)


def decode_sector_key(key: str) -> tuple[int, int, int]:
    """Inverse of :func:`encode_sector_key`."""
    n, twos, pg = (int(part) for part in key.split(","))
    return (n, twos, pg)


@dataclass(frozen=True)
class Sector:
    """A bond quantum-number label: particle number, twice the spin, irrep."""

    n: int
    twos: int
    pg: int

    @property
    def multiplicity(self) -> int:
        """Number of physical states this label stands for under SU(2)."""
        return self.twos + 1

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.n, self.twos, self.pg)

    def __str__(self) -> str:
        return f"N={self.n},2S={self.twos},pg={self.pg}"


def charge_resolved_spectra(
    cut: dict[Sector, np.ndarray], *, is_su2: bool = True
) -> dict[int, np.ndarray]:
    """Multiplet-expanded eigenvalues grouped by particle number ``N``.

    Each array sums to ``p[l, N]`` rather than to one, so the charge weights and
    the within-sector spectra are both recoverable from the result.
    """
    grouped: dict[int, list[np.ndarray]] = {}
    for sector, values in cut.items():
        degeneracy = sector.multiplicity if is_su2 else 1
        grouped.setdefault(sector.n, []).append(
            np.repeat(values / degeneracy, degeneracy)
        )
    total = sum(float(np.concatenate(v).sum()) for v in grouped.values())
    if total <= 0.0:
        return {}
    return {n: np.concatenate(v) / total for n, v in grouped.items()}


def symmetry_resolved_entropies(
    cut: dict[Sector, np.ndarray], alpha: float = 1.0, *, is_su2: bool = True
) -> tuple[dict[int, float], dict[int, float]]:
    """Charge weights ``p[l, N]`` and the entanglement *within* each charge sector.

    Returns ``(weights, entropies)``. The sector entropies are computed from
    each sector's spectrum renormalised to one, so they measure entanglement
    that is not accounted for by the charge distribution itself.

    For ``alpha == 1`` this yields the exact decomposition

        S_total = S_number + sum_N p[l, N] * S_N

    where the first term is the number entropy and the second the
    configurational entropy. The identity is a strong check on the whole
    pipeline: it ties the pooled spectrum, the charge marginal and the
    per-sector spectra together, and no one of them can be wrong on its own
    while it holds. It is exact only for von Neumann entropy; for other Renyi
    orders the weighted average below is still well defined but obeys no such
    additive rule.
    """
    grouped = charge_resolved_spectra(cut, is_su2=is_su2)
    weights = {n: float(v.sum()) for n, v in grouped.items()}
    entropies = {n: renyi_entropy(v, alpha) for n, v in grouped.items()}
    return weights, entropies


def configurational_entropy(
    cut: dict[Sector, np.ndarray], alpha: float = 1.0, *, is_su2: bool = True
) -> float:
    """``sum_N p[l, N] * S_N``: entanglement remaining once charge is known."""
    weights, entropies = symmetry_resolved_entropies(cut, alpha, is_su2=is_su2)
    return float(sum(weights[n] * entropies[n] for n in weights))


@dataclass
class SectorProfile:
    """Symmetry-resolved bipartition data for one converged MPS.

    ``sector_weights[l]`` maps a sector to ``p[l, q]``; every other field is
    derived from it and stored so that a cached record needs no recomputation.
    """

    n_orbitals: int
    symmetry: str
    is_su2: bool
    sector_weights: list[dict[tuple[int, int, int], float]]
    entanglement: list[float]
    number_entropy: list[float]
    mean_n_left: list[float]
    var_n_left: list[float]
    n_sectors: list[int]
    normalisation: list[float]
    bond_dimensions: list[int]
    #: Renyi order (as a string key) -> value per cut, for the pooled spectrum.
    renyi_entropies: dict[str, list[float]] = field(default_factory=dict)
    #: Renyi order -> per-cut ``sum_N p[l, N] * S_N``, the entanglement left
    #: once the charge sector is known. With :attr:`number_entropy` this is the
    #: symmetry-resolved decomposition of :attr:`entanglement`.
    configurational_entropies: dict[str, list[float]] = field(default_factory=dict)
    #: The full sector-resolved Schmidt spectrum, ``s^2`` per sector per cut.
    #: Retained because it is free -- the SVD has already been done -- and
    #: because discarding it means re-running DMRG to try any new spectral
    #: feature later.
    sector_spectra: list[dict[tuple[int, int, int], list[float]]] = field(
        default_factory=list
    )
    extra: dict[str, Any] = field(default_factory=dict)

    def schmidt_values(self, cut: int) -> np.ndarray:
        """Physical eigenvalues at ``cut``, multiplet-expanded and sorted."""
        stored = self.sector_spectra[cut]
        rebuilt = {
            Sector(*key): np.asarray(values, dtype=float)
            for key, values in stored.items()
        }
        return expand_multiplets(rebuilt, is_su2=self.is_su2)

    def effective_rank(self, cut: int, fraction: float = 0.99) -> int:
        """How many Schmidt values carry ``fraction`` of the weight.

        Reported as a diagnostic, not used as a feature: it is a step function
        of the geometry and therefore not differentiable.
        """
        cumulative = np.cumsum(self.schmidt_values(cut))
        return int(np.searchsorted(cumulative, fraction) + 1)

    @property
    def n_cuts(self) -> int:
        return len(self.sector_weights)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation. Sector tuples become ``"N,2S,pg"`` strings."""
        data = asdict(self)
        data["sector_weights"] = [
            {encode_sector_key(key): weight for key, weight in cut.items()}
            for cut in self.sector_weights
        ]
        data["sector_spectra"] = [
            {encode_sector_key(key): list(values) for key, values in cut.items()}
            for cut in self.sector_spectra
        ]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SectorProfile:
        """Inverse of :meth:`to_dict`, restoring tuple-keyed sector maps."""
        restored = dict(data)
        restored["sector_weights"] = [
            {decode_sector_key(key): float(weight) for key, weight in cut.items()}
            for cut in data.get("sector_weights", [])
        ]
        restored["sector_spectra"] = [
            {decode_sector_key(key): [float(v) for v in values]
             for key, values in cut.items()}
            for cut in data.get("sector_spectra", [])
        ]
        known = {f.name for f in fields(cls)}
        unknown = set(restored) - known
        if unknown:
            raise ValueError(f"unknown SectorProfile fields: {sorted(unknown)}")
        return cls(**restored)

    def number_distribution(self) -> list[dict[int, float]]:
        """``p[l, N]``, the sector weights marginalised over spin and irrep."""
        out = []
        for cut in self.sector_weights:
            marginal: dict[int, float] = {}
            for (n, _, _), w in cut.items():
                marginal[n] = marginal.get(n, 0.0) + w
            out.append(marginal)
        return out

    def summary(self) -> str:
        header = (
            f"{'cut':>4} {'sectors':>8} {'chi':>6} {'S':>9} {'S_number':>9} "
            f"{'<N_L>':>8} {'var N_L':>8}"
        )
        rows = [
            f"{i:>4} {self.n_sectors[i]:>8} {self.bond_dimensions[i]:>6} "
            f"{self.entanglement[i]:>9.5f} {self.number_entropy[i]:>9.5f} "
            f"{self.mean_n_left[i]:>8.4f} {self.var_n_left[i]:>8.4f}"
            for i in range(self.n_cuts)
        ]
        return "\n".join([header, "-" * len(header), *rows])


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def extract_sector_spectra(driver, ket) -> list[dict[Sector, np.ndarray]]:
    """Schmidt weights ``s^2`` per sector at every bond of a converged MPS.

    The MPS is exported to labelled NumPy blocks, swept right-to-left into
    right-canonical form, then swept back left-to-right. At each step the
    orthogonality centre is decomposed sector by sector, so the singular values
    obtained at bond ``l`` are genuine Schmidt values for the cut after orbital
    ``l``.

    ``ket`` is not modified: it is converted through a single-site copy.
    """
    from pyblock2.algebra.io import MPSTools

    # from_block2 rejects the two-site MPS that dmrg leaves behind, with a bare
    # assertion rather than a diagnostic, so normalise the form first.
    single_site = driver.adjust_mps(ket, dot=1)[0]
    exported = MPSTools.from_block2(single_site)

    tensors = [t.copy() for t in exported.tensors]
    n_sites = len(tensors)

    for i in range(n_sites - 1, 0, -1):
        left_blocks = tensors[i].right_canonicalize()
        absorbed = tensors[i - 1].right_multiply(left_blocks)
        if absorbed is not None:
            tensors[i - 1] = absorbed

    spectra: list[dict[Sector, np.ndarray]] = []
    for i in range(n_sites - 1):
        by_right_label: dict[Any, list] = {}
        for block in tensors[i].blocks:
            by_right_label.setdefault(block.q_labels[-1], []).append(block)

        cut: dict[Sector, np.ndarray] = {}
        for label, blocks in by_right_label.items():
            matrix = np.concatenate(
                [
                    b.reduced.reshape(int(np.prod(b.reduced.shape[:-1])), -1)
                    for b in blocks
                ],
                axis=0,
            )
            values = np.linalg.svd(matrix, compute_uv=False) ** 2
            sector = Sector(n=int(label.n), twos=int(label.twos), pg=int(label.pg))
            cut[sector] = values[values > WEIGHT_FLOOR]
        spectra.append(cut)

        tensors[i], right_blocks, _ = tensors[i].left_compress()
        absorbed = tensors[i + 1].left_multiply(right_blocks)
        if absorbed is not None:
            tensors[i + 1] = absorbed

    return spectra


def build_profile(
    spectra: list[dict[Sector, np.ndarray]],
    *,
    n_orbitals: int,
    symmetry: str,
    is_su2: bool,
) -> SectorProfile:
    """Derive the physical quantities from raw sector spectra."""
    weights, entanglement, number_entropy = [], [], []
    mean_n, var_n, n_sectors, norms, bond_dims = [], [], [], [], []
    renyi: dict[str, list[float]] = {str(a): [] for a in RENYI_ALPHAS}
    configurational: dict[str, list[float]] = {str(a): [] for a in RENYI_ALPHAS}
    stored_spectra: list[dict[tuple[int, int, int], list[float]]] = []

    for cut in spectra:
        total = float(sum(float(v.sum()) for v in cut.values()))
        if total <= 0.0:
            raise ValueError("a bond carries no Schmidt weight; the MPS is not valid")

        entropy = 0.0
        dimension = 0
        for sector, values in cut.items():
            degeneracy = sector.multiplicity if is_su2 else 1
            # Physical RDM eigenvalues: each multiplet member carries w.
            w = values / total / degeneracy
            w = w[w > WEIGHT_FLOOR]
            entropy -= degeneracy * float(np.sum(w * np.log(w)))
            dimension += degeneracy * len(values)

        cut_weights = {s.as_tuple(): float(v.sum()) / total for s, v in cut.items()}
        marginal: dict[int, float] = {}
        for (n, _, _), w in cut_weights.items():
            marginal[n] = marginal.get(n, 0.0) + w

        probabilities = np.array([p for p in marginal.values() if p > WEIGHT_FLOOR])
        counts = np.array([n for n, p in marginal.items() if p > WEIGHT_FLOOR], float)
        mean = float(np.sum(counts * probabilities))

        eigenvalues = expand_multiplets(cut, is_su2=is_su2)
        for alpha in RENYI_ALPHAS:
            renyi[str(alpha)].append(renyi_entropy(eigenvalues, alpha))
            configurational[str(alpha)].append(
                configurational_entropy(cut, alpha, is_su2=is_su2)
            )
        stored_spectra.append(
            {s.as_tuple(): [float(x) for x in v] for s, v in cut.items()}
        )

        weights.append(cut_weights)
        entanglement.append(entropy)
        number_entropy.append(-float(np.sum(probabilities * np.log(probabilities))))
        mean_n.append(mean)
        var_n.append(float(np.sum(counts**2 * probabilities)) - mean**2)
        n_sectors.append(len(cut))
        norms.append(total)
        bond_dims.append(dimension)

    return SectorProfile(
        n_orbitals=n_orbitals,
        symmetry=symmetry,
        is_su2=is_su2,
        sector_weights=weights,
        entanglement=entanglement,
        number_entropy=number_entropy,
        mean_n_left=mean_n,
        var_n_left=var_n,
        n_sectors=n_sectors,
        normalisation=norms,
        bond_dimensions=bond_dims,
        renyi_entropies=renyi,
        configurational_entropies=configurational,
        sector_spectra=stored_spectra,
    )


def symmetry_resolved_profile(driver, ket, *, n_orbitals: int | None = None) -> SectorProfile:
    """Full symmetry-resolved profile for a converged MPS."""
    from pyblock2.driver.core import SymmetryTypes

    symm = driver.symm_type
    is_su2 = bool(symm & SymmetryTypes.SU2)
    spectra = extract_sector_spectra(driver, ket)
    return build_profile(
        spectra,
        n_orbitals=n_orbitals if n_orbitals is not None else len(spectra) + 1,
        symmetry="SU2" if is_su2 else "SZ",
        is_su2=is_su2,
    )


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


def sector_tail_weight(
    spectra: list[dict[Sector, np.ndarray]], keep: int, *, is_su2: bool = True
) -> list[dict[tuple[int, int, int], float]]:
    """Weight per sector falling outside the ``keep`` largest Schmidt values.

    Ranking is global across sectors, matching how DMRG actually truncates, so
    this measures *where* a bond-dimension cut would lose weight rather than
    merely how much. It is computable from a cheap low-bond-dimension run and is
    the motivated candidate feature for predicting the finite-chi correction.
    """
    out = []
    for cut in spectra:
        ranked = sorted(
            (
                (float(value), sector)
                for sector, values in cut.items()
                for value in values
                for _ in range(sector.multiplicity if is_su2 else 1)
            ),
            key=lambda item: -item[0],
        )
        total = sum(value for value, _ in ranked)
        tails: dict[tuple[int, int, int], float] = {}
        for value, sector in ranked[keep:]:
            key = sector.as_tuple()
            tails[key] = tails.get(key, 0.0) + value / total
        out.append(tails)
    return out


FEATURE_NAMES_PER_CUT = (
    *(f"renyi_{a}" for a in RENYI_ALPHAS),
    "number_entropy",
    "configurational_entropy",
    "var_n_left",
)


def feature_vector(profile: SectorProfile) -> np.ndarray:
    """Flat, fixed-length features for a surrogate.

    Uses *symmetric functionals* of the Schmidt spectrum -- Renyi entropies at
    several orders -- rather than the raw sorted spectrum, for three reasons
    that the stored spectra make concrete:

    1. **Smoothness.** Individual sorted Schmidt values are not smooth in the
       geometry. In the strongly correlated regime they are near-degenerate
       (on H10 at R = 2.4 A the second through sixth eigenvalues lie within 2%
       of one another), so they exchange rank under small geometry changes and
       the k-th largest develops kinks. Renyi entropies are invariant to
       ordering and stay smooth -- which matters because the surrogate is meant
       to be differentiated.
    2. **No stable layout.** The number of significant Schmidt values varies by
       an order of magnitude over the surface: 3 values carry 90% of the weight
       at R = 1.0 A, 123 at R = 2.4 A. A fixed top-k truncation therefore means
       something different at each geometry.
    3. **Dimension.** The full spectrum is thousands of numbers per geometry
       against a training set of a few hundred points.

    The Renyi orders above are functionals of the *pooled* spectrum. The
    symmetry resolution enters through the last two entries: the number entropy
    and the configurational entropy, which together decompose the von Neumann
    entanglement exactly into "how is charge distributed across the cut" and
    "how entangled is the state once the charge is known".

    The spectra are still stored in full and sector-labelled, so a different
    featurisation can be tried later without re-running the solver. This
    function is the default, not the only option.
    """
    orders = [np.asarray(profile.renyi_entropies[str(a)], float) for a in RENYI_ALPHAS]
    return np.concatenate(
        [
            *orders,
            np.asarray(profile.number_entropy, dtype=float),
            np.asarray(profile.configurational_entropies["1.0"], dtype=float),
            np.asarray(profile.var_n_left, dtype=float),
        ]
    )


def feature_names(n_cuts: int) -> list[str]:
    return [f"{name}_cut{i}" for name in FEATURE_NAMES_PER_CUT for i in range(n_cuts)]


def raw_spectrum_features(profile: SectorProfile, top_k: int = 16) -> np.ndarray:
    """The ``top_k`` largest log-Schmidt weights per cut, zero-padded.

    Provided so the raw-spectrum option can actually be *tested* against the
    smooth featurisation rather than argued about. Logs are used because the
    weights span many orders of magnitude. Expect rank-exchange kinks; that is
    the point of the comparison.
    """
    rows = []
    for cut in range(profile.n_cuts):
        values = profile.schmidt_values(cut)[:top_k]
        padded = np.full(top_k, WEIGHT_FLOOR)
        padded[: values.size] = np.maximum(values, WEIGHT_FLOOR)
        rows.append(np.log(padded))
    return np.concatenate(rows)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def verify_against_driver(driver, ket, profile: SectorProfile) -> dict[str, Any]:
    """Compare this module's entanglement with block2's own routine.

    For an ``SZ`` MPS the two must agree: sectors are one-dimensional and the
    multiplicity question does not arise. For an ``SU2`` MPS they must *not*
    agree, because ``get_bipartite_entanglement`` omits the multiplet
    degeneracy; the deviation is reported rather than treated as an error.
    """
    reference = np.asarray(driver.get_bipartite_entanglement(ket), dtype=float)
    mine = np.asarray(profile.entanglement, dtype=float)
    size = min(len(reference), len(mine))
    difference = mine[:size] - reference[:size]
    return {
        "block2_bipartite_entanglement": reference[:size].tolist(),
        "sector_resolved_entanglement": mine[:size].tolist(),
        "max_abs_difference": float(np.max(np.abs(difference))),
        "agrees": bool(np.max(np.abs(difference)) < 1e-8),
        "expected_to_agree": not profile.is_su2,
        "normalisation_max_error": float(
            np.max(np.abs(np.asarray(profile.normalisation) - 1.0))
        ),
    }


def dmrg_sector_profile(asham, schedule, **kwargs) -> tuple[float, SectorProfile]:
    """Converge DMRG on ``asham`` and return its energy with the sector profile.

    The profile is extracted inside the solver's lifetime, so a single run
    yields both the energy and the symmetry-resolved structure at no extra
    solver cost.
    """
    from tn_quantum_chemistry.validation import run_dmrg

    captured: dict[str, SectorProfile] = {}

    def capture(driver, ket) -> None:
        captured["profile"] = symmetry_resolved_profile(
            driver, ket, n_orbitals=asham.n_orbitals
        )

    energy, _, _ = run_dmrg(asham, schedule, post_process=capture, **kwargs)
    return energy, captured["profile"]


def spin_square_expectation(driver, ket) -> float:
    """``<S^2>`` of the converged MPS.

    Under ``SU2`` this is structurally fixed and comes back at machine zero for
    a singlet; measuring it turns the guarantee into recorded evidence rather
    than an assumption. Under ``SZ`` it is a genuine check: only the projection
    ``S_z`` is constrained there, and a finite-bond-dimension state can carry
    weight in the wrong total-spin sector while still looking converged in
    energy.
    """
    return float(driver.expectation(ket, driver.get_spin_square_mpo(iprint=0), ket))


def profile_invariants(profile: SectorProfile) -> dict[str, float]:
    """Residuals of the identities a correct profile must satisfy."""
    total = np.asarray(profile.entanglement, dtype=float)
    number = np.asarray(profile.number_entropy, dtype=float)
    configurational = np.asarray(profile.configurational_entropies["1.0"], dtype=float)
    return {
        "sum_rule_error": float(np.max(np.abs(total - number - configurational))),
        "normalisation_error": float(
            np.max(np.abs(np.asarray(profile.normalisation, dtype=float) - 1.0))
        ),
    }
