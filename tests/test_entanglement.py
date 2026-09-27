"""Symmetry-resolved bipartition data: invariants and cross-solver agreement.

The extraction relies on a multiplicity convention that is easy to get wrong
and whose failure mode is silent -- a wrong factor still produces a smooth,
plausible, symmetric profile. These tests pin it down against quantities
computed by completely independent routes: the 1-RDM, and the same state
converged in an abelian symmetry where no multiplicity factor exists.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from tn_quantum_chemistry.entanglement import (
    FEATURE_NAMES_PER_CUT,
    feature_names,
    feature_vector,
    sector_tail_weight,
    symmetry_resolved_profile,
)
from tn_quantum_chemistry.validation import (
    DMRGSchedule,
    HamiltonianSpec,
    active_space_hamiltonian,
    build_mean_field,
    linear_hydrogen_chain,
    run_dmrg,
)

STO3G = HamiltonianSpec(basis="sto-3g")
# H6 rather than H10: the cross-symmetry test needs a converged SZ run, whose
# cost grows much faster than SU2's.
GEOMETRY = linear_hydrogen_chain(6, R=2.0, delta=0.0, label="H6_R2.0")


@pytest.fixture(scope="module")
def hamiltonian():
    return active_space_hamiltonian(build_mean_field(GEOMETRY, STO3G), STO3G)


@pytest.fixture(scope="module")
def su2_run(hamiltonian):
    """Converged SU2 MPS, with the profile and 1-RDM taken inside its lifetime."""
    captured: dict = {}

    def capture(driver, ket):
        captured["profile"] = symmetry_resolved_profile(
            driver, ket, n_orbitals=hamiltonian.n_orbitals
        )
        captured["dm1"] = driver.get_1pdm(ket)
        captured["dm2"] = driver.get_2pdm(ket)
        captured["block2_entanglement"] = np.asarray(
            driver.get_bipartite_entanglement(ket), dtype=float
        )

    energy, _, _ = run_dmrg(hamiltonian, DMRGSchedule.default(), post_process=capture)
    captured["energy"] = energy
    return captured


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_sector_weights_are_normalised(su2_run):
    assert np.allclose(su2_run["profile"].normalisation, 1.0, atol=1e-12)
    for cut in su2_run["profile"].sector_weights:
        assert sum(cut.values()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.slow
def test_mean_particle_number_matches_the_one_particle_rdm(su2_run):
    """The strongest independent check on the sector weights.

    ``<N_L>`` is the first moment of the sector distribution and is also the
    cumulative diagonal of the 1-RDM. The two are computed by entirely separate
    routes, so agreement constrains the weights themselves, not merely their
    normalisation.
    """
    profile = su2_run["profile"]
    cumulative = np.cumsum(np.diag(su2_run["dm1"]))[:-1]
    assert np.allclose(profile.mean_n_left, cumulative, atol=1e-10)


@pytest.mark.slow
def test_particle_number_variance_matches_the_two_particle_rdm(su2_run):
    """Second-moment check on the U(1) resolution.

    ``var(N_L)`` is the second central moment of ``p[l, N]`` and is separately
    obtainable from the 1- and 2-RDM via
    ``<n_i n_j> = dm2[i,j,j,i] + delta_ij <n_i>``. Agreement constrains how the
    weight is distributed over particle-number sectors, which the first moment
    alone does not: a distribution can carry the right mean with the weight
    spread across entirely wrong labels.
    """
    profile, dm1, dm2 = su2_run["profile"], su2_run["dm1"], su2_run["dm2"]
    occupations = np.diag(dm1)
    nn = np.einsum("ijji->ij", dm2) + np.diag(occupations)
    assert nn.sum() == pytest.approx(float(occupations.sum()) ** 2, abs=1e-8)

    for cut in range(profile.n_cuts):
        left = slice(0, cut + 1)
        mean = float(occupations[left].sum())
        variance = float(nn[left, left].sum()) - mean**2
        assert profile.var_n_left[cut] == pytest.approx(variance, abs=1e-10)


@pytest.mark.slow
def test_particle_number_support_is_physically_bounded(su2_run, hamiltonian):
    """``N_L`` cannot exceed twice the orbitals on the left, nor leave too few
    for the right block to hold the rest."""
    profile = su2_run["profile"]
    n_orbitals = profile.n_orbitals
    n_electrons = hamiltonian.n_electrons

    for cut, marginal in enumerate(profile.number_distribution()):
        assert sum(marginal.values()) == pytest.approx(1.0, abs=1e-12)
        occupied = [n for n, p in marginal.items() if p > 1e-12]
        n_right_orbitals = n_orbitals - (cut + 1)
        assert min(occupied) >= 0
        assert max(occupied) <= 2 * (cut + 1)
        # The complement must be able to hold the remaining electrons.
        assert min(occupied) >= n_electrons - 2 * n_right_orbitals


@pytest.mark.slow
def test_entropies_respect_their_bounds(su2_run):
    profile = su2_run["profile"]
    entropy = np.asarray(profile.entanglement)
    assert np.all(entropy >= 0.0)
    assert np.all(entropy <= np.log(np.asarray(profile.bond_dimensions)) + 1e-9)
    # Marginalising over spin and irrep cannot increase the entropy.
    assert np.all(np.asarray(profile.number_entropy) <= entropy + 1e-9)
    assert np.all(np.asarray(profile.var_n_left) >= -1e-12)


# --------------------------------------------------------------------------- #
# Cross-symmetry agreement -- the multiplicity convention
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_su2_extraction_reproduces_an_abelian_sz_calculation(hamiltonian):
    """Fixes the SU(2) multiplicity convention against a calculation without one.

    In SZ mode each sector is a single state, so ``get_bipartite_entanglement``
    is already the true von Neumann entropy. The SU2 extraction must reproduce
    it for the same physical state.
    """
    import tempfile

    import block2
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    block2.Random.rand_seed(1234)
    driver = DMRGDriver(
        scratch=tempfile.mkdtemp(), symm_type=SymmetryTypes.SZ, n_threads=4,
        stack_mem=4 << 30,
    )
    driver.initialize_system(
        n_sites=hamiltonian.n_orbitals, n_elec=hamiltonian.n_electrons,
        spin=hamiltonian.spin, orb_sym=hamiltonian.orb_sym,
    )
    mpo = driver.get_qc_mpo(
        h1e=(hamiltonian.h1e.copy(), hamiltonian.h1e.copy()),
        g2e=(hamiltonian.g2e.copy(), hamiltonian.g2e.copy(), hamiltonian.g2e.copy()),
        ecore=hamiltonian.e_core_constant, iprint=0,
    )
    ket = driver.get_random_mps(tag="SZREF", bond_dim=400, nroots=1)
    driver.dmrg(
        mpo, ket, n_sweeps=24, bond_dims=[400] * 8 + [1024] * 16,
        noises=[1e-4] * 8 + [1e-6] * 6 + [0] * 10, thrds=[1e-12] * 24,
        tol=1e-12, iprint=0,
    )
    sz_entanglement = np.asarray(driver.get_bipartite_entanglement(ket), dtype=float)

    captured: dict = {}
    run_dmrg(
        hamiltonian, DMRGSchedule.default(),
        post_process=lambda d, k: captured.update(
            profile=symmetry_resolved_profile(d, k, n_orbitals=hamiltonian.n_orbitals)
        ),
    )
    assert np.allclose(captured["profile"].entanglement, sz_entanglement, atol=1e-5)


@pytest.mark.slow
def test_block2_su2_bipartite_entanglement_omits_multiplet_degeneracy(su2_run):
    """Documents why this module does not simply call block2's routine.

    ``get_bipartite_entanglement`` takes the Shannon entropy of the reduced
    SU(2) spectrum, which counts each multiplet once instead of ``2S+1`` times,
    and so reports less entanglement than the state carries. If this test ever
    fails, block2 has changed its convention and this module should be
    rechecked against the SZ reference rather than simply trusted.
    """
    mine = np.asarray(su2_run["profile"].entanglement)
    block2_value = su2_run["block2_entanglement"][: len(mine)]
    assert np.all(mine >= block2_value - 1e-9)
    assert np.max(mine - block2_value) > 1e-3


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_feature_vector_is_finite_and_matches_its_names(su2_run):
    profile = su2_run["profile"]
    features = feature_vector(profile)
    assert features.shape == (len(FEATURE_NAMES_PER_CUT) * profile.n_cuts,)
    assert np.all(np.isfinite(features))
    assert len(feature_names(profile.n_cuts)) == features.size


@pytest.mark.slow
def test_sector_tail_weight_vanishes_when_nothing_is_discarded(hamiltonian):
    """Keeping every Schmidt value must leave no tail in any sector."""
    from tn_quantum_chemistry.entanglement import extract_sector_spectra

    captured: dict = {}
    run_dmrg(
        hamiltonian, DMRGSchedule.default(),
        post_process=lambda d, k: captured.update(spectra=extract_sector_spectra(d, k)),
    )
    spectra = captured["spectra"]
    assert sum(len(t) for t in sector_tail_weight(spectra, keep=10**6)) == 0

    truncated = sector_tail_weight(spectra, keep=2)
    assert any(sum(cut.values()) > 0 for cut in truncated)
    for cut in truncated:
        assert all(0.0 <= w <= 1.0 for w in cut.values())


# --------------------------------------------------------------------------- #
# Stored spectra and spectral features
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_stored_spectrum_is_a_normalised_sorted_probability_vector(su2_run):
    profile = su2_run["profile"]
    assert len(profile.sector_spectra) == profile.n_cuts
    for cut in range(profile.n_cuts):
        values = profile.schmidt_values(cut)
        assert values.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(np.diff(values) <= 1e-15)  # sorted descending
        assert np.all(values >= 0.0)


@pytest.mark.slow
def test_renyi_entropies_are_consistent_with_the_spectrum(su2_run):
    """S_1 is the von Neumann entropy and S_inf is -log of the largest weight.

    Both are computed by separate code paths from the profile's own
    ``entanglement`` field, so agreement ties the spectral machinery to the
    already-validated multiplicity convention.
    """
    from tn_quantum_chemistry.entanglement import RENYI_ALPHAS

    profile = su2_run["profile"]
    assert np.allclose(profile.renyi_entropies["1.0"], profile.entanglement, atol=1e-12)

    largest = np.array([profile.schmidt_values(c)[0] for c in range(profile.n_cuts)])
    assert np.allclose(profile.renyi_entropies["inf"], -np.log(largest), atol=1e-12)

    # Renyi entropy is non-increasing in its order.
    for lower, higher in itertools.pairwise(RENYI_ALPHAS):
        assert np.all(
            np.asarray(profile.renyi_entropies[str(lower)])
            >= np.asarray(profile.renyi_entropies[str(higher)]) - 1e-9
        )


@pytest.mark.slow
def test_raw_spectrum_features_have_a_fixed_layout(su2_run):
    """The raw-spectrum option must at least be shape-stable to be comparable."""
    from tn_quantum_chemistry.entanglement import raw_spectrum_features

    profile = su2_run["profile"]
    for top_k in (4, 16):
        features = raw_spectrum_features(profile, top_k=top_k)
        assert features.shape == (top_k * profile.n_cuts,)
        assert np.all(np.isfinite(features))


@pytest.mark.slow
def test_effective_rank_grows_with_correlation():
    """The spectrum flattens as the chain is stretched.

    This is why a fixed top-k truncation of the raw spectrum is not a stable
    representation across the surface: the number of Schmidt values that matter
    changes by an order of magnitude.
    """
    from tn_quantum_chemistry.entanglement import dmrg_sector_profile

    ranks = {}
    for spacing in (1.0, 2.4):
        hamiltonian = active_space_hamiltonian(
            build_mean_field(linear_hydrogen_chain(6, R=spacing), STO3G), STO3G
        )
        _, profile = dmrg_sector_profile(hamiltonian, DMRGSchedule.default())
        middle = profile.n_cuts // 2
        ranks[spacing] = profile.effective_rank(middle, fraction=0.99)

    assert ranks[2.4] > 2 * ranks[1.0]


# --------------------------------------------------------------------------- #
# Symmetry-resolved decomposition
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_entropy_decomposes_exactly_into_number_and_configurational_parts(su2_run):
    """``S = S_number + sum_N p_N S_N``, exactly, at every cut.

    The single strongest invariant available here. The three terms are computed
    from the pooled spectrum, the charge marginal, and the per-charge spectra
    respectively, so none of them can be wrong in isolation while the identity
    holds -- it simultaneously constrains the multiplicity convention, the
    sector labels and the weights.
    """
    profile = su2_run["profile"]
    total = np.asarray(profile.entanglement)
    number = np.asarray(profile.number_entropy)
    configurational = np.asarray(profile.configurational_entropies["1.0"])
    assert np.allclose(total, number + configurational, atol=1e-12)


@pytest.mark.slow
def test_symmetry_resolved_parts_are_individually_physical(su2_run):
    from tn_quantum_chemistry.entanglement import (
        Sector,
        charge_resolved_spectra,
        symmetry_resolved_entropies,
    )

    profile = su2_run["profile"]
    for cut in range(profile.n_cuts):
        rebuilt = {
            Sector(*key): np.asarray(values, dtype=float)
            for key, values in profile.sector_spectra[cut].items()
        }
        weights, entropies = symmetry_resolved_entropies(
            rebuilt, 1.0, is_su2=profile.is_su2
        )
        grouped = charge_resolved_spectra(rebuilt, is_su2=profile.is_su2)

        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-12)

        # Charge weights must agree with the marginal computed independently.
        marginal = profile.number_distribution()[cut]
        for n, weight in weights.items():
            assert weight == pytest.approx(marginal[n], abs=1e-12)

        for n, entropy in entropies.items():
            # An entropy within a sector is still an entropy, bounded by the
            # log of the number of *physical* states it spans. Note that this
            # count is the multiplet-expanded one: a single SU(2) multiplet of
            # spin S already carries log(2S+1) of entanglement.
            n_states = len(grouped[n])
            assert entropy > -1e-12
            assert entropy <= np.log(n_states) + 1e-9


@pytest.mark.slow
def test_configurational_entropy_cannot_exceed_the_total(su2_run):
    profile = su2_run["profile"]
    for alpha in ("1.0", "2.0"):
        configurational = np.asarray(profile.configurational_entropies[alpha])
        assert np.all(configurational > -1e-12)
        assert np.all(configurational <= np.asarray(profile.renyi_entropies[alpha]) + 1e-9)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_profile_survives_a_json_round_trip(su2_run):
    """The stored spectra must come back as spectra, not as strings.

    ``to_dict`` has to stringify the ``(N, 2S, pg)`` tuple keys because JSON
    cannot key on tuples, so the encoding is a real transformation and not a
    formality. The sum rule is re-checked on the restored object: it constrains
    the weights, the labels and the multiplicity convention together, so if the
    encoding lost or reordered anything it fails here.
    """
    import json

    from tn_quantum_chemistry.entanglement import SectorProfile

    original = su2_run["profile"]
    restored = SectorProfile.from_dict(json.loads(json.dumps(original.to_dict())))

    assert restored.n_cuts == original.n_cuts
    for cut in restored.sector_spectra:
        assert all(isinstance(key, tuple) and len(key) == 3 for key in cut)

    total = np.asarray(restored.entanglement)
    number = np.asarray(restored.number_entropy)
    configurational = np.asarray(restored.configurational_entropies["1.0"])
    assert np.allclose(total, number + configurational, atol=1e-12)
    assert np.allclose(total, original.entanglement, atol=1e-15)

    for cut in range(restored.n_cuts):
        assert restored.schmidt_values(cut).sum() == pytest.approx(1.0, abs=1e-12)
        assert np.allclose(
            restored.schmidt_values(cut), original.schmidt_values(cut), atol=1e-15
        )


@pytest.mark.slow
def test_profile_persists_through_a_record_store(su2_run, tmp_path):
    """A record carrying a profile must survive the append/read cycle."""
    from tn_quantum_chemistry.entanglement import SectorProfile
    from tn_quantum_chemistry.schema import CalculationRecord, RecordStore

    profile = su2_run["profile"]
    record = CalculationRecord(
        system_label="H6_R2.0",
        method="DMRG(chi=500)",
        basis="sto-3g",
        atoms=[["H", 0.0, 0.0, float(i)] for i in range(6)],
        e_total=su2_run["energy"],
        e_nuclear_repulsion=0.0,
        sector_profile=profile.to_dict(),
    )
    store = RecordStore(tmp_path / "records.jsonl")
    assert store.append(record)

    (read_back,) = list(store)
    assert read_back.sector_profile is not None
    restored = SectorProfile.from_dict(read_back.sector_profile)
    assert np.allclose(restored.entanglement, profile.entanglement, atol=1e-15)

    # Bulk data is archival, not a table column.
    assert "sector_profile" not in store.to_table()[0]
    assert "sector_profile" in store.to_table(include_bulk=True)[0]


def test_sector_key_encoding_round_trips():
    from tn_quantum_chemistry.entanglement import decode_sector_key, encode_sector_key

    for key in [(0, 0, 0), (3, 1, 0), (10, 5, 7)]:
        assert decode_sector_key(encode_sector_key(key)) == key
