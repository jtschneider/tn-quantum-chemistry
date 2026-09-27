# Methods and limitations

What was computed, under which conventions, and what the numbers do and do not
support. The claims here are deliberately narrow: every result is for one
finite Hamiltonian defined by a basis, an electron number and a symmetry
sector, and none of them is a chemically converged prediction.

Numbers quoted below are reproduced by the scripts in `scripts/` from the
cached data in `data/`, and the invariants are asserted in `tests/`.

---

## 1. The finite Hamiltonian

Everything is a Born–Oppenheimer electronic-structure calculation in a fixed
atomic-orbital basis. Within that basis, electron number and spin sector, full
configuration interaction (FCI) is exact — it is the complete diagonalisation
of the Hamiltonian in that space.

That word *exact* is doing narrow work. It says nothing about basis-set
incompleteness, and a minimal basis such as STO-3G is far from complete. The
H₁₀ energies here are exact solutions of a deliberately small model, useful for
validating a solver and for exposing where approximate methods fail. They are
not bond energies anyone should quote as chemistry.

Bases used: **STO-3G** for the hydrogen-chain work, **STO-6G** for the
comparison against published benchmark data (which uses that basis), and
**cc-pVDZ** alongside STO-3G in the N₂ vignette, so that basis-set effects are
measured rather than mentioned.

## 2. Conventions

Fixed before any production data was generated, recorded per calculation, and
pinned by executable assertions in `tests/test_conventions.py`.

- **Units.** Energies in Hartree, coordinates in Ångström unless a record says
  otherwise. The same geometry expressed in Ångström or Bohr must give the same
  energy. The project uses PySCF's Bohr constant (`pyscf.data.nist.BOHR`,
  CODATA 2010) throughout rather than mixing it with the newer CODATA 2018
  value; the difference is 3 × 10⁻¹¹ relative and irrelevant to any energy here,
  but two conventions in one repository is exactly the sort of silent
  inconsistency this section exists to prevent.
- **Nuclear repulsion.** Total and electronic energies are stored separately,
  with `E_nuc` alongside. Electronic energy is always `E_total − E_nuc`, never
  `E_total − ecore`: once a core is frozen, the integral constant absorbs the
  frozen electrons' energy, and for N₂ the two differ by ~100 Ha.
- **Spin.** `spin` is 2S throughout, matching both PySCF and block2, and fixes
  the `(n_α, n_β)` pair.
- **Frozen core.** Freezing the two N 1s orbitals of N₂ gives CAS(10e,8o). The
  nuclear repulsion is unchanged; only the core constant shifts, and the
  frozen-core energy lies variationally above the all-electron result.
- **Orbital ordering.** Converged energies are invariant under orbital
  permutation in both solvers. DMRG *cost* and truncation error do depend on
  ordering; the converged energy does not.
- **Point-group symmetry** is disabled by default. This removes irrep-ordering
  conventions as a possible explanation for any solver disagreement, and is
  verified against a published D2h result.
- **Geometry.** Hydrogen chains use uniform spacing `R` with alternating
  displacement `δ`, giving bond lengths `R ± δ/2`. Giner et al. use `R ± 2δ`
  for the same family, so `δ_project = 4 δ_Giner`; the conversion is implemented
  and tested against bond lengths rather than against nominal parameters. That
  paper is not open access, so the comparison to it here is qualitative and its
  convention is *not* asserted as verified against the published text.
- **Failures.** A failed or unconverged calculation is kept as a record with
  its status and error message. A missing row and a failed row mean different
  things.

## 3. Methods

**Single-reference ladder.** RHF, MP2, CCSD and CCSD(T), each tier's wall time
including the reference it is built on.

**DMRG** (block2, SU(2)-adapted). Each bond dimension χ is run as its own
record with a *flat* schedule — `[χ] × 20` sweeps with a noise ramp
(1e-4 → 1e-6 → 0) — rather than warmed up from a smaller χ. This is deliberate:
χ = 64 is published as a low-fidelity tier, and warming it up from χ = 500 would
make it a statement about the ramp rather than about χ. The local eigensolver
threshold and the sweep stopping rule are both 1e-14 (see §4).

**Multireference** (N₂ only). CASCI and CASSCF in the full-valence CAS(10e,8o)
obtained by leaving the two N 1s orbitals inactive, followed by NEVPT2 on the
converged CASSCF for the dynamical correlation outside the active space.

## 4. Solver validation and convergence

No DMRG result in this repository is published unless the following gate
passes. Both solvers consume the *same* `h1e`/`g2e`/`ecore` arrays, and a
SHA-256 digest of those arrays is stored in each record and re-checked after the
run, so "the same Hamiltonian" is recorded evidence rather than an assumption.
(That check earns its keep: block2 symmetrises and unpacks the two-electron
array in place, so it must be handed copies.)

| System | \|E_DMRG − E_FCI\| | ⟨S²⟩ | entropy sum rule | Schmidt norm |
| --- | --- | --- | --- | --- |
| N₂ at 1.1 Å, STO-3G | 1.7 × 10⁻¹³ | 4.5 × 10⁻¹⁷ | 2.2 × 10⁻¹³ | 1.8 × 10⁻¹⁴ |
| H₁₀ at R = 1.0 Å | 1.1 × 10⁻¹³ | 1.8 × 10⁻¹⁶ | 2.2 × 10⁻¹⁶ | 8.9 × 10⁻¹⁶ |
| H₁₀ at R = 2.4 Å | 2.5 × 10⁻¹³ | 1.7 × 10⁻¹⁵ | 8.9 × 10⁻¹⁶ | 6.7 × 10⁻¹⁶ |

against a stated tolerance of 10⁻⁸ Ha. The gate certifies the *state* and not
only the number, because energy agreement does not imply sector purity: an
abelian `SZ` calculation fixes only S_z and can converge into a different
total-spin sector while its energy still looks reasonable. Every DMRG state used
here therefore carries its ⟨S²⟩, and the one place an `SZ` run is needed (§5) is
checked against the spin-adapted result before its numbers are used.

**Reproducing published data.** Against the Simons Collaboration hydrogen-chain
benchmark, this project reproduces the published FCI energies at all ten
tabulated spacings to within **4.9 × 10⁻⁹ Ha** — the resolution limit of their
eight printed decimals. The geometry unit and basis are not stated at the
source and had to be established by reproduction (Bohr and STO-6G; reading the
spacings as Ångström reproduces nothing), and both are pinned by tests.

### Why the convergence thresholds are what they are

The DMRG–FCI discrepancy originally grew smoothly with chain spacing, from
10⁻¹² Ha near equilibrium to 1.8 × 10⁻⁹ Ha at R = 3.4 Å. The sign identified the
cause: converged DMRG sat *below* FCI everywhere, and since DMRG is variational
for the same Hamiltonian, the reference was the quantity above the true ground
state. As H₁₀ dissociates, the M_S = 0 sector fills with near-degenerate
triplets — at R = 3.4 Å the nearest state to the singlet ground state is a
triplet only 4.6 × 10⁻⁵ Ha above it, while the nearest *singlet* is 2.6× further
— and PySCF's Davidson, which diagonalises that whole sector with a default
12-vector subspace, converges progressively worse. Block2 is SU(2)-adapted and
never sees those states.

Raising the FCI Davidson subspace with `conv_tol` 1e-14, and tightening the DMRG
local-solver threshold and sweep tolerance to 1e-14, reduced the worst-case
discrepancy from 1.8 × 10⁻⁹ to the 10⁻¹² range, and both energies moved *down* —
confirming that each had been above the true value rather than disagreeing with
the other. The wider subspace is also faster, because it converges instead of
restarting. Bond dimension was never the constraint: at R = 3.4 Å, χ = 500, 1000
and 1500 agree to 10⁻¹⁴ with a discarded weight of 10⁻¹⁸, so the MPS is exact
there.

The subspace was raised to 30 first and to **200** on 2026-09-07, after χ = 512
runs — which carry the full Schmidt rank at H₁₀ and are therefore an independent
exact diagonalisation — were found to sit *below* FCI at all six spacings
tested. Since both methods are variational from above, the lower energy is the
better one, and that sign was proof the reference was still the weaker of the
two. At 200 the subspace is converged: 200 and 300 agree to every digit at
R = 3.4 Å, where 30 had left 3.9 × 10⁻¹² Ha on the table. See `run_fci` for the
full progression, including its non-monotonicity.

**The roles have now reversed.** Before the change, converged DMRG was below FCI
at 100% of comparable geometries — a systematic bias, and the signature of an
under-converged reference. After it, that falls to 81% across 169 records, the
worst case at R = 3.4 Å reverses sign to **+2.3 × 10⁻¹²** (DMRG above FCI), and
what remains is DMRG's own seed dependence at stretched geometries rather than
the reference's convergence.

The residual **2.3 × 10⁻¹² Ha** worst case is reported as the floor of the
comparison, with a median of 1.8 × 10⁻¹³ across those records. The error panel of
the dissociation figure shades it. No difference smaller than that is resolvable
between the two solvers.

Two cautions that follow from this:

- **Discarded weight is not an error bar here, and does not become one at large
  χ.** Part of the non-monotonicity seen before was the loose local-solver
  threshold above; what survives the fix is sharper. At R = 1.8 Å the DMRG
  energy is identical to thirteen decimal places for χ = 144, 160 and 176
  (−4.8141667172683 Ha) while the final discarded weight falls from
  1.5 × 10⁻² to 8.8 × 10⁻³ and the bipartite entropy rises from 3.27 to 3.37:
  the added bond dimension is spent on directions that carry entropy but no
  energy. At χ = 192 the same calculation reaches the reference
  (1.4 × 10⁻¹⁴ Ha) with a discarded weight still at 6.9 × 10⁻³ — larger than
  that of a *much worse* run at a different spacing. The familiar proportionality
  survives only within one geometry: at R = 1.0 Å the ratio of energy error to
  discarded weight is 0.11, 0.057 and 0.055 across χ = 128, 192 and 256. It is
  not comparable between geometries, so the quantity is recorded but never used
  to choose χ or to bound an error. See §6.
- **A diverged coupled-cluster energy is not a reproducible quantity.** Once the
  amplitude equations have no physical solution, which unphysical one a code
  reaches depends on the solver path; repeated runs here differ by 10⁻² Ha.
  Those points are excluded from numerical comparison and the *failure* is
  asserted instead.

## 5. Orbital entanglement

Single-orbital entropy `s_i` and two-orbital mutual information
`I_ij = (s_i + s_j − s_ij)/2`, computed from reduced density matrices of the
converged MPS at three points on the validated curve. These are the standard
QC-DMRG diagnostics; nothing here is novel.

| R (Å) | regime | Σ s_i | max I_ij | mean s_i, canonical | mean s_i, Löwdin |
| --- | --- | --- | --- | --- | --- |
| 1.0 | weakly correlated | 2.53 | 0.248 | **0.253** | 1.352 |
| 1.8 | moderately correlated | 10.42 | 0.405 | 1.042 | 1.030 |
| 2.6 | strongly correlated | 13.36 | 0.357 | 1.336 | **0.741** |

**These are not invariants of the molecule.** They describe entanglement between
the *chosen* orbitals. Localising the orbitals is a unitary change of basis, so
the FCI energy is unchanged to 10⁻¹⁰ — but `s_i` changes completely, and the two
bases swap roles across the curve. Values near the ln 4 ceiling appear in
whichever basis is wrong for the regime: delocalised orbitals for a stretched
Mott-like chain, localised orbitals for a compressed one. A saturated `s_i`
diagnoses an unsuitable orbital partition, not a state beyond the method's
reach. Σ s_i is likewise an upper bound rather than a measure — it
multiply-counts correlation shared between orbitals.

Two routes are used because one is broken: block2's `get_orbital_entropies`
raises in SU(2) mode and works in SZ, so `s_i` comes analytically from the
spin-traced RDMs of the validated SU(2) run, and `s_ij` from a separate SZ
calculation. The SZ state is verified against the SU(2) one before its numbers
are used — energies to 8.7 × 10⁻¹² Ha, ⟨S²⟩ below 7.7 × 10⁻⁹, and `s_i` to
8.0 × 10⁻⁷. All RDM invariants hold across all geometries (worst violation 0.0):
traces, Hermiticity, natural occupations inside [0, 2], normalised one-orbital
probabilities, subadditivity and the Araki–Lieb inequality.

## 6. Bond dimension and cost

A sweep over χ ∈ {16 … 512} at six spacings with three MPS seeds each, scored
against the FCI reference for the same integrals. The question was whether the
variational error falls smoothly with χ at a geometry-dependent rate, or whether
low-χ runs land in different local minima. Both happen, in different regimes,
and neither is predicted by the entanglement entropy.

`χ*` is the smallest sampled bond dimension reaching 10⁻⁹ Ha — three orders of
magnitude above the reference floor, so it measures the MPS and not the last
digits of the FCI comparison. The seed spread is taken only over bond dimensions
whose error is still above that threshold: once a run reaches the floor, ratios
between seeds are large but meaningless.

| R (Å) | max bipartite S | χ* to 10⁻⁹ Ha | worst seed spread |
| --- | --- | --- | --- |
| 1.0 | 0.84 | 384 | 1.2× |
| 1.4 | 2.08 | 192 | 1.1× |
| 1.8 | 3.60 | 192 | 1.2× |
| 2.2 | 4.44 | 256 | 1.3× |
| 2.6 | 4.66 | 256 | 10.3× |
| 3.0 | 4.67 | 384 | 58.7× |

Every spacing reaches the reference floor by χ = 384; the worst case, R = 3.0 Å,
lands at 1.1 × 10⁻¹² against a floor of 2.3 × 10⁻¹². The whole grid was run twice
— once with no effective singular-value cutoff and once at 1e-14, 450 runs in
total — and the two series return the same χ* at every spacing and the same
entropies to three decimals, so the table is a replication rather than a single
measurement. At χ = 512 the MPS carries
992 states at the central cut, which is exactly the full Schmidt rank there
(452 SU(2) multiplets, counted with their 2S+1 degeneracies), so DMRG is exact
by construction at that point rather than merely converged.

Four observations, each a caution rather than a method:

- **Entropy predicts representational cost, but not energetic cost.** These are
  different quantities and conflating them is easy. Taken from the exact Schmidt
  spectrum of the converged MPS, the number of states needed to reach a fixed
  *truncation* error tracks the von Neumann entropy as the standard argument
  says it should: Pearson correlation of ln n with S is **+0.85** at a discarded
  weight of 10⁻⁶ and **+0.86** at 10⁻⁹. Against χ* — the bond dimension needed
  for a fixed *energy* accuracy — the same correlation is **−0.08**, and χ* is
  U-shaped (384, 192, 192, 256, 256, 384) while the entropy rises monotonically.
  Rényi orders α = ½, 2 and ∞ behave no differently, so this is not a matter of
  having used the wrong entropy.

  The gap between the two is the conversion factor, and it is not constant. At a
  common bond dimension of 128, the energy error per unit of discarded weight is

  | R (Å) | 1.0 | 1.4 | 1.8 | 2.2 | 2.6 | 3.0 |
  | --- | --- | --- | --- | --- | --- | --- |
  | ΔE / w | 0.109 | 0.125 | 0.033 | 0.0032 | 0.0020 | 0.00082 |

  a monotone fall of 133× across the surface. Discarding a given amount of
  Schmidt weight costs two orders of magnitude more energy at the compressed end
  than at the stretched end. That is consistent with the composition of the tail
  measured above — at stretched geometries it is dominated by near-degenerate
  spin configurations, which carry entropy but almost no energy — and it is the
  same fact as the discarded-weight caution below, seen from the other side.

  So the practical statement is narrow and specific: entanglement entropy is a
  good guide to how hard the *state* is to represent, and not a guide to what
  bond dimension a target *energy* accuracy needs, unless the energy carried per
  unit of discarded weight is known to be comparable between the systems being
  compared. On this surface it is not.
- **The error moves in plateaus, not a smooth decay.** Adjacent bond dimensions
  give energies agreeing to ~10⁻¹³ over wide ranges, then jump by 10⁻⁴:

  | R (Å) | plateau in χ | jump on leaving it |
  | --- | --- | --- |
  | 1.0 | none | — |
  | 1.8 | 144 → 176 | 9.4 × 10⁻⁵ |
  | 2.2 | 128 → 192 | 1.5 × 10⁻⁴ |
  | 2.6 | 128 → 192 | 1.0 × 10⁻⁴ |
  | 3.0 | 128 → 256 | escapes only at χ = 384 |

  Sampling χ finely inside a plateau buys nothing, and a convergence check that
  compares two neighbouring bond dimensions inside one would report agreement
  while sitting 10⁻⁴ Ha from the answer.
- **Low-χ runs are reproducible except at the stretched end.** Seeds agree to
  within 1.3× out to R = 2.2 Å; the spread appears only at R = 2.6 (10.3×) and
  R = 3.0 (58.7×). A single low-χ number there is a draw from a distribution, not
  a property of the geometry, which is why no χ = 32 or χ = 64 tier is used as a
  fidelity level anywhere in this repository.
- **A singular-value cutoff does not reduce cost here, but it does separate
  static from dynamic correlation.** `DMRGSchedule` carries one, so the kept
  basis is set by whichever of the cutoff and the bond dimension binds first. At
  1e-14 it does not bind at χ ≥ 384: at every spacing the converged MPS reaches
  an identical bond dimension with and without it — 830, 816, 810, 800, 786 and
  782 states at χ = 384, and 992 at χ = 512 throughout. Reaching the floor is the
  bond dimension's doing, not the cutoff's, and no cutoff produced a systematic
  wall-time saving.

  Loosening it far enough to bite is informative rather than useful. At a 1e-8
  cutoff and a χ = 512 cap:

  | R (Å) | states kept of 992 | error (Ha) |
  | --- | --- | --- |
  | 1.0 | 708 | 5.0 × 10⁻⁷ |
  | 1.4 | 964 | 3.6 × 10⁻¹⁵ |
  | 1.8 | 991 | 2.1 × 10⁻¹³ |
  | 2.2 | 991 | 5.7 × 10⁻¹⁴ |
  | 2.6 | 983 | 4.1 × 10⁻¹⁴ |
  | 3.0 | 980 | 1.9 × 10⁻¹³ |

  At R = 1.0 Å nearly a third of the spectrum lies below 1e-8 and discarding it
  costs seven orders of magnitude in accuracy; everywhere else almost nothing
  lies below 1e-8 and discarding it is free. That is the static/dynamic
  distinction measured directly: the compressed geometry's correlation energy is
  assembled from many individually negligible Schmidt weights, while the
  stretched geometries carry theirs in a few large ones. It is the most direct
  evidence here for why entropy runs backwards against cost — the geometry with
  the *least* entanglement has the tail that cannot be truncated.

The condensed-matter expectation that the variational energy gradient falls off
smoothly with bond dimension describes the compressed end of this curve and not
the stretched end. That is a finite-system effect: the plateaus sit where the
state is a superposition of a small number of near-degenerate spin
configurations, and the energy only moves when χ crosses the threshold at which
that set becomes representable.

Figures are `figures/h10_chi_convergence.svg` (error against χ, one panel per
spacing, seed range shaded) and `figures/h10_chi_summary.svg`; the numbers behind
them are in `figures/h10_chi_convergence.csv`.

### Does the tensor network pay off on a longer chain?

Not on this system class, and the reason is worth stating because the opposite is
usually assumed. The standard argument is that DMRG's advantage appears with
length: the determinant count explodes combinatorially while the bond dimension
needed for fixed accuracy does not. Measured here (`scripts/chain_length_scaling.py`,
`data/raw/chain_scaling/`), it does not appear.

**Exact diagonalisation runs out first, as expected.** With 61 GB, FCI holds
one Davidson vector of 1.3 GB at H₁₆ and 18.9 GB at H₁₈; the wider subspace this
project's reference needs (§4) puts H₁₆ out of reach too. H₁₈ upward, DMRG is the
only exact method available.

**But DMRG does not reach useful accuracy there.** At H₁₂ and R = 3.0 Å, against
an FCI reference that takes 44 s:

| bond cap | χ_eff (states) | discarded weight | \|E − E_FCI\| | wall |
| --- | --- | --- | --- | --- |
| 128 | 217 | 4.8 × 10⁻³ | 1.2 × 10⁻¹ | 1.7 s |
| 256 | 654 | 3.3 × 10⁻⁷ | 4.2 × 10⁻³ | 18 s |
| 512 | 1412 | 2.2 × 10⁻⁴ | 2.5 × 10⁻³ | 366 s |

The error plateaus in the low millihartree range and is immune to effort:
holding the cap at 256 and running 14, 30 or 60 sweeps gives 3.7 × 10⁻³ every
time, to two significant figures. For context, the energy surrogate of §8 has a
test MAE of 4 × 10⁻⁴ Ha — **the DMRG labels would be an order of magnitude worse
than the model trained on them.**

The middle row is also the clearest demonstration in this repository of the
discarded-weight problem: at cap 256 the weight is 3.3 × 10⁻⁷, *below the
requested cutoff*, so every automatic convergence indicator calls the run
converged while the energy is 4.2 × 10⁻³ Ha wrong.

**The epistemic problem is worse than the cost.** Beyond H₁₆ there is no
reference to check a DMRG label against, and this repository has already measured
that DMRG's own diagnostics are unreliable on exactly this system — the discarded
weight is not an error bar across geometries, and the error sits on plateaus
where neighbouring bond dimensions agree to 10⁻¹³ while sitting 10⁻⁴ from the
answer. A dataset built there would consist of unverifiable labels produced by a
method whose self-reported convergence has been shown here to be untrustworthy.

**Why this system is the wrong shape for DMRG.** Hₙ in STO-3G has exactly n
orbitals and n electrons: a minimal basis at half filling, with maximal
entanglement per site and no weakly-correlated orbitals to compress. That is
close to the least favourable geometry for DMRG relative to exact
diagonalisation, which is the opposite of the regime where production QC-DMRG
wins — many orbitals carrying structured, mostly local entanglement. Lengthening
the chain moves along that unfavourable shape rather than out of it.

One correction is recorded with this. An earlier version of this argument held
that stretched chains are gapped one-dimensional systems obeying an area law, so
the bond dimension should saturate. That is false for the *uniform* chain: a
stretched Hₙ chain approaches a Heisenberg antiferromagnet, which is critical,
with entanglement growing as log n. The gap in this system class comes from
dimerisation, not stretching. Measured at R = 3.0 Å, δ = 0.4 lowers the maximum
bipartite entropy from 4.671 to 3.970 at H₁₀, consistent with a spin-Peierls gap
— but the longer dimerised chains could not be converged well enough to test
whether the entropy saturates, so that question is recorded as open rather than
answered.

## 7. The (R, δ) dataset

An 11 × 7 grid over spacing and alternating displacement: **77 geometries × 7
methods = 539 records**. FCI converged at every geometry; the 36 geometries
carrying an unconverged coupled-cluster result keep those rows with their
status.

`δ` spans both signs deliberately. H₁₀ has ten atoms and therefore *nine* bonds,
an odd number, so `+δ` and `−δ` are different systems: `δ < 0` gives five H₂
molecules, `δ > 0` gives four H₂ plus two unpaired terminal atoms, and the two
differ by 210 mHa at R = 1.0 Å. The symmetry that would excuse sampling one sign
holds for rings and for chains with an even bond count, not here.

## 8. Energy surrogates

Four models fitted to the same expensive labels, on the same splits, with the
same preprocessing discipline: a constant correction `E_HF + mean(E_FCI − E_HF)`,
a tensor-product cubic B-spline, a small JAX MLP on `E_FCI`, and the same
architecture on `E_FCI − E_HF` with HF restored at inference. The two neural
models differ *only* in the target.

Protocol: input and target normalisation fitted on the training fold alone;
identical training indices handed to all four models at every label count;
8 to 62 labels; three model seeds; five random splits and three blocked-spacing
holdouts reported separately. Blocked holdouts are drawn on `R` and not on `δ`,
because the coupled-cluster failures cluster at `δ > 0`. The MLP training budget
(80,000 steps) was set by measurement — at 4,000 steps both neural models are
under-trained and the comparison between them would have measured the optimiser.

Median test MAE at 62 labels (kcal/mol):

| split | constant | spline | MLP direct | MLP Δ |
| --- | --- | --- | --- | --- |
| random interpolation | 226.8 | 4.37 | 1.81 | **0.31** |
| held out R = 1.0–1.2 Å | 405.9 | 74.07 | 12.76 | **1.96** |
| held out R = 1.8–2.0 Å | 81.8 | 3.85 | 0.25 | 0.23 |
| held out R = 2.8–3.0 Å | 429.4 | 11.65 | **0.32** | 4.29 |

**Δ-learning helps, conditionally, and the condition is physical.** It wins by
~6× on random interpolation and into the compressed wall, ties in the interior
gap, and *loses* by 13× on the stretched block. The mechanism is measurable: the
correlation energy carries about half the curvature of the total energy per unit
of its own range in the compressed and middle regions (ratios 0.50 and 0.46) and
**2.8× more** in the stretched region, where `E_FCI` flattens toward dissociation
while `E_corr` is still changing rapidly. Δ-learning helps where the baseline's
error is the smoother of the two targets, and not otherwise. It is not a smaller
target: the Δ surface spans 833 kcal/mol against the total energy's 583.

Derivatives with respect to `(R, δ)` are obtained as `-jax.grad(E)` and agree
with central differences of the same fitted model to 3.1 × 10⁻⁶ relative. These
are two-component generalized-coordinate derivatives of a *fitted surface* — not
Cartesian atomic forces, and not evidence about the accuracy of the underlying
physics. No model here was trained on force labels.

Cost context: the FCI labels for the whole grid cost 214 s (2.8 s per point),
the HF baseline 5.4 s (0.07 s per point), and warm inference is sub-millisecond.

### Do the entanglement diagnostics predict where the surrogate fails?

They do not — or at least, not better than signals that are free. The surrogate
consumes geometry alone, so the natural way to make the tensor-network half of
this repository load-bearing would be for an entanglement descriptor of the
state to flag the geometries where the fitted surface is untrustworthy. That is
testable with data already stored: every surface geometry carries a
`sector_profile` at two bond dimensions.

Each surrogate was fitted at its largest label budget on all eight splits with
three model seeds, and the per-geometry held-out error was ranked against each
candidate signal. Correlations are taken *within* a split before pooling,
because errors differ by two orders of magnitude between the random and blocked
splits and a naive pooled correlation would mostly report which split a point
came from. Pooled Spearman ρ against |error|, 117 held-out points:

| signal | direct | Δ | direct, R controlled | Δ, R controlled |
| --- | --- | --- | --- | --- |
| max bipartite entropy | −0.596 | 0.098 | −0.287 | 0.146 |
| configurational entropy | −0.598 | 0.096 | −0.364 | 0.112 |
| number entropy | −0.511 | 0.117 | −0.210 | 0.147 |
| **ensemble disagreement** *(free)* | **0.760** | **0.622** | **0.526** | **0.599** |
| distance to training set *(free)* | 0.340 | 0.460 | — | — |
| spacing R *(free)* | −0.510 | 0.120 | — | — |

Four things follow, and none of them favour the diagnostic:

- **Seed disagreement wins outright.** Three model seeds were already fitted, so
  their spread costs nothing, and it is the strongest signal for both models
  before and after controlling for R.
- **Half the entanglement signal is just R.** The best descriptor falls from
  −0.598 to −0.364 once the spacing the model already consumes is partialled
  out.
- **It says nothing about the model that actually fails.** Δ-learning is the one
  that loses 13× on the stretched block; there the entanglement correlation is
  ≈ 0.10, indistinguishable from noise.
- **The residual signal changes sign between models** — negative for direct
  learning, positive for Δ-learning. That is physically coherent (the RHF
  baseline degrades exactly where entanglement is high, while the total-energy
  surface flattens there) but it means entanglement diagnoses *the baseline's
  validity*, not the surrogate's trustworthiness. A trust signal that reverses
  depending on which model it is asked about is not a trust signal.

The result is unchanged using the cheap χ = 64 tier instead of χ = 500.

This is reported as a negative result rather than dropped. It also settles a
cost question that this system cannot answer favourably: a converged DMRG here
costs 5.4 s against FCI's 2.4 s, so the diagnostic is more expensive than the
label it would be triaging. Any acquisition claim would need a system where DMRG
is the cheaper method, and would still need a signal that beats seed
disagreement.

## 9. The N₂ vignette

Three bond lengths in two bases, to exercise the chemistry choices that
H₁₀/STO-3G hides. Error against FCI in STO-3G, in kcal/mol:

| method | 1.098 Å | 1.6 Å | 2.4 Å |
| --- | --- | --- | --- |
| RHF | 98.5 | 224.2 | 492.4 |
| UHF | 98.5 | 60.8 | 2.7 |
| MP2 | 2.0 | −63.6 | −445.4 |
| CCSD(T) | 1.6 | 6.2 | 252.5 ‡ |
| CASSCF(10e,8o) | 0.08 | 0.02 | 0.001 |
| NEVPT2/CASSCF | 0.001 | 0.000 | 0.000 |

‡ amplitude equations did not converge. Negative means below FCI, which for a
fixed finite Hamiltonian is a failure and not an improvement.

Four things this is meant to show:

1. **RHF stops being a minimum.** Its internal stability analysis passes at
   equilibrium and fails at both stretched geometries. The stabilised UHF
   solution is 2.7 kcal/mol from FCI at 2.4 Å — but carries ⟨S²⟩ = 2.96 for a
   nominal singlet. The energy looks good because the state is wrong.
2. **MP2 diverges** where the reference does, landing 445 kcal/mol *below* FCI,
   while reporting no convergence failure of its own — it has no iterations to
   fail.
3. **In a minimal basis, CAS(10e,8o) *is* frozen-core FCI.** N₂/STO-3G has ten
   orbitals, so leaving the two 1s inactive leaves nothing outside the active
   space. This is why the NEVPT2 correction is ~10⁻⁴ Ha there and why
   CASSCF+NEVPT2 reproduces all-electron FCI to 10⁻⁶ Ha: it is explained, not
   mysterious.
4. **cc-pVDZ is where external correlation becomes visible.** The same NEVPT2
   correction is ~150 mHa (~94 kcal/mol), and the method ordering reverses along
   the stretch: CCSD(T) is the better number at equilibrium, and at 2.4 Å it has
   diverged while the multireference treatment has not.

The vignette's FCI uses `conv_tol` 1e-12 rather than the 1e-14 used for H₁₀.
PySCF's tolerance is absolute, and N₂ sits near −109 Ha against H₁₀'s −5 Ha, so
1e-14 there is 10⁻¹⁶ relative — machine epsilon, which no solver can report as
converged. An absolute threshold does not transfer between systems of different
total energy.

The vignette runs single-threaded. PySCF's NEVPT2 is bit-reproducible on one
thread and varies by ~2 × 10⁻⁵ Ha between identical repeated calls on eight —
non-deterministic summation order in a parallel reduction. On a system this
small, one thread is also *faster* (2.7 s against 9.9 s on 32).

## 10. The N₂ active-space ladder

The same molecule as §9, continued into active spaces exact diagonalisation
cannot reach. Applying DMRG in a large active space is standard practice; what
is reported here is the measurement, its convergence protocol, and what the
protocol is worth in the absence of a reference.

**Conventions.** Orbitals are RHF canonical in C1 symmetry; the ladder's valence
rungs freeze the two N 1s cores, so every one of them correlates the same ten
electrons and the energies are comparable along the ladder. The all-electron
rung freezes nothing. Both solvers at a rung consume one integral array built
once, and its SHA-256 digest is stored in every record, so two numbers can be
shown to come from the same Hamiltonian rather than assumed to.

**The SCF is pinned to one thread**, and this is not a performance choice. On 32
threads the RHF here returns a different set of orbitals on almost every run.
The SCF *energies* agree to 3 × 10⁻¹³, so it looks converged, but the orbitals
differ enough that four builds of `CAS(10e,8o)` gave four distinct integral
digests and a CASCI energy spanning 6.4 × 10⁻¹⁰ Ha — CASCI is not stationary
with respect to orbital rotations, so the noise lands directly in the reference.
It first appeared as DMRG beating the variational bound, which is impossible for
one Hamiltonian and in fact indicated two.

**The Davidson subspace is sized against memory, not fixed.** `run_fci` uses
`max_space=200`, chosen when a CI vector was 0.5 MB. At `CAS(10e,16o)` a vector
is 153 MB and 200 of them would be 61 GB. The subspace actually used is recorded
per calculation. It is not a free parameter in the accuracy sense: recomputing
the 16-orbital reference with the subspace nearly halved, from 28 to 16, moves
it by **less than 2 × 10⁻¹³ Ha** at all three geometries, so the reference is
solid and DMRG's best agreement there (3.4 × 10⁻¹¹) is a real measurement rather
than a comparison of two noises.

### Convergence without a reference

Above 16 orbitals there is no exact answer, so the protocol has to carry the
claim. Four quantities are reported, none of which bounds the true error:

1. the final step in the χ sequence (250 → 500 → 1000);
2. the disagreement between the two orbital orderings at the largest χ;
3. the residual sweep-to-sweep drift when the schedule ends;
4. the seed spread, where more than one seed was run.

The uncertainty quoted is the largest of these. The ordering disagreement
dominates at every Phase 2 point, and it is weighted most heavily on evidence:
it was the only one of the four that detected a genuine failure.

**The failure it detected.** In `CAS(10e,16o)` with Fiedler ordering, an earlier
sweep schedule converged to a state 0.19 Ha above the ground state at every χ
and both seeds at 1.6 Å, with a final discarded weight of 5 × 10⁻¹⁰. At 2.4 Å
raising χ from 500 to 1000 at fixed seed moved the answer *away* from the
correct one. Exact diagonalisation of the reordered integrals matched the
canonical value to 2.8 × 10⁻¹⁴, so the Hamiltonian was correct; a flat schedule
at the same seed and χ converged correctly, so the ordering was not the cause.
The cause was a sweep schedule that finished its perturbative-noise ramp while χ
was still a sixteenth of its target and then reached full bond dimension with
the noise at zero, leaving the sweep no mechanism to escape a local minimum once
it had the freedom to use one.

Discarded weight, seed agreement and entanglement all reported success on that
state. This is a second, sharper instance of the point in §4 and §6 that the
discarded weight is not a calibrated error bar — here it was wrong by seven
orders of magnitude. `tests/test_conventions.py` now asserts, for every schedule
constructor, that noise is still active at at least half the peak bond
dimension, that the schedule ends noise-free so the reported energy is
variational, and that it specifies all its own sweeps instead of relying on
backend padding.

### Testing the protocol against the literature

The protocol above was, until 2026-09-10, untested: it produced an uncertainty at
rungs where nothing could check it. Three published near-exact values now test it.
All were verified against their DOI or arXiv record before use and are vendored in
`data/reference/n2_ccpvdz_external.json`; `tests/test_n2_external_reference.py`
checks the values, the conventions and the interpolation.

| source | space | geometry | value (Ha) |
| --- | --- | --- | --- |
| CDFCI — Wang, Li & Lu, *JCTC* (2019) | all-electron (14e,28o) | 2.118 a₀ | −109.2821727 |
| DMRG — Chan, Kállay & Gauss, *JCP* **121**, 6110 (2004) | all-electron (14e,28o) | 2.118 a₀ | −109.282157 |
| SHCI — Sharma *et al.*, *JCTC* (2017) | frozen-core (10e,26o) | 1.0977 Å | −109.2769(1) |

CDFCI and Chan's DMRG are independent and agree to 1.6 × 10⁻⁵ Ha, so the
all-electron reference is known 37–73× better than the uncertainties being tested,
and 39× better at the geometry actually run here — sharp enough to be a test
rather than a coincidence.

**The Hamiltonian was matched before the energies were compared.** RHF at 2.118 a₀
through this project's own path reproduces the published −108.9493779 to
2.1 × 10⁻⁸ Ha, that value's own rounding. This fixes basis, geometry, unit and the
spherical/Cartesian d convention simultaneously; cc-pVDZ with Cartesian d gives 30
orbitals instead of 28, converges normally, and is indistinguishable in a log
file. Both facts are asserted by tests.

Two obstacles had to be handled rather than assumed away:

- **The sources quote different units** — 2.118 Bohr and 1.0977 Ångström. These
  are two geometries 0.023 Å apart, and treating them as one value in two
  conventions would have produced a 23 mÅ geometry error.
- **Neither geometry is on this project's grid.** The all-electron comparison was
  therefore made by running the ladder at the published bond length (eight DMRG
  runs, 16.4 h wall at 4 threads, a separate store). CDFCI additionally publishes its whole
  binding curve on a 0.05 a₀ grid near equilibrium, which interpolates onto two
  geometries already computed here. 2.4 Å = 4.535 a₀ lies past the curve's last
  point at 4.50 a₀ and is refused rather than extrapolated to.

Interpolation error is measured, not assumed: leave-one-out cubic-spline
prediction of each interior published point gives ~1.0 × 10⁻⁵ Ha near the two
geometries used and 6.2 × 10⁻⁵ Ha at worst over the whole curve. The estimate is
conservative by construction, since removing a point doubles the local grid
spacing, and it is added to the reference's own uncertainty.

| R | how | E<sub>DMRG</sub> (Ha) | published (Ha) | error | σ | σ / \|err\| | gap / \|err\| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2.118 a₀ | run at their geometry | −109.281909 | −109.282173 | +2.64 × 10⁻⁴ | 7.9 × 10⁻⁴ | 2.98× | 1.03× |
| 1.098 Å | interpolated curve | −109.280658 | −109.280913 | +2.54 × 10⁻⁴ | 7.3 × 10⁻⁴ | 2.87× | 1.05× |
| 1.600 Å | interpolated curve | −109.083789 | −109.084270 | +4.81 × 10⁻⁴ | 1.5 × 10⁻³ | 3.02× | 0.68× |
| 2.400 Å | outside the curve | −108.965255 | — | — | 1.4 × 10⁻³ | — | — |

Two conclusions, and the second qualifies the design decision recorded above.

**The uncertainty is honest and over-covers by a factor of three.** All three
comparisons fall inside the band; every DMRG energy sits above the published value
as a variational bound must; and σ / \|error\| is 2.87×, 2.98× and 3.02× — a 5%
spread over three geometries and two independent routes to the reference. The χ
step, not the ordering gap, supplies the margin in all three.

**The ordering gap is the right magnitude for the error but is not a bound.**
Alone it gives 1.05×, 1.03× and 0.68× the true error. At two geometries it
predicts the error to within 5%; at 1.600 Å it underestimates it by half again.
The diagnostic weighted most heavily because it caught a real failure turns out
also to be the best-calibrated size estimate — and therefore the most tempting
thing to quote on its own, which these numbers say would be wrong. Taking the
maximum over the four terms is what makes the quoted band a bound.

The frozen-core rung agrees with SHCI to 1.9 × 10⁻⁵ Ha, but across a 0.0003 Å
geometry offset worth 3.5 × 10⁻⁵ Ha measured by CCSD(T). The agreement is smaller
than the offset separating the two calculations, so the supportable claim is that
the rung agrees to better than that offset, not to 1.9 × 10⁻⁵. The SHCI value also
carries a stochastic perturbative correction and is not variational, so nothing
here may use it as a bound.

### What the numbers support

At and below 16 orbitals, DMRG reproduces exact diagonalisation in the same
active space; `CASCI(10e,8o)` computed through this path reproduces §9's stored
energies exactly. Orbital ordering changes the cost substantially — a fourfold
reduction in required bond dimension at 12 orbitals — and changes nothing about
the converged energy, as it cannot.

Above the wall, `CAS(10e,26o)` is frozen-core FCI in cc-pVDZ (4.3 × 10⁹
determinants) and `CAS(14e,28o)` is full CI (1.4 × 10¹²). Comparing NEVPT2
against the latter and not the former is deliberate: PySCF's NEVPT2 takes its
core from the CASSCF object and generates excitations out of it, so it contains
core correlation the frozen-core rung does not. That difference is 3–4 mHa,
measured two independent ways — directly as `E(28o) − E(26o)` (−3.78, −3.07,
−3.09 mHa) and beforehand from frozen-core against all-electron CCSD(T) (−3.89
and −3.15 mHa where CCSD(T) still converges) — and it is larger than the
agreement under discussion, so ignoring it would have compared two different
quantities.

NEVPT2 recovers 81–83% of the correlation outside `CAS(10e,8o)`, missing about
20 kcal/mol at all three geometries. Its error is 31.7, 32.2 and 35.8 mHa at
1.098, 1.600 and 2.400 Å. **Only the last step is resolved:** the first two
differ by 0.5 mHa against a 1.5 mHa uncertainty and are indistinguishable, while
the rise at 2.4 Å is about 2.5× the uncertainty there. The claim supported is
that the error is flat from equilibrium to 1.6 Å and rises modestly at 2.4 Å.

### What they do not support

- **Nothing about the physical N₂ molecule.** cc-pVDZ is not basis-converged.
  Every figure here is a statement about correlation treatment at a fixed basis.
- **The external check does not cover the hard geometry.** Three of the four
  points above are validated against published near-exact values, but 2.4 Å —
  the most strongly multireference geometry, and the one where the protocol is
  most likely to fail — is outside the published curve and is not checked. The
  agreement at 1.098 Å, 1.600 Å and 2.118 a₀ does not transfer to it, and a run
  at the curve's last point (4.50 a₀ = 2.381 Å) would be needed to test it.
- **The references are near-exact, not exact.** CDFCI and Chan's DMRG agree to
  1.6 × 10⁻⁵ Ha and that spread is carried as their uncertainty; SHCI's value is
  stochastic and not variational.
- **No machine-learning claim.** Three bond lengths are not a surface, and no
  model in this repository was trained on any tensor-network data; see §8 for
  what the surrogates were trained on and why the DMRG tier was not used.
- **Wall times are indicative.** Phase 1 timings were taken under varying load
  and are not benchmark figures; the canonical/Fiedler ratio within a rung is
  the meaningful comparison, not the seconds.

## 11. Limitations

- **Minimal-basis results are not chemistry.** FCI is exact for the stated
  basis, electron number and symmetry sector only. No energy here is a
  chemically converged bond energy.
- **DMRG is used as an exact solver on H₁₀, and as an active-space method only
  on N₂.** In realistic calculations, correlation involving orbitals outside the
  active space must be recovered separately. §9 shows what that step is worth
  perturbatively and §10 measures it variationally; the H₁₀ work does not
  include it at all.
- **The N₂ upper rungs have no external reference.** `CAS(10e,26o)` and
  `CAS(14e,28o)` rest entirely on the convergence protocol in §10 — a χ
  sequence, a two-ordering cross-check and the residual sweep drift. None of
  those bounds the true error; they bound the spread of what was tried. A
  published FCI value for N₂/cc-pVDZ would be a real external check and has not
  been sought.
- **A sweep schedule in this repository produced a confidently wrong answer**
  (§10): 0.19 Ha above the ground state, reproducible across seeds, with a
  discarded weight of 5 × 10⁻¹⁰. It was found only because that active space
  still had an exact reference. The defect is fixed and guarded by tests, but
  the general lesson stands — above the exact wall, agreement between two
  orbital orderings was the only diagnostic that worked.
- **Interpolation on one surface is not transferability.** The surrogate study
  interpolates and extrapolates within a single H₁₀ potential energy surface in
  two internal coordinates. It is silent about transfer to other molecules,
  other bases, or other chemical environments.
- **Discarded weight is not calibrated**, and in this data is not monotone in
  the energy error (§4).
- **The χ = 64 tier is a local-minimum lottery**, not a converged low-fidelity
  method. Tightening the Davidson threshold moved individual χ = 64 energies by
  up to 1.1 × 10⁻² Ha, and a low-χ warm-up ramp makes every seed converge to the
  *same* local minimum — reproducible, and about twice as far from FCI as a flat
  χ = 64 start.
- **The Giner et al. comparison is qualitative.** That work is paywalled; its
  dimerisation convention has not been verified against the published text.
- **No model was trained on tensor-network data.** The surrogates of §8 take
  two geometry coordinates as input and FCI energies as labels; DMRG-derived
  quantities were tested as *trust signals* and reported as a negative result,
  never used as features or labels. The N₂ ladder of §10 has no
  machine-learning component at all.
- **Nothing here is novel.** Hydrogen chains, orbital-entanglement diagnostics,
  Δ-learning, mutual-information orbital ordering and DMRG in large active
  spaces are all established; see `literature-review.md`.

## 12. Reproducibility

The environment is locked with `uv` (`.python-version`, `pyproject.toml`,
`uv.lock` are committed). Every calculation writes a structured record carrying
geometry, units, method, basis, convergence status, energies, wall time,
hardware identifier and package versions. Expensive results are cached and never
silently overwritten, and figure scripts read only cached data — they never
launch an electronic-structure calculation. Random seeds are recorded. The test
suite separates fast convention checks from tests that run calculations
(`pytest -m "not slow"`).

Wall times are only comparable within one hardware identifier, and for the N₂
vignette only at the thread count recorded with them.
