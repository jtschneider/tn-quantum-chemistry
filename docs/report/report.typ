#set document(title: "Tensor networks and molecular electronic structure")
#set page(paper: "a4", margin: (x: 2.4cm, y: 2.6cm), numbering: "1")
#set text(size: 10pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#show heading.where(level: 1): it => block(above: 1.6em, below: 0.9em)[#it]
#show raw: set text(size: 0.92em)
#show link: set text(fill: rgb("#1c5cab"))

#align(center)[
  #text(16pt, weight: "bold")[
    Tensor networks and molecular electronic structure
  ]
  #v(0.2em)
  #text(12pt)[A validated study of the H#sub[10] chain]

  #v(1em)
  Jan Thorben Schneider

  #v(0.8em)
  #text(9pt, style: "italic")[Technical report \ #datetime.today().display("[day]/[month]/[year]")]
]

#v(1em)

#block(inset: (x: 1.2em), [
  #text(weight: "bold")[Abstract.]
  This is a methodological study. We build an end-to-end pipeline linking
  tensor-network electronic structure to a machine-learned energy surface, and
  deliberately choose a system simple enough, the linear H#sub[10] chain in a
  minimal basis, that every step remains checkable against an exact answer.

  *Electronic structure.* Restricted Hartree--Fock, second-order Møller--Plesset
  perturbation theory, coupled cluster with singles and doubles and its
  perturbative-triples correction, full configuration interaction by Davidson
  diagonalisation, and SU(2)-adapted DMRG @white1992 with a matrix-product-operator
  Hamiltonian. Both correlated solvers consume a single integral tensor, so that any
  disagreement is attributable to the solvers rather than to their inputs.
  From the converged matrix product state we extract symmetry-conserved quantum-number-resolved Schmidt spectra, single-orbital entropies and two-orbital mutual information
  @legeza2003 @rissler2006, together with the exact decomposition of the
  bipartite entropy into charge and configurational parts.

  *Learning.* A two-parameter $(R, delta)$ surface of 77 geometries at seven
  levels of theory. Four models are compared at matched label budgets, splits and
  random seeds: a constant correction, a tensor-product B-spline, and a
  multilayer perceptron trained either on the correlated energy directly or on
  the correlation energy alone ($Delta$-learning, @ramakrishnan2015). Evaluation
  uses repeated random splits and blocked-coordinate holdouts that force
  extrapolation, learning curves in the label budget, automatic-differentiation
  derivatives of the fitted surface, and seed-ensemble spread as an uncertainty
  estimate.

  *What the system is for.* The physics here is textbook and no result is offered
  as new chemistry. The contribution is the protocol, and three measurements it
  makes falsifiable: that entanglement entropy predicts how hard a state is to
  represent but not the bond dimension a target *energy* accuracy needs, the
  conversion between the two varying by two orders of magnitude across the
  surface; that $Delta$-learning helps or hurts according to a measurable
  property of the target, losing by an order of magnitude where the correlation
  energy is the less smooth of the two; and that entanglement diagnostics do not
  identify where the fitted surface is untrustworthy, being beaten by the spread
  of a model ensemble that costs nothing.
])

= Introduction

Density matrix renormalisation group methods @white1992 have become standard for
strongly correlated molecular systems @hachmann2006, and their diagnostics ---
orbital entropies and mutual information @legeza2003 @rissler2006 @boguslawski2015
--- are widely used to characterise electronic structure. Separately,
$Delta$-learning @ramakrishnan2015 is a standard device for learning correlated
energies from cheap baselines. This report joins the two in one pipeline and
examines what the tensor-network half actually contributes to the learning half,
on a system small enough that every step can be checked against an exact answer.

*Nothing here is new physics or a new method.* Hydrogen chains are a
long-established benchmark @motta2017 @giner2013; the entanglement diagnostics
are standard @legeza2003 @rissler2006; $Delta$-learning is established
@ramakrishnan2015. The point of using a toy system is precisely that it removes
any excuse for an unverified claim: the contribution is the protocol --- matched
budgets, blocked holdouts, free baselines that a proposed signal must beat --- and
the willingness to report what it measured when the answer was unwelcome.

== What is claimed and what is not

We claim: that the energies reported are converged to the stated tolerances
against two independent solvers; that the comparisons between learning models are
matched in label budget, split and seed; and that the negative results are
measurements rather than failures to tune.

We do not claim: chemical accuracy for the *molecule* (a minimal basis is exact
for a model, not for H#sub[10]); transferability to other systems; any novel
method; or any acquisition or active-space-selection rule.

= Model and conventions

The system is a linear chain of ten hydrogen atoms in the STO-3G basis, with
nearest-neighbour spacing $R$ and a bond-alternation parameter $delta$, so that
successive bond lengths are $R plus.minus delta/2$. This gives a two-parameter
family with a genuine Peierls-type distortion at $delta eq.not 0$
@giner2013. Restricted Hartree--Fock canonical orbitals define the active space;
no orbitals are frozen, so the active space is exactly ten orbitals and ten
electrons at half filling.

Conventions fixed throughout, and asserted by tests rather than by review:
energies in hartree, distances in ångström, $C_1$ point group (block2 runs
without spatial symmetry), and the same integral tensor --- identified by a
SHA-256 digest --- handed to both solvers, so that a disagreement can never be
attributed to the two codes having received different Hamiltonians.

= Methods

== Electronic structure

Seven levels of theory are computed at every geometry: restricted Hartree--Fock,
MP2, CCSD, CCSD(T), full configuration interaction, and DMRG at two bond
dimensions. The two correlated solvers that must agree are exact diagonalisation,

$ E_"FCI" = min_(bold(c)) (bold(c)^dagger bold(H) bold(c)) / (bold(c)^dagger bold(c)), $

obtained by Davidson iteration in the full $M_S = 0$ determinant space, and
SU(2)-adapted DMRG,

$ E_"DMRG" (chi) = min_(psi in cal(M)_chi)
     (⟨ psi, hat(H) psi ⟩) / (⟨ psi, psi ⟩), $

minimised over matrix product states $cal(M)_chi$ of bond dimension at most
$chi$, with the Hamiltonian represented as a matrix product operator. Working in
the spin-adapted basis confines the calculation to the singlet sector by
construction, where the determinant-space solver sees the whole $M_S = 0$ space
including triplets.

#figure(
  image("/figures/h10_dissociation.svg", width: 100%),
  caption: [The dissociation curve across the method ladder (left) and the signed
    error against exact diagonalisation (right). The error axis is
    symmetric-logarithmic below the two-solver floor, which is shaded. Points
    where coupled cluster failed to converge are omitted rather than plotted at
    whatever the solver returned.],
) <fig-dissociation>

Both solvers receive one integral tensor, identified by a SHA-256 digest that is
re-checked after each run --- the DMRG backend writes into arrays it is handed,
so the digest also guards against a solver silently perturbing the other's input.

== The N#sub[2] active-space ladder

H#sub[10] in a minimal basis has no orbitals outside its active space, so DMRG
there can only reproduce exact diagonalisation. N#sub[2] in cc-pVDZ does: after
freezing the two nitrogen $1s$ cores, 26 orbitals hold 10 valence electrons.
Six nested active spaces are computed at three bond lengths,

$ "CAS"(10e, 8o) subset "CAS"(10e, 12o) subset dots subset "CAS"(10e, 26o)
  subset "CAS"(14e, 28o), $

spanning $3.1 times 10^3$ to $1.4 times 10^(12)$ determinants. Exact
diagonalisation is affordable to 16 orbitals and no further, which splits the
ladder into an overlap region where both solvers run and a DMRG-only region
above it, on the same molecule, basis and orbitals. `CAS`(10e, 26o) is
frozen-core FCI in this basis and `CAS`(14e, 28o) is full CI.

Two settings are fixed by measurement rather than convention. The Davidson
subspace is sized against a memory budget: the project default of 200 vectors,
chosen when a CI vector was 0.5 MB, would ask for 61 GB at 16 orbitals. Halving
the subspace that is used, from 28 to 16, moves the 16-orbital reference by less
than $2 times 10^(-13)$ hartree, so it is converged. The self-consistent field
that *defines* the orbitals runs single-threaded: on 32 threads it returns
different orbitals run to run, and while the SCF energies agree to
$3 times 10^(-13)$ hartree, the resulting active-space Hamiltonians differ enough
to move the CASCI energy by $6.4 times 10^(-10)$ hartree. Since CASCI is not
stationary with respect to orbital rotations, that noise enters the reference
directly.

Orbitals are ordered along the matrix product state in two ways --- by orbital
energy, and by the mutual-information (Fiedler) ordering implemented in `block2`
@zhai2023block2, which is standard practice in production DMRG codes. A permutation
of orbitals cannot move an eigenvalue, so the two differ only in representational
cost; that they must agree is used below as a diagnostic.

== Entanglement diagnostics

From the converged matrix product state we extract the quantum-number-resolved
Schmidt spectrum at every bond, the single-orbital entropy $s_i$, and the
two-orbital mutual information @legeza2003 @rissler2006

$ I_(i j) = 1/2 (s_i + s_j - s_(i j)), quad I_(i i) = 0. $

The bipartite von Neumann entropy decomposes exactly into a charge part and the
entanglement remaining once the charge is known,

$ S(l) = S_"number" (l) + sum_N p_(l,N) thin S_N (l), $ <eq-sumrule>

which is used as a numerical invariant: its three terms come from the pooled
spectrum, the charge marginal and the per-charge spectra, so none can be wrong in
isolation while it holds. It also separates two effects a single entropy
conflates --- stretching H#sub[10] from $R = 1.0$ to $2.4$ Å multiplies the
number entropy by 3.2 but the configurational part by 8.7, as entanglement moves
*inside* fixed charge sectors. These quantities are basis-dependent and are
computed in both canonical and Löwdin-orthogonalised bases to make that explicit.

#grid(columns: (1fr, 1fr), gutter: 8pt,
  figure(image("/figures/h10_orbital_entropy.svg", width: 100%),
    caption: [Single-orbital entropy.]),
  figure(image("/figures/h10_mutual_information.svg", width: 100%),
    caption: [Two-orbital mutual information.]),
)

== Learning protocol

The dataset is the $(R, delta)$ grid of 77 geometries. Exact diagonalisation
converged at all 77; coupled cluster converged at 43, and its failures cluster at
stretched and positively dimerised geometries rather than scattering --- which
constrains the split design, since a $delta$-blocked holdout would load one fold
with nearly all of them.

#figure(
  image("/figures/h10_surface.svg", width: 92%),
  caption: [The $(R, delta)$ energy surface and the per-geometry convergence
    status of each method. The coupled-cluster failures cluster rather than
    scatter, which is what constrains the split design.],
) <fig-surface>

Four models are compared: a constant correction, a tensor-product B-spline whose
capacity is tied to the training-set size, and a multilayer perceptron trained
either on $E_"FCI"$ directly or on $E_"FCI" - E_"RHF"$ with the baseline restored
at prediction time. The constant correction is $Delta$-learning with zero
capacity and exists to make the $Delta$-learning claim falsifiable.

Every model at a given split and budget receives the *same* drawn training
indices, so matched budgets are a property of the code rather than of
discipline. Evaluation uses five repeated random splits and three
blocked-coordinate holdouts in $R$ --- both ends of the range and one interior
gap --- so that interpolation and extrapolation are reported separately.
Learning curves run over eight label budgets; three model seeds per point give an
ensemble spread used later as an uncertainty estimate. Derivatives of the fitted
surface are obtained by automatic differentiation and checked against central
differences.

= Numerical convergence

This section documents the convergence checks behind every number that follows.
It contains no findings; it is the due diligence the rest of the report depends
on, recorded because the settings are not the defaults and the reasons are worth
stating.

== Agreement between solvers

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: (x, y) => if y == 0 { (bottom: 0.6pt) } else { none },
    [*System*], [$|E_"DMRG" - E_"FCI"|$], [$⟨ S^2 ⟩$],
      [sum rule], [Schmidt norm],
    [N#sub[2], 1.1 Å], [$1.7 times 10^(-13)$], [$4.5 times 10^(-17)$],
      [$2.2 times 10^(-13)$], [$1.8 times 10^(-14)$],
    [H#sub[10], $R = 1.0$ Å], [$1.1 times 10^(-13)$], [$1.8 times 10^(-16)$],
      [$2.2 times 10^(-16)$], [$8.9 times 10^(-16)$],
    [H#sub[10], $R = 2.4$ Å], [$2.5 times 10^(-13)$], [$1.7 times 10^(-15)$],
      [$8.9 times 10^(-16)$], [$6.7 times 10^(-16)$],
  ),
  caption: [Agreement between the two solvers, with the spin sector, the entropy
    sum rule of #ref(<eq-sumrule>) and the Schmidt normalisation certified
    alongside the energy. Hartree or dimensionless.],
) <tab-gate>

Across 169 comparisons at converged bond dimension the two solvers agree to
$2.3 times 10^(-12)$ hartree in the worst case and $1.8 times 10^(-13)$ in the
median. That worst case is quoted throughout as the floor below which the two
methods cannot be distinguished.

== Solver settings

Three settings differ from the defaults of the underlying packages, each because
the default was measurably insufficient on this system.

The Davidson subspace of the determinant-space solver was raised from 12 to 200.
At stretched geometries the low-lying spectrum is dense --- at $R = 3.4$ Å the
nearest state to the singlet ground state is a *triplet* $4.6 times 10^(-5)$
hartree above it, while the nearest singlet is $2.6 times$ further --- and a
12-vector subspace restarts repeatedly and loses the directions it needs. The
progression at that geometry is not monotone, so agreement between two nearby
subspace sizes is not evidence of convergence:

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, right, right),
    stroke: (x, y) => if y == 0 { (bottom: 0.6pt) } else { none },
    [*Subspace*], [*Error*], [*Wall time*],
    [12 (package default), `conv_tol` $10^(-12)$], [$2.52 times 10^(-9)$], [14.9 s],
    [12, `conv_tol` $10^(-14)$], [$2.42 times 10^(-11)$], [23.3 s],
    [30], [$3.91 times 10^(-12)$], [7.0 s],
    [120], [$3.59 times 10^(-12)$], [6.8 s],
    [200], [$0$ (best)], [8.0 s],
    [300], [$0$ (best)], [7.8 s],
  ),
  caption: [Convergence of the determinant-space reference with Davidson subspace
    size at $R = 3.4$ Å. The wider subspace is also faster, converging rather
    than restarting. Two further knobs were measured and change nothing: the
    preconditioner space (400 versus 4000) and a singlet-adapted solver, which
    returns identical energies about 40% slower.],
) <tab-davidson>

The DMRG local-eigensolver threshold and sweep stopping tolerance are both
$10^(-14)$ throughout. A loose local solver is indistinguishable from
bond-dimension truncation in the final energy, so leaving it at a default while
tightening everything else confounds the two. This is enforced by a test:
no schedule may ship a looser threshold and no script may assemble its own.

== Behaviour of the bond-dimension convergence

Three properties of this system's DMRG convergence bear on how the results below
should be read.

*The error moves in plateaus.* Adjacent bond dimensions give energies agreeing to
$tilde 10^(-13)$ over wide ranges and then jump by $10^(-4)$; at $R = 3.0$ Å the
plateau extends from $chi = 128$ to 256 and breaks only at 384. The plateaus are
variational rather than an artefact of the optimiser: they survive a low-$chi$
warm-up ramp, are unchanged at 14, 30 or 60 sweeps, and are reproduced when the
optimisation is started from the exact state compressed to that bond dimension.
Compressing the converged $chi = 384$ state to $chi = 192$ costs
$1.28 times 10^(-2)$ hartree, confirming no state of that bond dimension holds
the exact answer. The practical consequence is that comparing two neighbouring
bond dimensions can report agreement to $10^(-13)$ while sitting $10^(-4)$
hartree away.

*The discarded weight is not an error bar.* Within one geometry the expected
proportionality holds --- at $R = 1.0$ Å the ratio of energy error to discarded
weight is 0.11, 0.057, 0.055 across $chi = 128, 192, 256$. Across geometries it
does not, for reasons quantified in #ref(<sec-entropy>). In one case at
H#sub[12] a run reports a discarded weight of $3.3 times 10^(-7)$, below its own
requested cutoff, while its energy is $4.2 times 10^(-3)$ hartree wrong.

*Low bond dimensions are seed-dependent at stretched geometries.* Three
matrix-product-state initialisations agree to within $1.3 times$ out to
$R = 2.2$ Å, but spread by $10.3 times$ at $R = 2.6$ Å and $58.7 times$ at
$R = 3.0$ Å. No low-$chi$ tier is therefore used as a fidelity level anywhere in
this work.

#figure(
  image("/figures/h10_chi_convergence.svg", width: 100%),
  caption: [Error against bond dimension, one panel per spacing. Line is the
    median of three initialisations, band their full range, dashed line the
    two-solver floor.],
) <fig-chi>

= Findings

== Entropy predicts representational cost, not energetic cost <sec-entropy>

Let $chi^*$ be the smallest bond dimension reaching $10^(-9)$ hartree against the
reference. Measured this way the entropy is uninformative: $chi^*$ is U-shaped
(384, 192, 192, 256, 256, 384 across $R = 1.0$ to $3.0$ Å) while the entropy
rises monotonically (0.84 to 4.67), and the Pearson correlation of $ln chi^*$
with $S$ is $-0.08$.

That is not a failure of the standard argument, but of the question. Taken from
the exact Schmidt spectrum, the number of states required for a fixed
*truncation* error does follow the entropy: the correlation of $ln n$ with $S$ is
$+0.85$ at a discarded weight of $10^(-6)$ and $+0.86$ at $10^(-9)$. Rényi orders
$alpha = 1/2, 2, infinity$ behave the same, so the choice of entropy measure is
not the issue.

What separates the two is the conversion from discarded weight to energy, and it
is not a constant of the method:

#figure(
  table(
    columns: 7,
    align: (left, right, right, right, right, right, right),
    stroke: (x, y) => if y == 0 { (bottom: 0.6pt) } else { none },
    [$R$ (Å)], [1.0], [1.4], [1.8], [2.2], [2.6], [3.0],
    [$Delta E slash w$], [0.109], [0.125], [0.033], [0.0032], [0.0020], [0.00082],
  ),
  caption: [Energy error per unit of discarded weight at a common bond dimension
    of 128 --- a monotone fall of $133 times$ across the surface.],
) <tab-conversion>

Discarding a given amount of Schmidt weight costs two orders of magnitude more
energy at the compressed end than at the stretched end. This is consistent with
the composition of the discarded tail, which at stretched geometries is dominated
by near-degenerate spin configurations: they carry entropy but almost no energy.
It is also the same fact as the discarded-weight caution above, seen from the
other side.

The usable statement is therefore narrower than the one usually made: entropy is
a good guide to how hard a state is to *represent*, and not a guide to the bond
dimension a target *energy* accuracy demands, unless the energy per unit of
discarded weight is known to be comparable across the systems being compared.

#figure(
  image("/figures/h10_chi_summary.svg", width: 100%),
  caption: [Bond dimension for a fixed energy target, entanglement, and
    reproducibility across seeds, against chain spacing.],
) <fig-chisummary>

== $Delta$-learning is conditional, and the condition is measurable

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    stroke: (x, y) => if y == 0 { (bottom: 0.6pt) } else { none },
    [*Split*], [direct], [$Delta$], [ratio],
    [random (5 draws)], [$2.9 times 10^(-3)$], [$4.9 times 10^(-4)$], [0.17],
    [blocked, compressed], [$2.0 times 10^(-2)$], [$3.1 times 10^(-3)$], [0.16],
    [blocked, interior gap], [$4.1 times 10^(-4)$], [$3.6 times 10^(-4)$], [0.87],
    [blocked, stretched], [$5.2 times 10^(-4)$], [$6.7 times 10^(-3)$], [13.0],
  ),
  caption: [Test mean absolute error in hartree at the largest label budget.],
) <tab-surrogate>

$Delta$-learning wins on two splits, ties on one, and loses by an order of
magnitude on the stretched block. The condition separating these is a property of
the targets that can be measured in advance: the correlation energy carries about
half the curvature of the total energy per unit of its own range in the
compressed and middle regions, and $2.8 times$ more in the stretched region,
where the total energy flattens toward dissociation while the correlation energy
is still changing rapidly. $Delta$-learning helps where the baseline's error is
the smoother of the two targets, and not otherwise. It is not simply a smaller
target: the $Delta$ surface spans a wider range than the total energy.

#figure(
  image("/figures/h10_surrogate_learning_curves.svg", width: 100%),
  caption: [Learning curves at matched label budgets. The two neural models share
    an architecture and differ only in whether the baseline is subtracted.],
) <fig-learning>

== Entanglement diagnostics do not identify where the surrogate fails

The surrogate consumes geometry alone, so the natural route to making the tensor
network load-bearing is for an entanglement descriptor to flag untrustworthy
predictions. Both surrogates were fitted at their largest budget on all eight
splits with three seeds, the per-geometry held-out error recorded, and each
candidate signal ranked against it. Correlations are taken *within* a split
before pooling, because the errors differ by two orders of magnitude between the
random and blocked splits.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: (x, y) => if y == 0 { (bottom: 0.6pt) } else { none },
    [*Signal*], [direct], [$Delta$], [direct, $R$ controlled], [$Delta$, controlled],
    [max bipartite entropy], [$-0.596$], [$0.098$], [$-0.287$], [$0.146$],
    [configurational entropy], [$-0.598$], [$0.096$], [$-0.364$], [$0.112$],
    [*ensemble disagreement*], [*0.760*], [*0.622*], [*0.526*], [*0.599*],
    [distance to training set], [$0.340$], [$0.460$], [---], [---],
    [spacing $R$], [$-0.510$], [$0.120$], [---], [---],
  ),
  caption: [Pooled within-split Spearman correlation with the absolute held-out
    error over 117 points. Ensemble disagreement and the two geometric signals
    cost nothing; the entanglement descriptors require a converged DMRG run.],
) <tab-trust>

The spread of a three-seed model ensemble --- already computed, therefore free
--- is the strongest signal for both models, before and after controlling for the
spacing. About half of the apparent entanglement signal is the spacing itself,
which the model already consumes. For $Delta$-learning, the model that actually
fails, the entanglement correlation is indistinguishable from noise. The residual
changes sign between the two models, so it tracks the validity of the
Hartree--Fock *baseline* rather than the trustworthiness of the surrogate; a
signal that reverses depending on which model it is asked about is not a trust
signal.

== System shape, not system size, decides whether DMRG is economical

On this system the tensor network is never the cheaper route to a label: a
converged DMRG run costs 5.4 s against 2.4 s for exact diagonalisation. The
standard expectation is that this reverses with size, since the determinant count
grows combinatorially while the bond dimension needed for fixed accuracy need not.

Exact diagonalisation does run out first, as arithmetic requires: one Davidson
vector is 1.3 GB at H#sub[16] and 18.9 GB at H#sub[18]. But at H#sub[12] and
$R = 3.0$ Å, against a reference costing 44 s, DMRG plateaus at
$2.5$--$4.2 times 10^(-3)$ hartree and stays there, unchanged at 14, 30 or 60
sweeps and 366 s at a bond cap of 512. The surrogate of #ref(<tab-surrogate>) has
a test error of $4 times 10^(-4)$ hartree, so such labels would be an order of
magnitude worse than the model trained on them --- and beyond H#sub[16] there is
no reference against which to notice.

The reason is the shape of the system rather than its size. H#sub[$n$] in a
minimal basis has exactly $n$ orbitals and $n$ electrons: half filling, maximal
entanglement per site, and no weakly correlated orbitals to compress. Production
DMRG wins in the opposite regime, with many orbitals carrying structured
entanglement @hachmann2007 @larsson2022. Lengthening the chain moves along the
unfavourable shape rather than out of it, which is a design lesson for choosing a
benchmark system rather than a property of the method.

A related correction is recorded. An earlier form of this argument held that
stretched chains are gapped one-dimensional systems obeying an area law, so the
bond dimension should saturate. That is false for the *uniform* chain, which
approaches a Heisenberg antiferromagnet and is critical, its entanglement growing
logarithmically in length; the gap in this system class comes from dimerisation.
Consistent with that, at $R = 3.0$ Å a dimerisation of $delta = 0.4$ Å lowers the
maximum bipartite entropy from 4.671 to 3.970 at H#sub[10]. Whether it saturates
with length could not be established, because the longer dimerised chains did not
converge well enough to tell.

== Above the exact wall, the only working convergence check was two orbital orderings <sec-ladder>

Below 16 orbitals, DMRG reproduces exact diagonalisation in the same active
space and the cost of doing so depends strongly on orbital ordering. Writing
$chi^*$ for the smallest bond dimension within $10^(-9)$ hartree of the exact
answer, the mutual-information ordering needs a quarter of the bond dimension at
12 orbitals, and at 16 the orbital-energy ordering does not reach that accuracy
anywhere on the grid tried.

Above 16 orbitals there is no reference, and the protocol has to carry the
claim. What that protocol is worth was established by a failure. With the
mutual-information ordering at 16 orbitals, an earlier sweep schedule converged
to a state $0.19$ hartree above the ground state --- at every bond dimension and
both seeds at $1.6$ Å --- while reporting a final discarded weight of
$5 times 10^(-10)$. At $2.4$ Å, raising $chi$ from 500 to 1000 at fixed seed moved
the answer away from the correct one. Three controls isolated the cause: exact
diagonalisation of the permuted integrals reproduced the canonical energy to
$2.8 times 10^(-14)$ hartree, so the Hamiltonian was right; a flat schedule at the
same seed and $chi$ converged correctly, so the ordering was not to blame; and a
schedule whose perturbative noise extended into the full-$chi$ sweeps also
converged. The defective ramp had spent its entire noise budget while $chi$ was a
sixteenth of its target, reaching full bond dimension with no mechanism left to
leave a local minimum.

The diagnostics available without a reference all certified that state:
discarded weight $5 times 10^(-10)$, seeds agreeing to $10^(-12)$, and an
entanglement entropy lower than the correct state's at one geometry and higher at
another. Only the disagreement between the two orbital orderings --- $0.19$
hartree --- revealed it. This is a sharper form of the discarded-weight caution
of #ref(<sec-entropy>): here that quantity was wrong by seven orders of
magnitude. Uncertainties on the upper rungs are therefore taken as the largest of
the final $chi$ step, the ordering disagreement, and the residual sweep drift,
and the ordering term dominates at every point.

== The convergence protocol tested against published near-exact energies <sec-external>

The protocol of #ref(<sec-ladder>) produces an uncertainty precisely where
nothing in this work can check it. Three published near-exact energies for
N#sub[2] in cc-pVDZ now test it. Coordinate-descent full CI @wang2019cdfci and
the DMRG binding curve of Chan, Kállay and Gauss @chan2004 give the all-electron
value at $2.118 a_0$ and agree with each other to $1.6 times 10^(-5)$ hartree ---
37 to 73 times finer than the uncertainties under test, and 39 times finer at the
geometry actually run here, which makes it a test rather than a coincidence. Semistochastic heat-bath CI @sharma2017shci gives a
frozen-core value at $1.0977$ Å.

The Hamiltonian was matched before the energies were compared. RHF at $2.118 a_0$
through the same path used throughout reproduces the published $-108.9493779$
hartree to $2.1 times 10^(-8)$, that value's own rounding, which fixes basis,
geometry, unit and the spherical-against-Cartesian $d$ convention at once ---
cc-pVDZ with Cartesian $d$ functions gives 30 orbitals rather than 28, converges
normally, and is indistinguishable in a log file.

Two obstacles were handled rather than assumed away. The sources quote different
units, $2.118$ Bohr and $1.0977$ Å, which are two geometries $0.023$ Å apart and
not one value written twice. And neither is on this work's grid, so the
all-electron comparison was made by running the ladder at the published bond
length --- eight DMRG runs, 16.4 hours of wall time --- while the published
curve, tabulated on a $0.05 a_0$ grid near equilibrium, interpolates onto two
geometries already computed. The interpolation error is measured by leave-one-out
prediction of each interior published point rather than assumed: $10^(-5)$ hartree
near both geometries used, $6.2 times 10^(-5)$ at worst across the curve, and
conservative by construction because removing a point doubles the local spacing.
At $2.4$ Å, $4.535 a_0$, the curve has ended and extrapolation is refused; an
extrapolated reference would be this work's own smoothness assumption carrying
someone else's citation.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, left, right, right, right, right, right),
    stroke: (x, y) => if y == 0 { (bottom: 0.6pt) } else { none },
    [*Geometry*], [via], [$E_"DMRG"$], [published], [error], [$sigma$],
      [$sigma \/ |"err"|$],
    [$2.118 a_0$], [run here], [$-109.281909$], [$-109.282173$],
      [$+2.64 times 10^(-4)$], [$7.9 times 10^(-4)$], [$2.98$],
    [$1.098$ Å], [curve], [$-109.280658$], [$-109.280913$],
      [$+2.54 times 10^(-4)$], [$7.3 times 10^(-4)$], [$2.87$],
    [$1.600$ Å], [curve], [$-109.083789$], [$-109.084270$],
      [$+4.81 times 10^(-4)$], [$1.5 times 10^(-3)$], [$3.02$],
    [$2.400$ Å], [---], [$-108.965255$], [---], [---],
      [$1.4 times 10^(-3)$], [---],
  ),
  caption: [The `CAS`(14e, 28o) rung against published near-exact energies.
    Hartree. Every comparison falls inside the quoted uncertainty and every DMRG
    energy lies above the reference, as a variational bound must.],
) <tab-external>

The uncertainty is honest and loose by a consistent factor of three: the ratio of
the quoted band to the error actually made is $2.87$, $2.98$ and $3.02$ across
three geometries and two independent routes to the reference, and the $chi$ step
rather than the ordering term supplies that margin in all three.

The ordering gap taken alone tells a sharper and less comfortable story. It is
$1.05$, $1.03$ and $0.68$ times the true error --- at two geometries predicting it
to within five percent, and at $1.600$ Å *under*-estimating it by half again. The
diagnostic weighted most heavily in #ref(<sec-ladder>) because it caught a real
failure is therefore also the best-calibrated estimate of the error's size, and
for that reason the most tempting quantity to quote as an error bar; these numbers
say that would be wrong. Taking the maximum over the four terms is what makes the
band a bound rather than an estimate.

The frozen-core rung agrees with heat-bath CI to $1.9 times 10^(-5)$ hartree, but
across a $0.0003$ Å geometry offset worth $3.5 times 10^(-5)$ by CCSD(T). The
agreement is smaller than the offset separating the two calculations, so the
supportable statement is that the rung agrees to better than that offset. That
value also carries a stochastic perturbative correction and is not variational,
so it is not used as a bound anywhere.

== A perturbative correction, measured against the variational answer

The reduced N#sub[2] vignette computes NEVPT2 on `CASSCF`(10e, 8o), a
perturbative estimate of the correlation lying outside that space. The
28-orbital rung measures the same quantity variationally. The comparison is made
against `CAS`(14e, 28o) rather than the frozen-core rung because NEVPT2 as
implemented excites out of the CASSCF core and so contains core correlation the
frozen-core space does not --- a difference of 3--4 mhartree, larger than the
agreement under discussion. That correction is confirmed two ways: directly as
$E(14e, 28o) - E(10e, 26o)$, giving $-3.78$, $-3.07$ and $-3.09$ mhartree, and
independently from frozen-core against all-electron CCSD(T), giving $-3.89$ and
$-3.15$ mhartree where CCSD(T) still converges.

#figure(
  table(
    columns: 5,
    align: (right, right, right, right, right),
    stroke: none,
    table.hline(),
    [*$R$ (Å)*], [*NEVPT2*], [*variational*], [*NEVPT2 $-$ full CI*], [*kcal/mol*],
    table.hline(),
    [1.098], [$-0.146323$], [$-0.174230$], [$+0.031684$], [19.9],
    [1.600], [$-0.155883$], [$-0.184996$], [$+0.032184$], [20.2],
    [2.400], [$-0.150373$], [$-0.183127$], [$+0.035839$], [22.5],
    table.hline(),
  ),
  caption: [Correlation outside `CAS`(10e, 8o) in hartree: the NEVPT2 estimate,
    the variational measurement from frozen-core FCI, and the residual against
    full CI in the same basis.],
) <tab-nevpt2>

NEVPT2 recovers 81--83% of that correlation and misses about 20 kcal/mol at
every geometry. Whether the error worsens toward dissociation is only partly
resolved: from $1.098$ to $1.600$ Å it moves by $0.5$ mhartree against an
uncertainty of $1.5$ mhartree, so those two are indistinguishable, while the rise
at $2.4$ Å is about $2.5$ times the uncertainty there. The supportable statement
is that the error is flat from equilibrium to $1.6$ Å and rises modestly at
$2.4$ Å.

The same reference makes CCSD(T)'s failure quantitative for the first time here:
it lies $+1.5$ mhartree above full CI at equilibrium, $+7.4$ at $1.6$ Å and
$+265$ at $2.4$ Å.

= Limitations

A minimal basis is exact for a model, not for the molecule: none of these
energies is chemically converged, and no basis-set extrapolation is attempted.
The N#sub[2] results are cc-pVDZ throughout and are equally statements about
correlation treatment at a fixed basis rather than about the molecule. The upper
rungs of that ladder rest on the convergence protocol of #ref(<sec-ladder>),
which bounds the spread of what was tried rather than the true error; that
protocol is tested against published near-exact energies in
#ref(<sec-external>) and holds at three geometries, over-covering by a factor of
three. The test does not reach the hard case: $2.4$ Å is the most strongly
multireference geometry here and the one where the protocol is likeliest to fail,
and it lies outside the published curve, so the agreement elsewhere does not
transfer to it. The references themselves are near-exact rather than exact. The
$(R, delta)$ surface is a two-parameter slice of the nine independent bond lengths
of an H#sub[10] chain. The learning study interpolates one surface of one system;
nothing here demonstrates transferability. Derivatives reported are
generalised-coordinate derivatives of a *fitted* surface, not Cartesian atomic
forces, and no model was trained on force labels. The entanglement diagnostics
depend on the orbital basis and are not invariants of the state. No model in
this work was trained on tensor-network data: the surrogates take two geometry
coordinates as input and exact-diagonalisation energies as labels, DMRG-derived
quantities were tested only as trust signals and reported as a negative result,
and the N#sub[2] ladder has no learning component.

= Reproducibility

Every calculation writes a structured record carrying geometry, units, method,
basis, convergence status, energies, wall time, hardware identifier, package
versions, and a SHA-256 digest of the integral tensor actually consumed. Figures
are generated only from stored data; the figure scripts perform no
electronic-structure calculation, and this is asserted by comparing checksums of
the data directory before and after. The solver-convergence conventions are
themselves under test: a regression test asserts that no sweep schedule ships a
loose local-solver threshold, that no script assembles its own, and that the
exact-diagonalisation subspace does not fall below the value established in
#ref(<tab-davidson>). Software: PySCF @sun2020pyscf and block2 @zhai2023block2.

#bibliography(
  "refs.bib",
  // style: "american-physics-society",
  style: "springer-mathphys",
)
