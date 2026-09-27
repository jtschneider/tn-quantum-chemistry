# Literature review

A novelty and validation map for this repository. Every technique used here —
DMRG on hydrogen chains, orbital-entanglement diagnostics, Δ-learning, energy
surrogates differentiated for forces — is established work. The table records
where each came from and how it is used here, so that established methods are
cited as precedent rather than presented as contributions.

**Citation provenance.** Every reference below was checked against Crossref or
arXiv metadata (title, authors, journal, volume, year) on 2026-09-05 rather than
quoted from memory. One further item — a workshop contribution on DMRG datasets
for multireference machine learning — was omitted because its record sits behind
an access wall and could not be verified.

---

## Prior art, by project component

| Component | Precedent | How it is used here |
| --- | --- | --- |
| DMRG on hydrogen chains | Hachmann, Cardoen and Chan applied quadratic-scaling ab initio DMRG to long molecules including hydrogen chains ([*J. Chem. Phys.* **125**, 144101 (2006)](https://doi.org/10.1063/1.2345196)). The Simons Collaboration later benchmarked the H₁₀–H₅₀ equation of state across many-body methods ([Motta *et al.*, *Phys. Rev. X* **7**, 031059 (2017)](https://doi.org/10.1103/PhysRevX.7.031059)). | Established benchmark. The Motta *et al.* data is open access and machine-readable, and is reproduced here to 4.9 × 10⁻⁹ Ha at all ten published spacings — the resolution limit of their printed decimals. |
| H₁₀ surface in spacing and dimerization, FCI/STO-3G | Giner, Bendazzoli, Evangelisti and Monari computed FCI/STO-3G surfaces for Hₙ rings and chains, studying Peierls dimerization and the metal–insulator transition ([*J. Chem. Phys.* **138**, 074315 (2013)](https://doi.org/10.1063/1.4792197)). | Direct prior art for the `(R, δ)` surface. That work is paywalled, so the comparison here is qualitative and its dimerization convention is **not** asserted as verified. This project uses bonds `R ± δ/2`; the conversion to a `R ± 2δ` convention is implemented and tested against bond lengths. |
| Dimerization and the metal–insulator picture | Bond-alternating hydrogen chains were studied at correlated levels decades ago ([Suhai, *Phys. Rev. B* **50**, 14791 (1994)](https://doi.org/10.1103/PhysRevB.50.14791)), and the modern ab initio phase diagram covers dimerization and magnetic phases ([Motta *et al.*, *Phys. Rev. X* **10**, 031058 (2020)](https://arxiv.org/abs/1911.01618)). | Established physics. The `δ ≠ 0` behaviour seen here is expected, not new. |
| Single-orbital entropy and mutual information | Introduced into DMRG optimisation by [Legeza and Sólyom, *Phys. Rev. B* **68**, 195116 (2003)](https://doi.org/10.1103/PhysRevB.68.195116), and developed as an orbital-interaction analysis by [Rissler, Noack and White, *Chem. Phys.* **323**, 519 (2006)](https://doi.org/10.1016/j.chemphys.2005.10.018); reviewed for quantum chemistry by [Boguslawski and Tecmer, *Int. J. Quantum Chem.* **115**, 1289 (2015)](https://doi.org/10.1002/qua.24832). | Standard QC-DMRG machinery. Computed here from reduced density matrices of the validated MPS, with invariants checked, purely as a bridge between tensor-network structure and a molecular Hamiltonian. |
| Entropy-based active-space selection | Automated active-space selection from inexpensive, partially converged DMRG entropies ([Stein and Reiher, *J. Chem. Theory Comput.* **12**, 1760 (2016)](https://doi.org/10.1021/acs.jctc.6b00156)). | Established. This repository computes the diagnostics but makes **no** acquisition or active-space-selection claim. |
| Mutual-information orbital ordering | Implemented in production DMRG codes, with theoretical justification ([Ali, arXiv:2103.01111 (2021)](https://arxiv.org/abs/2103.01111)). | Implementation context, not novelty. Orbital ordering is noted here as affecting DMRG *cost* and truncation, not the converged energy. |
| Predicting orbital entropy with machine learning | Neural prediction of DMRG single-site entropies for transition-metal active spaces ([Golub, Antalik, Veis and Brabec, *J. Chem. Theory Comput.* **17**, 6053 (2021)](https://doi.org/10.1021/acs.jctc.1c00235)). | Conceptual precedent for combining entanglement diagnostics with ML. Not attempted here. |
| Predicting mutual information with machine learning | Mutual-information matrices predicted for strongly correlated systems ([Golub, Antalik, Beran and Brabec, *Chem. Phys. Lett.* **813**, 140297 (2023)](https://doi.org/10.1016/j.cplett.2023.140297)). | Conceptual precedent. Not attempted here. |
| Machine learning across DMRG accuracy levels | Machine learning used to boost quantum chemical DMRG ([Golub, Yang, Vlček and Veis, *J. Phys. Chem. Lett.* **16**, 3295 (2025)](https://doi.org/10.1021/acs.jpclett.5c00207)). | Direct overlap with the idea of a low-χ fidelity tier. This repository measures that the χ = 64 tier on H₁₀ is seed-dependent and does not use it as a fidelity level. |
| Near-exact N₂/cc-pVDZ benchmarks | The all-electron binding curve was computed by DMRG ([Chan, Kállay and Gauss, *J. Chem. Phys.* **121**, 6110 (2004)](https://doi.org/10.1063/1.1783212)) and reproduced by coordinate-descent full CI ([Wang, Li and Lu, *J. Chem. Theory Comput.* (2019)](https://doi.org/10.1021/acs.jctc.9b00138)); a frozen-core value comes from semistochastic heat-bath CI ([Sharma *et al.*, *J. Chem. Theory Comput.* (2017)](https://doi.org/10.1021/acs.jctc.6b01028)). | External check on this project's top ladder rung, which has no reference of its own. The two all-electron methods are independent and agree to 1.6 × 10⁻⁵ Ha. Used to test the convergence protocol rather than to claim novelty: the published values are the standard, and this project's DMRG reproduces them within its own quoted uncertainty. Conventions were verified by reproducing the published RHF energy to 2.1 × 10⁻⁸ Ha. |
| Δ-learning | The Δ-machine-learning approach ([Ramakrishnan, Dral, Rupp and von Lilienfeld, *J. Chem. Theory Comput.* **11**, 2087 (2015)](https://doi.org/10.1021/acs.jctc.5b00099)). | The method under test here. Whether it helps is treated as an empirical question and measured against matched baselines, not assumed. |
| Energy-conserving forces from a learned scalar | Learning an energy and differentiating it for conservative forces is standard in ML potentials ([Chmiela *et al.*, arXiv:1611.04678 (2016)](https://arxiv.org/abs/1611.04678)). | Method used for the derivative check. No force labels were generated or trained on. |
| Multifidelity data hierarchies in quantum-chemical ML | Data hierarchies for multifidelity machine learning of excitation energies ([Vinod and Zaspel, *J. Chem. Theory Comput.* **21**, 3077 (2025)](https://doi.org/10.1021/acs.jctc.4c01491)). | Precedent for asking how labels of differing cost and accuracy should be combined. This repository does not make a label-allocation claim. |

## What this repository does not claim

- Not a novel hydrogen-chain benchmark. H₁₀ is a long-established strongly
  correlated test system, and the surface here reproduces known behaviour.
- Not a novel Δ-learning method. The contribution is a *matched* comparison
  under identical splits and label budgets, reported with its negative result
  intact where the method loses.
- Not an entanglement-based acquisition policy. The diagnostics are computed and
  their basis dependence is demonstrated; no selection rule is derived from them.
- Not chemically converged energetics. Minimal-basis FCI is exact for a model,
  not for the molecule; see `methods-and-limitations.md`.

## Software

- **PySCF** — integrals, SCF, MP2, coupled cluster, FCI, CASSCF and NEVPT2.
- **block2** — SU(2)-adapted DMRG, reduced density matrices and Schmidt spectra.
- **JAX** — the energy surrogates and their autodiff derivatives.
- **tblite** — GFN2-xTB, exercised only as an ordinary-molecule smoke test.

Versions are pinned in `uv.lock` and recorded in every result record.
