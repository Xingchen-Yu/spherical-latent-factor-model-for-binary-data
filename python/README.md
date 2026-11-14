# sphfactor: the spherical factor model in JAX

A Python/JAX port of the R + Rcpp code in this repository for

> Yu, X. & Rodríguez, A. *A Bayesian Approach to Spherical Factor Analysis for Binary Data.* arXiv:2008.05109.

Same model and the same Metropolis-within-Gibbs scheme (geodesic HMC for the
positions, random walks for κ_j, ω, τ, a Gibbs draw for λ). What changed:

* every block is one vectorised GPU call, and several chains or datasets run in one program (`sample_many`);
* gradients come from autodiff of the unconstrained log density, so the hand-derived gradient and its
  pre-built matrices are no longer needed (and nothing is divided by `x_{K+1}`);
* missing responses drop out of the likelihood instead of being imputed;
* step sizes adapt during burn-in towards 60–90% acceptance, since the right scale changes with K;
* K = 1 (the circle) is included;
* positions use the paper's coordinates (`prior.py`); `prior.from_r_coords` converts from the R convention.

The accept/reject step is evaluated in float64, the leapfrog gradients in float32. The chain is exact either
way; a less precise gradient can only lower the acceptance rate.

## Install and test

    uv sync
    uv run pytest                                              # ~3 min on a GPU, ~15 on CPU
    JAX_PLATFORMS=cpu uv run python scripts/geweke_check.py --K 3   # joint-distribution test of the whole sampler

`geweke_check.py` alternates "simulate data given parameters" with one sweep of the sampler; if every update is
right, the parameters keep their prior distribution, which is compared with exact prior draws.

## Use

```python
from sphfactor.data import load_ord
from sphfactor.model import Priors, SamplerConfig, sample

y, names, party = load_ord("S", 115)            # 96 x 599, NaN where missing
post = sample(y, K=2, config=SamplerConfig(burnin=20_000, n_samples=20_000))
post.criteria    # lppd, the paper's DIC, WAIC, in-sample accuracy
post.beta        # (draws, I, K+1) positions on the sphere
```

## Reproducing the paper's analyses

    uv run python scripts/simulation_study.py                  # spherical, K = 1..10, four scenarios + sensitivity (GPU)
    uv run python scripts/senate_study.py                      # 102nd and 115th Senates (GPU)
    JAX_PLATFORMS=cpu uv run python scripts/euclidean_study.py # Euclidean probit comparison (CPU)
    JAX_PLATFORMS=cpu uv run python scripts/make_figures.py    # figures/ and figures/criteria_table.csv

The Euclidean probit model (`euclid.py`) uses the priors of `main_script/Euclidean_nd.R`. One thing had to be assumed:
the options of the principal-nested-spheres decomposition (`analysis.pns`).

## Layout

    src/sphfactor/special.py   log CDF of the link (shared with the circular-model package)
    src/sphfactor/ghmc.py      geodesic HMC on the sphere, batched
    src/sphfactor/prior.py     spherical von Mises prior, coordinates
    src/sphfactor/model.py     the sampler
    src/sphfactor/euclid.py    Euclidean probit factor model (Albert-Chib Gibbs)
    src/sphfactor/analysis.py  principal nested spheres, Procrustes, R-hat
    src/sphfactor/data.py      .Rdata simulations and Voteview .ord files

## What the reproduction shows (runs of 2026-11-13)

Settings: 2 chains per fit, 10,000 burn-in + 20,000 kept sweeps (20,000 + 20,000 for the spherical Senate fits).
The Euclidean model uses the priors of `main_script/Euclidean_nd.R` (beta_ik ~ N(0, 1/k^2), item parameters N(0, I)).
Missing votes are left out of the log-likelihood in both models. DIC is the paper's definition (larger is better).
All numbers are in `figures/criteria_table.csv`.

| Dataset | Best spherical (K, DIC) | Best Euclidean (K, DIC) | Paper |
|---|---|---|---|
| Scenario 1 (S^2) | 2, -26,268 | 3, -26,829 | spherical K = 2: reproduced |
| Scenario 2 (S^3) | 3, -25,276 | 4, -25,771 | spherical K = 3: reproduced |
| Scenario 3 (S^5) | 5, -27,069 | 6, -28,552 | spherical K = 5: reproduced |
| Scenario 4 (R^3) | 3, -26,502 | 3, -26,415 | Euclidean K = 3, spherical close behind: reproduced |
| 102nd Senate | 3, -16,488 | 3, -16,464 | Euclidean K = 3 best: a tie here (see below) |
| 115th Senate | 2, -6,671 (K = 1: -6,904) | 2, -7,297 | spherical K = 2, then K = 1: reproduced |

Also reproduced:

* The Euclidean model needs one more dimension than the sphere on spherical data (K = 3, 4, 6).
* Accuracy and principal-nested-spheres elbows at the true dimension (`figures/sim_scenarios.png`).
* The alternative hyperprior gives the same DIC choice, accuracy and positions (`figures/sim_sensitivity*.png`).
* Principal nested spheres (great subspheres, K = 8 fit): the first component explains 96% for the 115th Senate and
  the first three 99%; for the 102nd Senate the first three explain 67% (paper: >95%, close to 99%, <70%).
* 1D Euclidean vs circular ranks: agreement in the 102nd Senate, large disagreements among Republicans in the 115th
  (`figures/senate_ranks.png`, `figures/senate_ranks.txt`).

Differences from the paper:

* **102nd Senate.** Euclidean and spherical K = 3 are indistinguishable: the means differ by 24 and the two
  Euclidean chains by 73.
* **Scenario 1 coverage** (original data, truth from `data/o1_2_sphdata.Rdata`): 89% (dimension 1) and 64%
  (dimension 2) when the posterior is aligned with the truth by axis reflections only, 94% and 58% after a Procrustes
  rotation; the paper reports 98% and 68%. The dimension-2 shortfall is systematic: the posterior is more spread out
  than the truth (`figures/sim_recovery_sim_2_reflection.png`).

DIC and WAIC (`figures/criteria_dic_waic.png`):

* WAIC ranks the two geometries the same way as DIC at the selected dimensions in every dataset.
* WAIC does not select a dimension: in the simulations it flattens at the true dimension but does not come down, and
  for the Senates it keeps rising to K = 10. Its pointwise penalty saturates once the extra dimensions are shrunk,
  while the DIC penalty 2 Var(log-likelihood) keeps growing.
* The DIC penalty is estimated from one chain's log-likelihood trace and is noisy: chains differ by 10 to 80 at the
  selected dimensions and by several hundred at some larger K (`dic_sd` column).

Sensitivity of the comparison to choices that are easy to get wrong:

* **Euclidean prior.** Without the k^2 shrinkage (beta_i ~ N(0, I); `--no-shrink`) the Euclidean model has the higher
  DIC in Scenarios 2 and 3 (by about 200), i.e. the paper's conclusion reverses there.
* **Missing votes.** The original `Euclidean_nd.R` summed its log-likelihood trace over imputed cells too. That
  lowers the Euclidean DIC by 340 to 2,700 for the 102nd Senate and 600 to 1,150 for the 115th (`dic_imputed` in
  `results/euclid/summary.jsonl`). Fixed in the R file.

A few misspecified fits have chains in different modes (R-hat of the log-likelihood above 1.1 in
`criteria_table.csv`); none is at a selected dimension.
