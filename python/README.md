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

Settings: 2 chains per fit, 10,000 burn-in + 20,000 kept sweeps for the simulations, 20,000 + 20,000 for the Senates.
All numbers are in `figures/criteria_table.csv`; DIC is the paper's definition (larger is better).

Reproduced:

* **Dimension selection within the spherical family.** DIC picks the true dimension in every scenario
  (K = 2, 3, 5 for the spherical data, K = 3 for the Euclidean data), and the accuracy and principal-nested-spheres
  elbows sit at the same place (`figures/sim_scenarios.png`).
* **Euclidean model needs one more dimension on spherical data.** Its DIC peaks at K = 3, 4, 6 in Scenarios 1 to 3
  and at K = 3 in Scenario 4.
* **Prior sensitivity.** The alternative hyperprior gives the same DIC choice, accuracy and positions
  (`figures/sim_sensitivity*.png`).
* **Senates, spherical family.** K = 3 is best for the 102nd Senate; K = 2 is best for the 115th, with K = 1 next,
  as in the paper. The first nested-sphere component of the K = 8 fit explains 97% of the variance for the 115th and
  74% for the 102nd. The 1D-Euclidean vs circular rank plots show the same pattern as the paper (agreement in the
  102nd, disagreement among Republicans in the 115th).
* **Coverage** on a new S^2 dataset with known truth: 97% (dimension 1) and 77% (dimension 2); the paper reports
  98% and 68%.

Not reproduced, and worth checking against the original Euclidean code before submission:

* **Spherical vs Euclidean by DIC.** The paper says the best model overall is spherical in Scenarios 1 to 3 and for
  the 115th Senate. Here the best Euclidean model has the higher DIC in Scenarios 2 and 3 under both Euclidean priors
  tried (item-parameter variance 25 and 1), by about 200 to 550; in Scenario 1 the ranking flips with the prior
  (Euclidean ahead by 180 with variance 25, spherical ahead by 200 with variance 1). For the 115th Senate the
  Euclidean K = 2 model is ahead of the spherical K = 2 model by about 300. The Euclidean priors of the paper are not
  in the repository, so this comparison depends on an assumption (`euclid.py`; rerun with
  `scripts/euclidean_study.py --item-var`).
  WAIC at the DIC-selected dimensions does favour the spherical model in Scenarios 1 to 3 and the Euclidean model in
  Scenario 4.
* **102nd Senate, Euclidean dimension.** The Euclidean DIC is nearly flat from K = 4 to 7 with its maximum at K = 6;
  the paper reports K = 3.
* **Nested-sphere share for the 102nd Senate.** The first three components explain 87% here; the paper says under
  70%. The decomposition options of the original R package are unknown.

Caveats on DIC itself:

* The penalty 2 Var(loglik) is estimated from one chain's log-likelihood trace and is noisy: the two chains differ by
  a few tens in DIC at the selected dimensions but by several hundred in some larger fits (the `dic_sd` column).
  Differences of that size between models should not be read as evidence without more chains.
* For the 115th Senate the Euclidean model with K = 9, 10 reaches 99.6% accuracy and its DIC rises again: with a
  vague prior the probit model separates the data almost perfectly.
* A few misspecified fits have chains in different modes (R-hat of the log-likelihood above 1.1: Scenario 1 K = 1,
  Scenario 3 K = 1, 2, Scenario 4 K = 9, 10); none is at a selected dimension.
