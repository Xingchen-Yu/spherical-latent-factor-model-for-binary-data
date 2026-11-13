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

Two things had to be assumed because they are not in the repository: the priors of the Euclidean probit model
(`euclid.py`) and the options of the principal-nested-spheres decomposition (`analysis.pns`).

## Layout

    src/sphfactor/special.py   log CDF of the link (shared with the circular-model package)
    src/sphfactor/ghmc.py      geodesic HMC on the sphere, batched
    src/sphfactor/prior.py     spherical von Mises prior, coordinates
    src/sphfactor/model.py     the sampler
    src/sphfactor/euclid.py    Euclidean probit factor model (Albert-Chib Gibbs)
    src/sphfactor/analysis.py  principal nested spheres, Procrustes, R-hat
    src/sphfactor/data.py      .Rdata simulations and Voteview .ord files
