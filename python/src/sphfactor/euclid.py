"""Euclidean probit factor model, the comparison model of the paper (Section 2).

    P(y_ij = 1) = Phi(mu_j + alpha_j' beta_i),   beta_i in R^K
    beta_i ~ N(0, I_K),   (mu_j, alpha_j) ~ N(0, item_var * I)

fitted with the Albert & Chib (1993) data-augmentation Gibbs sampler.  The
original Euclidean code is not in the repository, so the priors here are an
assumption (the usual ones for ideal-point models; item_var = 25).

Missing responses get an untruncated latent z_ij, which leaves the posterior
of the parameters unchanged and keeps the conditional covariances shared
across subjects and across items (one small matrix inverse per sweep).
"""

from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import log_ndtr, ndtr, ndtri

jax.config.update("jax_enable_x64", True)


def _sweep(key, beta, mu, alpha, sign, item_var):
    I, K = beta.shape
    J = mu.shape[0]
    kz, kb, ka = jax.random.split(key, 3)
    m = mu[None, :] + beta @ alpha.T
    # z_ij: s z > 0 for observed cells (s = +-1), free for missing cells (s = 0)
    u = jax.random.uniform(kz, (I, J), minval=1e-300, maxval=1.0)
    s = jnp.where(sign == 0, 1.0, sign)
    a = -s * m  # e = s z - s m must exceed a
    e_trunc = -ndtri(u * ndtr(-a))
    e_free = ndtri(u)
    z = m + jnp.where(sign == 0, e_free, s * e_trunc)
    # beta_i | z: precision alpha'alpha + I, shared by all subjects
    Vb = jnp.linalg.inv(alpha.T @ alpha + jnp.eye(K))
    Lb = jnp.linalg.cholesky(Vb)
    beta = (z - mu[None, :]) @ alpha @ Vb + jax.random.normal(kb, (I, K)) @ Lb.T
    # (mu_j, alpha_j) | z: precision X'X + I / item_var, shared by all items
    X = jnp.concatenate([jnp.ones((I, 1)), beta], axis=1)
    Va = jnp.linalg.inv(X.T @ X + jnp.eye(K + 1) / item_var)
    La = jnp.linalg.cholesky(Va)
    coef = z.T @ X @ Va + jax.random.normal(ka, (J, K + 1)) @ La.T
    mu, alpha = coef[:, 0], coef[:, 1:]
    ll = jnp.where(sign == 0, 0.0, log_ndtr(sign * (mu[None, :] + beta @ alpha.T)))
    return beta, mu, alpha, ll


def sample_probit(y, K: int, n_samples=20_000, burnin=10_000, item_var=25.0, seed=0, n_beta_draws=1000) -> dict:
    """Fit the K-dimensional probit factor model; returns criteria comparable with ``model.sample``."""
    y = np.asarray(y, float)
    mask = ~np.isnan(y)
    sign = jnp.asarray(np.where(mask, 2 * np.nan_to_num(y) - 1, 0.0))
    I, J = y.shape
    rng = np.random.default_rng(seed)
    beta, mu, alpha = (jnp.asarray(v) for v in (rng.normal(size=(I, K)), np.zeros(J), 0.1 * rng.normal(size=(J, K))))

    @partial(jax.jit, static_argnames="n")
    def run(key, beta, mu, alpha, n):
        def body(c, k):
            beta, mu, alpha, lse, s1, s2 = c
            beta, mu, alpha, ll = _sweep(k, beta, mu, alpha, sign, item_var)
            return (beta, mu, alpha, jnp.logaddexp(lse, ll), s1 + ll, s2 + ll * ll), (ll.sum(), beta.astype(jnp.float32))

        init = (beta, mu, alpha, jnp.full((I, J), -jnp.inf), jnp.zeros((I, J)), jnp.zeros((I, J)))
        return jax.lax.scan(body, init, jax.random.split(key, n))

    t0 = time.perf_counter()
    k1, k2 = jax.random.split(jax.random.key(seed))
    (beta, mu, alpha, *_), (trace0, _) = run(k1, beta, mu, alpha, burnin)
    (beta, mu, alpha, lse, s1, s2), (trace, draws) = run(k2, beta, mu, alpha, n_samples)
    lse, s1, s2, trace = (np.asarray(v) for v in (lse, s1, s2, trace))
    S = n_samples
    log_mean_p = np.where(mask, lse - np.log(S), np.nan)
    lppd = float(np.nansum(log_mean_p))
    var_ll = float(np.var(trace, ddof=1))
    p_waic = float(np.sum((s2[mask] - s1[mask] ** 2 / S) / (S - 1)))
    keep = np.linspace(0, S - 1, min(n_beta_draws, S)).astype(int)
    return {
        "criteria": {"lppd": lppd, "var_loglik": var_ll, "dic": lppd - 2 * var_ll, "p_waic": p_waic, "elpd_waic": lppd - p_waic,
                     "accuracy": float(np.mean(np.exp(log_mean_p[mask]) > 0.5)), "n_obs": int(mask.sum())},
        "loglik_trace": np.concatenate([np.asarray(trace0), trace]),
        "beta": np.asarray(draws)[keep],
        "seconds": time.perf_counter() - t0,
    }
