"""Geweke (2004) joint-distribution test of the Euclidean probit Gibbs sampler (priors of Euclidean_nd.R).

    JAX_PLATFORMS=cpu uv run python scripts/geweke_euclid.py

Alternates y | parameters with one Gibbs sweep; the parameter draws must then follow the prior:
beta_ik ~ N(0, 1/k^2), mu_j ~ N(0, 1), alpha_jk ~ N(0, 1).  Reports z-scores of first and second moments
(standard errors from batch means).
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr

import sphfactor  # noqa: F401
from sphfactor.euclid import _sweep

I, J, K, N = 6, 8, 3, 400_000
prec = jnp.arange(1, K + 1.0) ** 2


def step(c, k):
    beta, mu, alpha = c
    k1, k2 = jax.random.split(k)
    p = ndtr(mu[None, :] + beta @ alpha.T)
    y = jax.random.uniform(k1, (I, J)) < p
    sign = jnp.where(y, 1.0, -1.0).at[0, 0].set(0.0)  # one missing cell
    beta, mu, alpha, _ = _sweep(k2, beta, mu, alpha, sign, 1.0, prec)
    g = jnp.concatenate([beta[0], beta[0] ** 2, mu[:1], mu[:1] ** 2, alpha[0], alpha[0] ** 2, (beta[0, :1] * alpha[0, :1])])
    return (beta, mu, alpha), g


key = jax.random.key(7)
k0, k = jax.random.split(key)
kb, km, ka = jax.random.split(k0, 3)
init = (jax.random.normal(kb, (I, K)) / jnp.sqrt(prec), jax.random.normal(km, (J,)), jax.random.normal(ka, (J, K)))
_, g = jax.jit(lambda: jax.lax.scan(step, init, jax.random.split(k, N)))()
g = np.asarray(g)
v = 1 / np.arange(1, K + 1.0) ** 2
expect = np.concatenate([np.zeros(K), v, [0, 1], np.zeros(K), np.ones(K), [0]])
names = [f"E beta_{k+1}" for k in range(K)] + [f"E beta_{k+1}^2" for k in range(K)] + ["E mu", "E mu^2"] + [f"E alpha_{k+1}" for k in range(K)] + [f"E alpha_{k+1}^2" for k in range(K)] + ["E beta_1 alpha_1"]
B = 200
bm = g[: N // B * B].reshape(B, -1, g.shape[1]).mean(1)
se = bm.std(0, ddof=1) / np.sqrt(B)
z = (g.mean(0) - expect) / se
for n, m, e, zz in zip(names, g.mean(0), expect, z):
    print(f"{n:18s} mean {m:8.4f}  prior {e:7.4f}  z {zz:6.2f}")
print(f"max |z| = {np.abs(z).max():.2f} over {len(z)} moments")
