"""Geweke (2004) joint-distribution test of the JAX sampler.

Alternates "simulate y given the parameters" with one full sweep.  If every
conditional update is correct, the parameters stay distributed as the prior,
which we can sample exactly.  Run on CPU:

    JAX_PLATFORMS=cpu uv run python scripts/geweke_check.py --K 3
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

import sphfactor  # noqa: F401
from sphfactor import prior
from sphfactor.model import Data, Jitter, Priors, SamplerConfig, State, _P, _sweep, loglik_matrix, utility_gap
from sphfactor.special import log_link_cdf

ap = argparse.ArgumentParser()
ap.add_argument("--K", type=int, default=3)
ap.add_argument("--I", type=int, default=4)
ap.add_argument("--J", type=int, default=5)
ap.add_argument("--sweeps", type=int, default=300_000)
ap.add_argument("--step", type=float, nargs=2, default=(0.05, 0.25))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--grad-dtype", default="float32")
a = ap.parse_args()
K, I, J = a.K, a.I, a.J
pri = Priors(omega_shape=6.0, omega_rate=2.0, tau_shape=6.0, tau_rate=2.0, kappa_shape=2.0, lam_shape=6.0, lam_rate=40.0)
P = _P(*(jnp.asarray(getattr(pri, f)) for f in _P._fields))
cfg = SamplerConfig(grad_dtype=a.grad_dtype)


def prior_draw(key, n):
    ks = jax.random.split(key, 7)
    omega = jax.random.gamma(ks[0], pri.omega_shape, (n,)) / pri.omega_rate
    tau = jax.random.gamma(ks[1], pri.tau_shape, (n,)) / pri.tau_rate
    lam = jax.random.gamma(ks[2], pri.lam_shape, (n,)) / pri.lam_rate
    kappa = jax.random.gamma(ks[3], pri.kappa_shape, (n,)) / lam
    k2 = jnp.arange(1, K + 1) ** 2
    x = lambda k, w: jax.vmap(lambda kk, ww: prior.sample_svm(kk, ww * k2))(jax.random.split(k, n), w)
    return omega, tau, lam, kappa, x(ks[4], omega), x(ks[5], tau), x(ks[6], tau)


def stats(omega, tau, lam, kappa, b, y, n):
    out = {"omega": omega, "log omega": jnp.log(omega), "tau": tau, "tau^2": tau**2, "lambda": lam, "log kappa_1": jnp.log(kappa), "sqrt kappa_1": jnp.sqrt(kappa)}
    for name, x in (("beta_1", b), ("psi_1", y), ("zeta_J", n)):
        phi = prior.to_angles(x)
        out[f"{name}: cos phi1"] = jnp.cos(phi[:, 0])
        out[f"{name}: sin phi1"] = jnp.sin(phi[:, 0])
        for k in range(1, K):
            out[f"{name}: cos 2phi{k + 1}"] = jnp.cos(2 * phi[:, k])
            out[f"{name}: sin phi{k + 1}"] = jnp.sin(phi[:, k])
    return {k: np.asarray(v) for k, v in out.items()}


@jax.jit
def run(key, state):
    def body(state, key):
        k_y, k_j, k_l, k_s = jax.random.split(key, 4)
        p = jnp.exp(log_link_cdf(utility_gap(state.beta, state.yes, state.no), state.kappa[None, :]))
        yv = jax.random.uniform(k_y, (I, J)) < p
        data = Data(jnp.where(yv, 1.0, -1.0), jnp.ones((I, J), bool))
        state = state._replace(ll=loglik_matrix(state.beta, state.yes, state.no, state.kappa, data))
        kj = jax.random.split(k_j, 3)
        u = lambda k, m: jax.random.uniform(k, (m,), minval=a.step[0], maxval=a.step[1])
        jit = Jitter(u(kj[0], I), u(kj[1], J), u(kj[2], J), jax.random.randint(k_l, (), 1, 11))
        state, st = _sweep(k_s, state, jit, data, P, cfg)
        rec = (state.omega, state.tau, state.lam, state.kappa[0], state.beta[0], state.yes[0], state.no[-1])
        return state, (rec, jnp.stack([s.astype(float).mean() for s in st[:6]]))

    return jax.lax.scan(body, state, jax.random.split(key, a.sweeps))


key = jax.random.key(a.seed)
k0, k1, k2 = jax.random.split(key, 3)
om, ta, la, ka, b, y, n = prior_draw(k0, max(I, J))
ones = lambda m: jnp.ones(m)
state = State(b[:I], y[:J], n[:J], ka[:J], la[0], om[0], ta[0], jnp.zeros((I, J)), jnp.full(J, 0.8), jnp.asarray(0.3), jnp.asarray(0.3), ones(I), ones(J), ones(J))
_, (rec, acc) = run(k1, state)
burn = a.sweeps // 20
chain = stats(*(r[burn:] for r in rec))
n_prior = 2_000_000
ref = stats(*prior_draw(k2, n_prior)[:3], *(lambda d: (d[3], d[4], d[5], d[6]))(prior_draw(k2, n_prior)))
z = {}
for k, v in chain.items():
    nb = 60
    bm = v[: len(v) // nb * nb].reshape(nb, -1).mean(1)
    z[k] = (v.mean() - ref[k].mean()) / np.sqrt(bm.var(ddof=1) / nb + ref[k].var() / n_prior)
zz = np.array(list(z.values()))
worst = max(z, key=lambda k: abs(z[k]))
print(f"K={K} I={I} J={J} sweeps={a.sweeps} grad={a.grad_dtype}: {len(zz)} statistics, mean z={zz.mean():+.2f}, sd z={zz.std():.2f}, max|z|={abs(zz).max():.2f} ({worst}), |z|>3: {int((abs(zz) > 3).sum())}")
print("   acceptance beta/psi/zeta/kappa/omega/tau:", np.round(np.asarray(acc).mean(0), 2))
for k in z:
    print(f"      {k:20s} sampler {chain[k].mean():+.4f}  prior {ref[k].mean():+.4f}  z={z[k]:+.2f}")
