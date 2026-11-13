import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

from sphfactor import prior
from sphfactor.analysis import pns, procrustes_rotation
from sphfactor.euclid import sample_probit
from sphfactor.model import Priors, SamplerConfig, loglik_matrix, make_data, sample, sample_many, utility_gap
from sphfactor.special import log_link_cdf


def _sim(seed, K, I, J, w=2.0, kappa=40.0):
    ks = jax.random.split(jax.random.key(seed), 4)
    om = jnp.full(K, w)
    beta, yes, no = (prior.sample_svm(k, om, (n,)) for k, n in zip(ks[:3], (I, J, J)))
    p = jnp.exp(log_link_cdf(utility_gap(beta, yes, no), jnp.full((1, J), kappa)))
    return np.array((jax.random.uniform(ks[3], (I, J)) < p).astype(float)), np.asarray(p), np.asarray(beta)


def test_loglik_matches_high_precision_reference():
    mp.mp.dps = 30
    rng = np.random.default_rng(0)
    K, I, J = 3, 5, 6
    u = lambda n: (lambda v: v / np.linalg.norm(v, axis=1, keepdims=True))(rng.normal(size=(n, K + 1)))
    beta, yes, no, kappa = u(I), u(J), u(J), rng.gamma(2, 20, J)
    y = rng.integers(0, 2, (I, J)).astype(float)
    y[0, 0] = np.nan
    ll = np.asarray(loglik_matrix(*(jnp.asarray(a) for a in (beta, yes, no, kappa)), make_data(y)))
    assert ll[0, 0] == 0.0
    for i in range(I):
        for j in range(J):
            if np.isnan(y[i, j]):
                continue
            e = mp.acos(mp.mpf(float(beta[i] @ no[j]))) ** 2 - mp.acos(mp.mpf(float(beta[i] @ yes[j]))) ** 2
            s = (e + mp.pi**2) / (2 * mp.pi**2)
            k = mp.mpf(float(kappa[j]))
            G = mp.betainc(k, k, 0, s, regularized=True) if s <= 0.5 else 1 - mp.betainc(k, k, 0, 1 - s, regularized=True)
            Gc = mp.betainc(k, k, 0, 1 - s, regularized=True) if s > 0.5 else 1 - mp.betainc(k, k, 0, s, regularized=True)
            ref = float(mp.log(G if y[i, j] == 1 else Gc))
            assert np.isclose(ll[i, j], ref, rtol=1e-8, atol=1e-10)


def test_nested_spheres_have_the_same_likelihood():
    """A model on S^K with last angle zero equals the model on S^{K-1} (paper, end of Section 3.1)."""
    rng = np.random.default_rng(1)
    phi = lambda n: np.column_stack([rng.uniform(-np.pi, np.pi, n), rng.uniform(-1.2, 1.2, (n, 2))])
    b, y, n = phi(4), phi(5), phi(5)
    e3 = utility_gap(*(prior.to_cartesian(jnp.asarray(np.column_stack([a, np.zeros(len(a))]))) for a in (b, y, n)))
    e2 = utility_gap(*(prior.to_cartesian(jnp.asarray(a)) for a in (b, y, n)))
    assert np.allclose(e3, e2)


@pytest.mark.parametrize("K", [1, 3])
def test_sampler_recovers_probabilities(K):
    y, p, _ = _sim(3, K, 60, 300)
    y[::7, ::5] = np.nan
    post = sample(y, K, config=SamplerConfig(n_samples=1500, burnin=1500, print_every=0))
    obs = ~np.isnan(y)
    p_hat = np.where(y == 1, post.prob_mean, 1 - post.prob_mean)[obs]
    assert np.corrcoef(p_hat, p[obs])[0, 1] > 0.93
    oracle = np.mean((p[obs] > 0.5) == (y[obs] == 1))  # accuracy of the true probabilities
    assert post.criteria["accuracy"] > oracle - 0.03
    assert np.all(np.isnan(post.prob_mean[~obs]))
    assert 0.5 < post.accept["beta"] < 0.97 and 0.5 < post.accept["yes"] < 0.97


def test_batched_chains_match_shapes_and_differ_by_seed():
    y, _, _ = _sim(4, 2, 20, 40)
    posts = sample_many([y, y], 2, [Priors(), Priors(tau_rate=0.1)], SamplerConfig(n_samples=100, burnin=100, print_every=0), seeds=[1, 2])
    assert posts[0].beta.shape == (100, 20, 3) and posts[0].omega.shape == (100,)
    assert not np.allclose(posts[0].omega, posts[1].omega)
    assert posts[1].priors["tau_rate"] == 0.1


def test_pns_finds_a_great_circle():
    rng = np.random.default_rng(0)
    t = rng.uniform(-2, 2, 200)
    x = np.column_stack([np.cos(t), np.sin(t), 0.02 * rng.normal(size=200), 0.02 * rng.normal(size=200)])
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    _, pct = pns(x @ q)
    assert pct[0] > 99 and pct.shape == (3,)
    assert np.allclose(procrustes_rotation(x @ q, x), q.T, atol=1e-8)


def test_probit_sampler_runs():
    rng = np.random.default_rng(0)
    b, a, m = rng.normal(size=(50, 2)), rng.normal(size=(200, 2)), rng.normal(size=200)
    y = (rng.normal(size=(50, 200)) < m + b @ a.T).astype(float)
    r = sample_probit(y, 2, 600, 600)
    assert r["criteria"]["accuracy"] > 0.75
