import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from sphfactor import prior


def angular_reference(x, om):
    """log density built from the angular density divided by the surface element."""
    phi = prior.to_angles(x)
    K = phi.shape[-1]
    lp = om[0] * jnp.cos(phi[0]) + jnp.sum(om[1:] * jnp.cos(2 * phi[1:])) - prior.svm_log_norm(om)
    return lp - jnp.sum(jnp.arange(1, K) * jnp.log(jnp.cos(phi[1:])))


@pytest.mark.parametrize("K", [1, 2, 3, 5, 10])
def test_density_and_gradient_match_angular_reference(K):
    rng = np.random.default_rng(K)
    for scheme in range(2):
        om = jnp.asarray(0.7 * np.arange(1, K + 1) ** 2 if scheme == 0 else rng.gamma(2, 2, K))
        for _ in range(20):
            x = rng.normal(size=K + 1)
            x = jnp.asarray(x / np.linalg.norm(x))
            assert np.isclose(prior.svm_logpdf(x, om), angular_reference(x, om), rtol=1e-11, atol=1e-11)
            P = np.eye(K + 1) - np.outer(x, x)
            g, g_ref = jax.grad(prior.svm_logpdf)(x, om), jax.grad(angular_reference)(x, om)
            assert np.allclose(P @ g, P @ g_ref, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("K", [1, 2, 4, 10])
def test_coordinates_roundtrip_and_r_convention(K):
    rng = np.random.default_rng(K)
    phi = np.column_stack([rng.uniform(-np.pi, np.pi, 50)] + [rng.uniform(-np.pi / 2, np.pi / 2, 50) for _ in range(K - 1)])
    x = prior.to_cartesian(jnp.asarray(phi))
    assert np.allclose(np.linalg.norm(x, axis=1), 1)
    assert np.allclose(prior.to_angles(x), phi)
    # R code: getcord_auto with a_1 = phi_1 + pi, a_k = phi_k + pi/2
    a = phi + np.concatenate([[np.pi], np.full(K - 1, np.pi / 2)])
    sup = np.ones((50, 1))
    for i in range(K):
        sup = np.hstack([sup * np.sin(a[:, [i]]), np.cos(a[:, [i]])])
    assert np.allclose(prior.from_r_coords(jnp.asarray(sup)), x)


def test_exact_sampler_marginals():
    om = jnp.asarray([2.0, 8.0, 18.0])
    phi = np.asarray(prior.to_angles(prior.sample_svm(jax.random.key(0), om, (20000,))))
    assert stats.kstest(phi[:, 0], stats.vonmises(2.0).cdf).pvalue > 1e-3
    for k in (1, 2):
        assert stats.kstest(2 * phi[:, k], stats.vonmises(float(om[k])).cdf).pvalue > 1e-3
