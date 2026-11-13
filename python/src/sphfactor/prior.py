"""The spherical von Mises (SvM) prior and the hyperspherical coordinates of the paper.

Coordinates (paper eq. 5): a point of S^K is given by K angles
phi_1 in [-pi, pi), phi_k in [-pi/2, pi/2] (k >= 2), or by x in R^{K+1}, |x| = 1:

    x_1 = cos phi_1 * prod_{k>=2} cos phi_k
    x_2 = sin phi_1 * prod_{k>=2} cos phi_k
    x_{k+1} = sin phi_k * prod_{l>k} cos phi_l,   k = 2..K.

The original R code uses the other standard convention (a_1 = phi_1 + pi,
a_k = phi_k + pi/2); ``from_r_coords`` converts.

SvM(omega_1..omega_K): phi_1 ~ vonMises(0, omega_1) and 2 phi_k ~ vonMises(0, omega_k),
independently.  Its density with respect to the surface measure of the sphere is
(with S_k = x_1^2 + ... + x_k^2)

    log p(x) = omega_1 x_1 / sqrt(S_2) - log(2 pi I0(omega_1))
               + sum_{k=2}^K [ omega_k (1 - 2 x_{k+1}^2 / S_{k+1}) - log(pi I0(omega_k)) ]
               - 1/2 sum_{k=2}^K log S_k.

The last line is the Jacobian: the surface element is prod_k |cos phi_k|^{k-1} d phi
and cos^2 phi_k = S_k / S_{k+1}.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import i0e

LOG_PI = float(np.log(np.pi))
LOG_2PI = float(np.log(2 * np.pi))


def precisions(omega, K):
    """The paper's prior scale sequence (omega, 2^2 omega, ..., K^2 omega)."""
    return omega * jnp.arange(1, K + 1, dtype=jnp.result_type(omega, jnp.float32)) ** 2


def log_i0(w):
    return jnp.log(i0e(w)) + w


def svm_log_norm(omega_vec):
    """Log normalising constant (to be subtracted) of the SvM density."""
    return LOG_2PI + log_i0(omega_vec[0]) + jnp.sum(LOG_PI + log_i0(omega_vec[1:]))


def svm_log_kernel(x, omega_vec):
    """Unnormalised log SvM density w.r.t. surface measure; x (..., K+1), omega_vec (K,).

    Defined for any x != 0 (it only depends on ratios), which gives autodiff a
    smooth extension off the sphere; geodesic HMC projects the gradient onto
    the tangent space, where every extension has the same gradient.
    """
    K = x.shape[-1] - 1
    tiny = jnp.finfo(x.dtype).tiny
    S = jnp.maximum(jnp.cumsum(x * x, axis=-1), tiny)
    out = omega_vec[0] * x[..., 0] / jnp.sqrt(S[..., 1])
    if K > 1:
        out = out + jnp.sum(omega_vec[1:] * (1.0 - 2.0 * x[..., 2:] ** 2 / S[..., 2:]), axis=-1)
        out = out - 0.5 * jnp.sum(jnp.log(S[..., 1:K]), axis=-1)
    return out


def svm_logpdf(x, omega_vec):
    return svm_log_kernel(x, omega_vec) - svm_log_norm(omega_vec)


def to_cartesian(phi):
    """Angles (..., K) -> unit vectors (..., K+1)."""
    K = phi.shape[-1]
    cos, sin = jnp.cos(phi), jnp.sin(phi)
    # c = (cos phi_2, ..., cos phi_K, 1); tail[j] = prod_{m >= j} c[m] = prod_{l >= j+2} cos phi_l
    c = jnp.concatenate([cos[..., 1:], jnp.ones_like(cos[..., :1])], axis=-1)
    tail = jnp.flip(jnp.cumprod(jnp.flip(c, -1), axis=-1), -1)
    x1 = cos[..., :1] * tail[..., :1]
    x2 = sin[..., :1] * tail[..., :1]
    rest = sin[..., 1:] * tail[..., 1:K]  # x_{k+1} = sin phi_k * prod_{l > k} cos phi_l, k = 2..K
    return jnp.concatenate([x1, x2, rest], axis=-1)


def to_angles(x):
    """Unit vectors (..., K+1) -> angles (..., K)."""
    S = jnp.cumsum(x * x, axis=-1)
    phi1 = jnp.arctan2(x[..., 1], x[..., 0])
    rest = jnp.arctan2(x[..., 2:], jnp.sqrt(S[..., 1:-1]))
    return jnp.concatenate([phi1[..., None], rest], axis=-1)


def from_r_coords(x_r):
    """Embedded coordinates of the original R code -> this package's convention."""
    return jnp.concatenate([-x_r[..., 1:2], -x_r[..., 0:1], -x_r[..., 2:]], axis=-1)


def sample_svm(key, omega_vec, shape=()):
    """Exact draws from SvM(omega_vec) as unit vectors, shape + (K+1,)."""
    K = omega_vec.shape[-1]
    z = _von_mises(key, jnp.broadcast_to(omega_vec, shape + (K,)))
    scale = jnp.concatenate([jnp.ones(1), jnp.full(K - 1, 0.5)])
    return to_cartesian(z * scale)


def _von_mises(key, kappa):
    """vonMises(0, kappa) draws, elementwise (Best & Fisher 1979)."""
    from .ghmc import sample_von_mises

    flat = kappa.reshape(-1)
    keys = jax.random.split(key, flat.shape[0])
    return jax.vmap(lambda k, w: sample_von_mises(k, 0.0, w))(keys, flat).reshape(kappa.shape)
