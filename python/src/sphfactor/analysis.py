"""Post-processing: principal nested spheres, Procrustes alignment, simple convergence checks."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# --- principal nested spheres (Jung, Dryden & Marron 2012) ---------------------


def _fit_subsphere(x, great=False):
    """Least-squares subsphere {z : rho(z, v) = r} of the data x (n, m) on S^{m-1}."""
    m = x.shape[1]
    _, vecs = np.linalg.eigh(x.T @ x)

    def resid(p):
        v = p[:m] / np.linalg.norm(p[:m])
        r = np.pi / 2 if great else p[m]
        return np.arccos(np.clip(x @ v, -1, 1)) - r

    best = None
    for j in range(min(m, 3)):  # start from the directions of least variation
        for sgn in (1.0, -1.0):
            v0 = sgn * vecs[:, j]
            p0 = v0 if great else np.append(v0, np.mean(np.arccos(np.clip(x @ v0, -1, 1))))
            fit = least_squares(resid, p0, method="lm")
            if best is None or fit.cost < best.cost:
                best = fit
    v = best.x[:m] / np.linalg.norm(best.x[:m])
    r = np.pi / 2 if great else float(np.mod(best.x[m], 2 * np.pi))
    if r > np.pi:
        r = 2 * np.pi - r
    if r > np.pi / 2:  # same subsphere seen from the antipode
        v, r = -v, np.pi - r
    return v, r


def _rotate_to_north(v):
    """Orthogonal matrix taking v to the last coordinate axis."""
    m = v.shape[0]
    e = np.zeros(m)
    e[-1] = 1.0
    w = e - v
    n = np.linalg.norm(w)
    if n < 1e-12:
        return np.eye(m)
    w = w / n
    return np.eye(m) - 2.0 * np.outer(w, w)  # Householder reflection v -> e


def pns(x, great=False):
    """Principal nested spheres of points x (n, d+1) on S^d.

    Returns the residuals (n, d), column 0 being the leading component (the
    final circle) and the last column the first subsphere removed, each scaled
    to geodesic distances on the original sphere, and the percentage of
    variance carried by each component.
    """
    x = unit(np.asarray(x, float))
    res, scale = [], 1.0
    while x.shape[1] > 2:
        v, r = _fit_subsphere(x, great)
        rho = np.arccos(np.clip(x @ v, -1, 1))
        res.append(scale * (rho - r))
        proj = (np.sin(r) * x + np.sin(rho - r)[:, None] * v) / np.sin(rho)[:, None]
        y = proj @ _rotate_to_north(v).T
        x = unit(y[:, :-1] / np.sin(r))
        scale *= np.sin(r)
    theta = np.arctan2(x[:, 1], x[:, 0])
    # geodesic mean on the circle: minimise the sum of squared wrapped deviations over a grid, then refine
    wrap = lambda a: np.mod(a + np.pi, 2 * np.pi) - np.pi
    grid = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    m0 = grid[np.argmin([np.sum(wrap(theta - g) ** 2) for g in grid])]
    m0 = m0 + np.mean(wrap(theta - m0))
    res.append(scale * wrap(theta - m0))
    R = np.column_stack(res[::-1])
    var = np.sum(R**2, axis=0)
    return R, 100.0 * var / var.sum()


# --- alignment and diagnostics --------------------------------------------------


def procrustes_rotation(x, target):
    """Orthogonal Q (rotations and reflections) minimising |x Q - target|_F."""
    u, _, vt = np.linalg.svd(x.T @ target)
    return u @ vt


def rhat(chains):
    """Split-free Gelman-Rubin statistic for an (m, n) array of scalar draws."""
    chains = np.asarray(chains, float)
    m, n = chains.shape
    w = chains.var(axis=1, ddof=1).mean()
    b = n * chains.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))
