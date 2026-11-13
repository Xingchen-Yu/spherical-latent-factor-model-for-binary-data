"""Simulation study of the paper (Section 5.1) with the spherical model.

Fits S^K models, K = 1..10, to the four simulated datasets stored in ../data
(scenarios 1-3: data from spherical models on S^2, S^3, S^5; scenario 4: data
from a Euclidean probit model in R^3), plus scenario 1 under the alternative
hyperpriors of the sensitivity analysis, plus one new S^2 dataset whose true
positions are kept (the stored datasets do not include their truth).

All chains for one K run in a single vectorised GPU program.  Each fit is saved
to <out>/<label>_K<K>_s<seed>.npz; finished fits are skipped on rerun.

    uv run python scripts/simulation_study.py --K 2 3 5 1 4 6 7 8 9 10
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import sphfactor  # noqa: F401
from sphfactor import prior
from sphfactor.data import load_rdata_matrix
from sphfactor.model import Priors, SamplerConfig, sample_many, utility_gap
from sphfactor.special import log_link_cdf

ALT = Priors(omega_rate=0.1, tau_rate=0.1, lam_rate=25.0)  # sensitivity analysis (Section 5.1.1)


def simulate_spherical(seed, K=2, I=100, J=700, precision=2.0, kappa_shape=2.0, kappa_scale=25.0):
    """New dataset from the spherical model, all component precisions equal (as in the paper), truth returned."""
    ks = jax.random.split(jax.random.key(seed), 5)
    w = jnp.full(K, precision)
    beta, yes, no = (prior.sample_svm(k, w, (n,)) for k, n in zip(ks[:3], (I, J, J)))
    kappa = jax.random.gamma(ks[3], kappa_shape, (J,)) * kappa_scale
    p = jnp.exp(log_link_cdf(utility_gap(beta, yes, no), kappa[None, :]))
    y = (jax.random.uniform(ks[4], (I, J)) < p).astype(float)
    return np.asarray(y), {k: np.asarray(v) for k, v in dict(beta=beta, yes=yes, no=no, kappa=kappa, prob=p).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--out", default="results/sim")
    ap.add_argument("--burnin", type=int, default=10_000)
    ap.add_argument("--samples", type=int, default=20_000)
    ap.add_argument("--chains", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    y_new, truth = simulate_spherical(2026)
    np.savez_compressed(out / "new_s2_truth.npz", y=y_new, **truth)
    jobs = [(name, load_rdata_matrix(name), Priors()) for name in ("sim_2", "sim_3", "sim_5", "ymat_eu_3")]
    jobs += [("sim_2_altprior", load_rdata_matrix("sim_2"), ALT), ("new_s2", y_new, Priors())]

    for K in a.K:
        todo = [(label, y, pri, seed) for label, y, pri in jobs for seed in range(a.chains) if not (out / f"{label}_K{K}_s{seed}.npz").exists()]
        if not todo:
            continue
        cfg = SamplerConfig(n_samples=a.samples, burnin=a.burnin)
        posts = sample_many([t[1] for t in todo], K, [t[2] for t in todo], cfg, seeds=[1000 * K + t[3] for t in todo], labels=[t[0] for t in todo])
        for (label, _, _, seed), post in zip(todo, posts):
            post.save(out / f"{label}_K{K}_s{seed}.npz")
            row = {"label": label, "K": K, "seed": seed, **post.criteria, "accept": post.accept, "seconds": post.seconds,
                   "omega_mean": float(post.omega.mean()), "tau_mean": float(post.tau.mean()), "inv_lam_mean": float((1 / post.lam).mean())}
            with open(out / "summary.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "accept"}), flush=True)


if __name__ == "__main__":
    main()
