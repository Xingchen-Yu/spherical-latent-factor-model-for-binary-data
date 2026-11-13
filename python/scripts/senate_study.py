"""Spherical models, K = 1..10, for the 102nd and 115th U.S. Senates (paper Section 5.2).

    uv run python scripts/senate_study.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

import sphfactor  # noqa: F401
from sphfactor.data import load_ord
from sphfactor.model import Priors, SamplerConfig, sample_many

ap = argparse.ArgumentParser()
ap.add_argument("--K", type=int, nargs="+", default=[1, 2, 3, 4, 8, 5, 6, 7, 9, 10])
ap.add_argument("--congress", type=int, nargs="+", default=[102, 115])
ap.add_argument("--out", default="results/senate")
ap.add_argument("--chains", type=int, default=2)
ap.add_argument("--burnin", type=int, default=20_000)
ap.add_argument("--samples", type=int, default=20_000)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for K in a.K:
    for congress in a.congress:
        label = f"S{congress}"
        seeds = [s for s in range(a.chains) if not (out / f"{label}_K{K}_s{s}.npz").exists()]
        if not seeds:
            continue
        y, names, party = load_ord("S", congress)
        # Democrats (and independents, who caucus with them) start at negative first angles, Republicans at positive ones.
        group = np.where(party == 200, 1.0, -1.0)
        cfg = SamplerConfig(n_samples=a.samples, burnin=a.burnin)
        posts = sample_many([y] * len(seeds), K, None, cfg, seeds=[1000 * K + s for s in seeds], groups=[group] * len(seeds), labels=[label] * len(seeds))
        for s, post in zip(seeds, posts):
            post.save(out / f"{label}_K{K}_s{s}.npz")
            row = {"label": label, "K": K, "seed": s, **post.criteria, "accept": post.accept, "seconds": post.seconds,
                   "omega_mean": float(post.omega.mean()), "tau_mean": float(post.tau.mean()), "inv_lam_mean": float((1 / post.lam).mean())}
            with open(out / "summary.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "accept"}), flush=True)
