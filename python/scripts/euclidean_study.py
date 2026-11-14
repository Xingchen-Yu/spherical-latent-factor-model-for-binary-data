"""Euclidean probit factor models, K = 1..10, on the simulated datasets and the two Senates (runs on CPU).

    JAX_PLATFORMS=cpu uv run python scripts/euclidean_study.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

import sphfactor  # noqa: F401
from sphfactor.data import load_ord, load_rdata_matrix
from sphfactor.euclid import sample_probit

ap = argparse.ArgumentParser()
ap.add_argument("--K", type=int, nargs="+", default=list(range(1, 11)))
ap.add_argument("--data", nargs="+", default=["sim_2", "sim_3", "sim_5", "ymat_eu_3", "new_s2", "S102", "S115"])
ap.add_argument("--out", default="results/euclid")
ap.add_argument("--chains", type=int, default=2)
ap.add_argument("--burnin", type=int, default=10_000)
ap.add_argument("--samples", type=int, default=20_000)
ap.add_argument("--item-var", type=float, default=1.0, help="prior variance of intercepts and discriminations")
ap.add_argument("--no-shrink", action="store_true", help="beta_i ~ N(0, I) instead of the original N(0, diag(1/k^2))")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for name in a.data:
    if name == "new_s2":
        y = np.load("results/sim/new_s2_truth.npz")["y"]
    elif name[0] == "S" and name[1:].isdigit():
        y = load_ord("S", int(name[1:]))[0]
    else:
        y = load_rdata_matrix(name)
    for K in a.K:
        for seed in range(a.chains):
            f = out / f"{name}_K{K}_s{seed}.npz"
            if f.exists():
                continue
            r = sample_probit(y, K, a.samples, a.burnin, item_var=a.item_var, shrink=not a.no_shrink, seed=1000 * K + seed)
            np.savez_compressed(f, loglik_trace=r["loglik_trace"], beta=r["beta"])
            row = {"label": name, "K": K, "seed": seed, **r["criteria"], "seconds": r["seconds"]}
            with open(out / "summary.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
