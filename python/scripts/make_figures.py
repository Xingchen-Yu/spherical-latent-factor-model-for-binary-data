"""Figures of the paper, redrawn from the JAX runs under results/.  Missing runs are skipped.

    JAX_PLATFORMS=cpu uv run python scripts/make_figures.py            # everything available
    JAX_PLATFORMS=cpu uv run python scripts/make_figures.py prior      # one group: prior, sim, senate
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import special, stats

import sphfactor  # noqa: F401
from sphfactor.analysis import pns, procrustes_rotation, rhat, unit

R = Path("results")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#1f1f1e", "#6b6a65", "#e4e2dc"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 2, "lines.markersize": 4, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "axes.titlesize": 9,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def rows(folder):
    f = R / folder / "summary.jsonl"
    out = defaultdict(list)
    if f.exists():
        for line in f.read_text().splitlines():
            r = json.loads(line)
            out[(r["label"], r["K"])].append(r)
    return out


def load(folder, label, K, seed=0):
    f = R / folder / f"{label}_K{K}_s{seed}.npz"
    return np.load(f) if f.exists() else None


# ------------------------------------------------------------------ prior figures
def to_cart(phi):
    cos, sin = np.cos(phi), np.sin(phi)
    c = np.concatenate([cos[:, 1:], np.ones((len(phi), 1))], axis=1)
    tail = np.flip(np.cumprod(np.flip(c, 1), axis=1), 1)
    return np.concatenate([cos[:, :1] * tail[:, :1], sin[:, :1] * tail[:, :1], sin[:, 1:] * tail[:, 1:]], axis=1)


def svm(rng, n, w):  # w (K,) or (n, K)
    w = np.broadcast_to(w, (n, np.shape(w)[-1]))
    phi = rng.vonmises(0.0, w)
    phi[:, 1:] /= 2
    return to_cart(phi)


def vmf(rng, n, d, w):
    if d == 2:
        t = rng.vonmises(0, w, n)
        return np.column_stack([np.cos(t), np.sin(t)])
    mu = np.zeros(d)
    mu[0] = 1
    return stats.vonmises_fisher(mu, w, seed=rng).rvs(n)


def theta(b, ps, ze, kap):
    rho2 = lambda a, c: np.arccos(np.clip((a * c).sum(1), -1, 1)) ** 2
    return special.betainc(kap, kap, (rho2(ze, b) - rho2(ps, b) + np.pi**2) / (2 * np.pi**2))


def fig_prior():
    rng = np.random.default_rng(1)
    n, Ks, kappas = 100_000, [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30], [(1.0, BLUE, "o"), (10.0, ORANGE, "s"), (100.0, AQUA, "^")]
    for name, title in (("vmf", "von Mises–Fisher priors"), ("svm", "spherical von Mises priors, precisions $k^2\\omega$, $k^2\\tau$")):
        fig, axes = plt.subplots(3, 3, figsize=(8.2, 6.6), sharex=True, sharey=True)
        for i, tau in enumerate((0.5, 2.0, 10.0)):
            for j, om in enumerate((0.5, 2.0, 10.0)):
                ax = axes[i, j]
                for kap, col, mk in kappas:
                    v = []
                    for K in Ks:
                        k2 = np.arange(1, K + 1) ** 2
                        draw = (lambda w: vmf(rng, n, K + 1, w)) if name == "vmf" else (lambda w: svm(rng, n, w * k2))
                        v.append(theta(draw(om), draw(tau), draw(tau), kap).var())
                    ax.plot(Ks, v, color=col, marker=mk, label=f"$\\kappa_j={kap:g}$")
                ax.set_title(f"$\\omega={om:g}$, $\\tau={tau:g}$")
                ax.set_ylim(0, 0.25)
        for ax in axes[-1]:
            ax.set_xlabel("dimension $K$")
        for ax in axes[:, 0]:
            ax.set_ylabel("Var$(\\theta_{ij})$")
        fig.suptitle(f"Prior variance of $\\theta_{{ij}}$: {title}", y=0.995)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3)
        save(fig, f"prior_variance_{name}")

    # shapes of the induced prior on theta, K = 10
    K, n = 10, 10_000
    k2 = np.arange(1, K + 1) ** 2
    panels = [("$\\kappa_j=200$, $\\omega=\\tau=20$", lambda: (20.0, 20.0, 200.0)), ("$\\kappa_j=50$, $\\omega=\\tau=1$", lambda: (1.0, 1.0, 50.0)),
              ("paper hyperpriors", lambda: (rng.gamma(1, 10, (n, 1)), rng.gamma(1, 1 / 5, (n, 1)), rng.gamma(1, 1 / rng.gamma(2, 1 / 150, n)))),
              ("alternative hyperpriors", lambda: (rng.gamma(1, 10, (n, 1)), rng.gamma(1, 10, (n, 1)), rng.gamma(1, 1 / rng.gamma(2, 1 / 25, n))))]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.6), sharey=False)
    for ax, (title, f) in zip(axes, panels):
        om, tau, kap = f()
        th = theta(svm(rng, n, om * k2), svm(rng, n, tau * k2), svm(rng, n, tau * k2), kap)
        ax.hist(th, bins=40, range=(0, 1), color=BLUE, edgecolor="white", linewidth=0.4)
        ax.set_title(title)
        ax.set_xlabel("$\\theta_{ij}$")
        ax.grid(False)
    axes[0].set_ylabel("draws (of 10,000)")
    fig.suptitle("Induced prior on $\\theta_{ij}$, $K=10$", y=1.02)
    fig.tight_layout()
    save(fig, "prior_theta_shapes")


# ------------------------------------------------------------------ criteria panels
def crit_panel(ax, sph, euc, label, key, truth=None, elabel=None):
    for res, col, mk, ls, name in ((sph, BLUE, "o", "-", "spherical"), (euc, ORANGE, "s", "--", "Euclidean")):
        Ks = sorted(K for (lab, K) in res if lab == (label if name == "spherical" else (elabel or label)))
        if not Ks:
            continue
        lab = label if name == "spherical" else (elabel or label)
        vals = [[r[key] for r in res[(lab, K)]] for K in Ks]
        ax.plot(Ks, [np.mean(v) for v in vals], color=col, marker=mk, linestyle=ls, label=name)
        for K, v in zip(Ks, vals):  # individual chains
            ax.plot([K] * len(v), v, linestyle="none", marker=".", color=col, alpha=0.45, markersize=5)
    if truth:
        ax.axvline(truth, color=MUTED, linewidth=1, linestyle=":")
    ax.set_xticks(range(1, 11))


def pns_panel(ax, post, color=BLUE, label=None, marker="o"):
    if post is None:
        return
    _, pct = pns(unit(post["beta_mean"]), great=True)  # great subspheres, as in the original (shapes::pns, sphere.type = "great")
    ax.plot(range(1, len(pct) + 1), pct, color=color, marker=marker, label=label)
    ax.set_xticks(range(1, len(pct) + 1))
    ax.set_ylim(0, 100)


def best(res, label, key="dic"):
    Ks = sorted(K for (lab, K) in res if lab == label)
    return max(Ks, key=lambda K: np.mean([r[key] for r in res[(label, K)]])) if Ks else None


def hyper_panels(axes, posts, priors=(1 / 0.1, 1 / 5.0, 150.0)):
    """Histograms of omega, tau, 1/lambda (rows) for a list of (title, Posterior npz) columns, with prior densities."""
    om_scale, tau_scale, lam_rate = priors
    spec = [("omega", "$\\omega$", lambda x: stats.gamma(1, scale=om_scale).pdf(x)), ("tau", "$\\tau$", lambda x: stats.gamma(1, scale=tau_scale).pdf(x)),
            ("lam", "$1/\\lambda$", lambda x: stats.invgamma(2, scale=lam_rate).pdf(x))]
    for i, (key, name, pdf) in enumerate(spec):
        for j, (title, post) in enumerate(posts):
            ax = axes[i, j]
            ax.grid(False)
            if post is None:
                continue
            x = post[key] if key != "lam" else 1 / post["lam"]
            ax.hist(x, bins=40, density=True, color=BLUE, edgecolor="white", linewidth=0.3)
            g = np.linspace(*ax.get_xlim(), 200)
            ax.plot(g, pdf(np.maximum(g, 1e-9)), color=INK, linewidth=1.2, label="prior")
            ax.set_xlabel(name)
            if i == 0:
                ax.set_title(title)
    axes[0, 0].legend()


def fig_sim():
    sph, euc = rows("sim"), rows("euclid")
    if not sph:
        return
    scen = [("sim_2", "Scenario 1 ($S^2$)", 2), ("sim_3", "Scenario 2 ($S^3$)", 3), ("sim_5", "Scenario 3 ($S^5$)", 5), ("ymat_eu_3", "Scenario 4 ($\\mathbb{R}^3$)", 3)]
    fig, axes = plt.subplots(4, 3, figsize=(9.5, 10))
    for i, (label, title, truth) in enumerate(scen):
        crit_panel(axes[i, 0], sph, euc, label, "dic", truth)
        crit_panel(axes[i, 1], sph, euc, label, "accuracy", truth)
        pns_panel(axes[i, 2], load("sim", label, 10))
        axes[i, 0].set_ylabel(title)
    for ax, t in zip(axes[0], ("DIC (larger is better)", "In-sample accuracy", "Principal nested spheres, $K=10$ fit: % variance")):
        ax.set_title(t)
    for ax in axes[-1, :2]:
        ax.set_xlabel("dimension $K$ (dotted line: truth)")
    axes[-1, 2].set_xlabel("component")
    axes[0, 0].legend()
    fig.tight_layout()
    save(fig, "sim_scenarios")

    fig, axes = plt.subplots(3, 2, figsize=(6.5, 6.5))
    hyper_panels(axes, [("Scenario 2 ($S^3$ data), $K=3$", load("sim", "sim_3", 3)), ("Scenario 4 ($\\mathbb{R}^3$ data), $K=3$", load("sim", "ymat_eu_3", 3))])
    fig.tight_layout()
    save(fig, "sim_hyperparameters")

    # sensitivity (scenario 1, original vs alternative hyperpriors)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for label, col, mk, ls, name in (("sim_2", BLUE, "o", "-", "original prior"), ("sim_2_altprior", ORANGE, "s", "--", "alternative prior")):
        Ks = sorted(K for (lab, K) in sph if lab == label)
        for ax, key in zip(axes[:2], ("dic", "accuracy")):
            ax.plot(Ks, [np.mean([r[key] for r in sph[(label, K)]]) for K in Ks], color=col, marker=mk, linestyle=ls, label=name)
            ax.set_xticks(range(1, 11))
            ax.set_xlabel("dimension $K$")
        pns_panel(axes[2], load("sim", label, 10), col, name, mk)
    axes[0].set_title("DIC")
    axes[1].set_title("In-sample accuracy")
    axes[2].set_title("Principal nested spheres, $K=10$: % variance")
    axes[2].set_xlabel("component")
    axes[0].legend()
    fig.tight_layout()
    save(fig, "sim_sensitivity")

    fig, axes = plt.subplots(3, 2, figsize=(6.5, 6.5))
    a, b = load("sim", "sim_2", 2), load("sim", "sim_2_altprior", 2)
    hyper_panels(axes[:, :1], [("original prior", a)])
    hyper_panels(axes[:, 1:], [("alternative prior", b)], priors=(10.0, 10.0, 25.0))
    fig.suptitle("Scenario 1, $K=2$")
    fig.tight_layout()
    save(fig, "sim_sensitivity_hyperparameters")
    if a is not None and b is not None:
        from sphfactor.prior import to_angles

        xa = unit(a["beta_mean"])
        xb = unit(b["beta_mean"]) @ procrustes_rotation(unit(b["beta_mean"]), xa)
        pa, pb = np.asarray(to_angles(xa)), np.asarray(to_angles(xb))
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
        for k, ax in enumerate(axes):
            ax.plot(pa[:, k], pb[:, k], linestyle="none", marker="o", color=BLUE, markersize=3)
            lim = [min(pa[:, k].min(), pb[:, k].min()), max(pa[:, k].max(), pb[:, k].max())]
            ax.plot(lim, lim, color=MUTED, linewidth=1)
            ax.set_xlabel(f"original prior, $\\beta_{{i,{k + 1}}}$")
            ax.set_ylabel(f"alternative prior, $\\beta_{{i,{k + 1}}}$")
        fig.suptitle("Scenario 1: posterior mean positions under the two priors")
        fig.tight_layout()
        save(fig, "sim_sensitivity_positions")

    # recovery of the true positions: Scenario 1 (truth in data/o1_2_sphdata.Rdata) and a new S^2 dataset
    from sphfactor.prior import from_r_coords, to_angles

    def truth_of(name):
        if name == "sim_2":
            import rdata

            f = Path(__file__).resolve().parents[2] / "data" / "o1_2_sphdata.Rdata"
            a = np.asarray(rdata.conversion.convert(rdata.parser.parse_file(f))["o1"][1])[:, :2]  # R angles
            xr = np.stack([np.sin(a[:, 0]) * np.sin(a[:, 1]), np.cos(a[:, 0]) * np.sin(a[:, 1]), np.cos(a[:, 1])], axis=1)
            return np.asarray(from_r_coords(xr))
        f = R / "sim" / f"{name}_truth.npz"
        return np.load(f)["beta"] if f.exists() else None

    wrap = lambda a: np.mod(a + np.pi, 2 * np.pi) - np.pi
    report = []
    for name, title in (("sim_2", "Scenario 1"), ("new_s2", "new $S^2$ dataset")):
        post, truth = load("sim", name, 2), truth_of(name)
        if post is None or truth is None:
            continue
        t = np.asarray(to_angles(truth))
        x = unit(post["beta"].astype(float))
        # two alignments: reflections of the coordinate axes only (as in the original analysis, where the prior
        # fixes the axes), and a full Procrustes rotation of the posterior mean onto the truth
        signs = [np.diag(sg) for sg in np.array(np.meshgrid(*[[1, -1]] * 3)).T.reshape(-1, 3)]
        Qr = min(signs, key=lambda Q: np.sum((unit(post["beta_mean"]) @ Q - truth) ** 2))
        Qp = procrustes_rotation(unit(post["beta_mean"]), truth)
        for Q, how in ((Qr, "reflection"), (Qp, "procrustes")):
            draws = np.asarray(to_angles(x @ Q))  # (B, I, 2)
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
            cover = []
            for k, ax in enumerate(axes):
                dev = wrap(draws[:, :, k] - t[None, :, k]) if k == 0 else draws[:, :, k] - t[None, :, k]
                lo, med, hi = np.quantile(dev, [0.025, 0.5, 0.975], axis=0)
                order = np.argsort(t[:, k])
                xs = np.arange(len(order))
                ax.vlines(xs, (t[:, k] + lo)[order], (t[:, k] + hi)[order], color=BLUE, linewidth=1)
                ax.plot(xs, (t[:, k] + med)[order], linestyle="none", marker="o", color=BLUE, markersize=2.5, label="posterior median, 95% interval")
                ax.plot(xs, t[order, k], linestyle="none", marker="^", color=ORANGE, markersize=3.5, label="truth")
                c = np.mean((lo <= 0) & (hi >= 0))
                cover.append(c)
                ax.set_title(f"{title}, dimension {k + 1}: coverage {100 * c:.0f}%")
                ax.set_xlabel("subject (sorted by true value)")
                ax.set_ylabel(f"$\\beta_{{i,{k + 1}}}$")
            axes[0].legend()
            fig.tight_layout()
            save(fig, f"sim_recovery_{name}_{how}")
            report.append(f"{name}, {how} alignment: coverage of 95% intervals dim 1 {cover[0]:.3f}, dim 2 {cover[1]:.3f}")
    (OUT / "sim_recovery.txt").write_text("\n".join(report) + "\n")

    # table of criteria + convergence
    lines = ["label,model,K,chains,dic_mean,dic_sd,lppd_mean,var_loglik_mean,elpd_waic_mean,accuracy_mean,rhat_loglik"]
    for folder, res, model in (("sim", sph, "spherical"), ("euclid", euc, "euclidean"), ("senate", rows("senate"), "spherical")):
        for (label, K), rr in sorted(res.items()):
            traces = [load(folder, label, K, r["seed"]) for r in rr]
            rh = rhat(np.stack([t["loglik_trace"][-20000:] for t in traces])) if len(traces) > 1 and all(t is not None for t in traces) else float("nan")
            m = lambda k: np.mean([r[k] for r in rr])
            lines.append(f"{label},{model},{K},{len(rr)},{m('dic'):.1f},{np.std([r['dic'] for r in rr]):.1f},{m('lppd'):.1f},{m('var_loglik'):.1f},{m('elpd_waic'):.1f},{m('accuracy'):.4f},{rh:.3f}")
    (OUT / "criteria_table.csv").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "criteria_table.csv")


def fig_senate():
    sph, euc = rows("senate"), rows("euclid")
    if not sph:
        return
    from sphfactor.data import load_ord
    from sphfactor.prior import to_angles

    fig, axes = plt.subplots(2, 2, figsize=(8, 6.4))
    for i, c in enumerate((102, 115)):
        label = f"S{c}"
        crit_panel(axes[i, 0], sph, euc, label, "dic")
        K_big = max([K for (lab, K) in sph if lab == label and K <= 8] or [None], key=lambda v: v or 0)
        pns_panel(axes[i, 1], load("senate", label, K_big) if K_big else None)
        axes[i, 0].set_ylabel(f"{c}th Senate" if c == 115 else f"{c}nd Senate")
        axes[i, 1].set_title(f"Principal nested spheres, $K={K_big}$ fit: % variance")
        axes[i, 0].set_title("DIC (larger is better)")
    axes[1, 0].set_xlabel("dimension $K$")
    axes[1, 1].set_xlabel("component")
    axes[0, 0].legend()
    fig.tight_layout()
    save(fig, "senate_dic_pns")

    fig, axes = plt.subplots(3, 2, figsize=(6.5, 6.5))
    hyper_panels(axes, [("102nd Senate, $K=3$", load("senate", "S102", 3)), ("115th Senate, $K=2$", load("senate", "S115", 2))])
    fig.tight_layout()
    save(fig, "senate_hyperparameters")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    out = []
    for ax, c in zip(axes, (102, 115)):
        s, e = load("senate", f"S{c}", 1), load("euclid", f"S{c}", 1)
        if s is None or e is None:
            continue
        _, names, party = load_ord("S", c)
        rep = party == 200
        phi = np.asarray(to_angles(unit(s["beta"].astype(float))))[:, :, 0]  # (B, I) angles in [-pi, pi)
        eb = e["beta"][:, :, 0].astype(float)
        eb = eb * np.sign(eb[:, rep].mean(1) - eb[:, ~rep].mean(1))[:, None]  # orient each draw: Republicans on the right
        rank = lambda d: np.median(np.argsort(np.argsort(d, axis=1), axis=1) + 1, axis=0)
        rs, re = rank(phi), rank(eb)
        for grp, col, mk, name in ((~rep, BLUE, "o", "Democrats and independents"), (rep, ORANGE, "^", "Republicans")):
            ax.plot(re[grp], rs[grp], linestyle="none", marker=mk, color=col, markersize=4, label=name)
        ax.plot([1, len(re)], [1, len(re)], color=MUTED, linewidth=1)
        ax.set_title(f"{c}{'nd' if c == 102 else 'th'} Senate")
        ax.set_xlabel("median rank, 1D Euclidean")
        ax.set_ylabel("median rank, circular ($K=1$)")
        top = np.argsort(-np.abs(rs - re))[:5]
        out.append(f"S{c}: largest rank differences: " + ", ".join(f"{names[i]} ({re[i]:.0f} -> {rs[i]:.0f})" for i in top))
    axes[0].legend()
    fig.tight_layout()
    save(fig, "senate_ranks")
    (OUT / "senate_ranks.txt").write_text("\n".join(out) + "\n")


def fig_criteria():
    """DIC, WAIC and their two ingredients side by side for every dataset."""
    sim, sen, euc = rows("sim"), rows("senate"), rows("euclid")
    data = [("sim_2", "Scenario 1 ($S^2$)", 2, sim), ("sim_3", "Scenario 2 ($S^3$)", 3, sim), ("sim_5", "Scenario 3 ($S^5$)", 5, sim),
            ("ymat_eu_3", "Scenario 4 ($\\mathbb{R}^3$)", 3, sim), ("S102", "102nd Senate", None, sen), ("S115", "115th Senate", None, sen)]
    data = [d for d in data if any(lab == d[0] for lab, _ in d[3])]
    if not data:
        return
    cols = [("dic", "DIC = lppd $-$ 2 Var(loglik)"), ("elpd_waic", "WAIC = lppd $-$ $p_{WAIC}$"), ("var2", "DIC penalty"), ("p_waic", "WAIC penalty")]
    for res in (sim, sen, euc):
        for rr in res.values():
            for r in rr:
                r["var2"] = 2 * r["var_loglik"]
    fig, axes = plt.subplots(len(data), 4, figsize=(12.5, 2.5 * len(data)), squeeze=False)
    for i, (label, title, truth, sph) in enumerate(data):
        for j, (key, t) in enumerate(cols):
            crit_panel(axes[i, j], sph, euc, label, key, truth)
            if i == 0:
                axes[i, j].set_title(t)
        axes[i, 0].set_ylabel(title)
    axes[0, 0].legend()
    for ax in axes[-1]:
        ax.set_xlabel("dimension $K$")
    fig.tight_layout()
    save(fig, "criteria_dic_waic")


if __name__ == "__main__":
    which = sys.argv[1:] or ["prior", "sim", "senate", "criteria"]
    for name in which:
        {"prior": fig_prior, "sim": fig_sim, "senate": fig_senate, "criteria": fig_criteria}[name]()
