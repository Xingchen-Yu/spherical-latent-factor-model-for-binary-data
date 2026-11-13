"""The spherical latent factor model for binary data and its MCMC sampler.

Model (Yu & Rodriguez, arXiv:2008.05109), subjects i = 1..I, items j = 1..J,
latent positions on the sphere S^K embedded in R^{K+1}:

    y_ij | .      ~ Bernoulli(G_kappa_j(e_ij)),
    e_ij          = rho(zeta_j, beta_i)^2 - rho(psi_j, beta_i)^2,   rho(x, z) = arccos(x'z)
    beta_i        ~ SvM(omega, 2^2 omega, ..., K^2 omega),   omega ~ Gamma(omega_shape, omega_rate)
    psi_j, zeta_j ~ SvM(tau, 2^2 tau, ..., K^2 tau),         tau   ~ Gamma(tau_shape, tau_rate)
    kappa_j       ~ Gamma(kappa_shape, rate=lam),            lam   ~ Gamma(lam_shape, lam_rate)

G_kappa is the CDF of a Beta(kappa, kappa) variable scaled to [-pi^2, pi^2]
(``special.py``); SvM is the spherical von Mises prior (``prior.py``).  K = 1 is
the circular model with von Mises priors on all three sets of positions.

One sweep, in the order of the original R/C++ code:

    beta_i   geodesic HMC, all subjects in one vectorised call
    psi_j    geodesic HMC, all items at once
    zeta_j   geodesic HMC, all items at once
    kappa_j  random-walk MH on the log scale
    omega    random-walk MH on the log scale
    tau      random-walk MH on the log scale
    lam      conjugate Gamma draw

Differences from the original implementation, none of which changes the target:

  * gradients come from autodiff of the unconstrained log density (no
    division by x_{K+1}); they are evaluated in ``grad_dtype`` (float32 by
    default) while the accept/reject step uses float64, so the chain is exact;
  * missing responses drop out of the likelihood instead of being imputed;
  * step sizes are rescaled per coordinate during burn-in towards 60-90%
    acceptance (``adapt_steps``), since the right scale changes with K.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .ghmc import ghmc_step
from .prior import precisions, svm_log_kernel, svm_log_norm, to_cartesian
from .special import CF_ITERS, KAPPA_MAX, make_log_link_cdf

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class Priors:
    """Hyperparameters; defaults are the paper's (Figure 5)."""

    omega_shape: float = 1.0
    omega_rate: float = 0.1
    tau_shape: float = 1.0
    tau_rate: float = 5.0
    kappa_shape: float = 1.0  # c
    lam_shape: float = 2.0
    lam_rate: float = 150.0


@dataclass(frozen=True)
class SamplerConfig:
    n_samples: int = 20_000
    burnin: int = 10_000
    n_beta_draws: int = 2_000  # evenly spaced draws of beta kept (it is S x I x (K+1))
    jitter_every: int = 50  # redraw step sizes / leapfrog counts this often
    beta_step: tuple[float, float] = (0.01, 0.04)
    item_step: tuple[float, float] = (0.01, 0.07)  # psi and zeta
    # Leapfrog steps ~ U{1..max_leap}, redrawn every jitter_every sweeps and shared by all coordinates
    # (the R code drew one per coordinate; sharing it lets the GPU run only as many steps as needed).
    max_leap: int = 10
    kappa_prop_sd: float = 1.0
    omega_prop_sd: float = 0.1
    tau_prop_sd: float = 0.1
    adapt_steps: bool = True
    grad_dtype: str = "float32"
    grad_cf_iters: int = 16  # continued-fraction iterations in the leapfrog gradients (accept step uses 40)
    seed: int = 0
    print_every: int = 2000  # iterations; 0 to silence


class Data(NamedTuple):
    sign: jax.Array  # (I, J) +1 for y = 1, -1 for y = 0, 0 if missing
    mask: jax.Array  # (I, J) bool, True where observed


class State(NamedTuple):
    beta: jax.Array  # (I, K+1)
    yes: jax.Array  # (J, K+1) psi
    no: jax.Array  # (J, K+1) zeta
    kappa: jax.Array  # (J,)
    lam: jax.Array  # ()
    omega: jax.Array  # ()
    tau: jax.Array  # ()
    ll: jax.Array  # (I, J) log-likelihood at the current state (0 if missing)
    kappa_sd: jax.Array  # (J,)
    omega_sd: jax.Array
    tau_sd: jax.Array
    beta_scale: jax.Array  # (I,) step-size multipliers
    yes_scale: jax.Array  # (J,)
    no_scale: jax.Array  # (J,)


class Jitter(NamedTuple):
    beta_eps: jax.Array
    yes_eps: jax.Array
    no_eps: jax.Array
    leap: jax.Array  # () leapfrog steps, shared by all coordinates and chains of a block


class SweepStats(NamedTuple):
    acc_beta: jax.Array
    acc_yes: jax.Array
    acc_no: jax.Array
    acc_kappa: jax.Array
    acc_omega: jax.Array
    acc_tau: jax.Array
    loglik: jax.Array


def make_data(y) -> Data:
    """(I, J) array of 1 / 0 / NaN -> Data."""
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y)
    return Data(jnp.asarray(np.where(mask, 2 * np.nan_to_num(y) - 1, 0.0)), jnp.asarray(mask))


def _rho2(c):
    """Squared geodesic distance from the inner product."""
    eps = 1e-15 if c.dtype == jnp.float64 else 1e-6
    a = jnp.arccos(jnp.clip(c, -1.0 + eps, 1.0 - eps))
    return a * a


def utility_gap(beta, yes, no):
    """e_ij = rho(zeta_j, beta_i)^2 - rho(psi_j, beta_i)^2, shape (I, J)."""
    return _rho2(beta @ no.T) - _rho2(beta @ yes.T)


# Same values as special.log_link_cdf; the continued fraction stops as soon as every element has converged.
_cdf64 = make_log_link_cdf(-CF_ITERS)


def loglik_matrix(beta, yes, no, kappa, data: Data, cdf=_cdf64):
    """(I, J) matrix of log P(y_ij | params); zero where y_ij is missing."""
    dt = beta.dtype
    u = data.sign.astype(dt) * utility_gap(beta, yes, no)
    return jnp.where(data.mask, cdf(u, kappa[None, :].astype(dt)), 0.0)


def log_post_scale(w, x, shape, rate):
    """log p(w | x_1..x_n) up to a constant, for x_n ~ SvM(w, 4w, ..., K^2 w), w ~ Gamma."""
    K = x.shape[-1] - 1
    vec = precisions(w, K)
    return jnp.sum(svm_log_kernel(x, vec)) - x.shape[0] * svm_log_norm(vec) + (shape - 1.0) * jnp.log(w) - rate * w


def _sweep(key, state: State, jitter: Jitter, data: Data, priors: Priors, cfg: SamplerConfig) -> tuple[State, SweepStats]:
    gdt = jnp.dtype(cfg.grad_dtype)
    data_g = Data(data.sign.astype(gdt), data.mask)
    cdf_g = make_log_link_cdf(cfg.grad_cf_iters)
    kb, ky, kn, kk, kku, ko, kou, kt, ktu, kl = jax.random.split(key, 10)
    beta, yes, no, kappa, lam, omega, tau, ll = state.beta, state.yes, state.no, state.kappa, state.lam, state.omega, state.tau, state.ll
    K = beta.shape[-1] - 1
    om_vec, tau_vec = precisions(omega, K), precisions(tau, K)

    def grad_of(lp, *args):
        cast = [a.astype(gdt) for a in args]
        return lambda x: jax.grad(lambda x: lp(x, *cast).sum())(x.astype(gdt))

    # --- beta_i ---------------------------------------------------------------
    def lp_beta(x, yes, no, kappa, vec, d=data_g):
        return loglik_matrix(x, yes, no, kappa, d, cdf_g).sum(1) + svm_log_kernel(x, vec)

    def lp_beta_aux(x):
        m = loglik_matrix(x, yes, no, kappa, data)
        return m.sum(1) + svm_log_kernel(x, om_vec), m

    beta, info_b = ghmc_step(
        kb, beta, lp_beta_aux, jitter.beta_eps * state.beta_scale, jitter.leap, jitter.leap,
        grad_fn=grad_of(lp_beta, yes, no, kappa, om_vec),
        log_prob_x=(ll.sum(1) + svm_log_kernel(beta, om_vec), ll), has_aux=True,
    )
    ll = info_b.aux

    # --- psi_j ----------------------------------------------------------------
    def lp_yes(x, beta, no, kappa, vec, d=data_g):
        return loglik_matrix(beta, x, no, kappa, d, cdf_g).sum(0) + svm_log_kernel(x, vec)

    def lp_yes_aux(x):
        m = loglik_matrix(beta, x, no, kappa, data)
        return m.sum(0) + svm_log_kernel(x, tau_vec), m.T

    yes, info_y = ghmc_step(
        ky, yes, lp_yes_aux, jitter.yes_eps * state.yes_scale, jitter.leap, jitter.leap,
        grad_fn=grad_of(lp_yes, beta, no, kappa, tau_vec),
        log_prob_x=(ll.sum(0) + svm_log_kernel(yes, tau_vec), ll.T), has_aux=True,
    )
    ll = info_y.aux.T

    # --- zeta_j ---------------------------------------------------------------
    def lp_no(x, beta, yes, kappa, vec, d=data_g):
        return loglik_matrix(beta, yes, x, kappa, d, cdf_g).sum(0) + svm_log_kernel(x, vec)

    def lp_no_aux(x):
        m = loglik_matrix(beta, yes, x, kappa, data)
        return m.sum(0) + svm_log_kernel(x, tau_vec), m.T

    no, info_n = ghmc_step(
        kn, no, lp_no_aux, jitter.no_eps * state.no_scale, jitter.leap, jitter.leap,
        grad_fn=grad_of(lp_no, beta, yes, kappa, tau_vec),
        log_prob_x=(ll.sum(0) + svm_log_kernel(no, tau_vec), ll.T), has_aux=True,
    )
    ll = info_n.aux.T

    # --- kappa_j: log-normal random walk ----------------------------------------
    log_k = jnp.log(kappa)
    log_kp = log_k + state.kappa_sd * jax.random.normal(kk, kappa.shape)
    kappa_p = jnp.exp(log_kp)
    ll_p = loglik_matrix(beta, yes, no, kappa_p, data)
    # (shape - 1) log kappa from the prior plus log kappa from the Jacobian
    log_r = ll_p.sum(0) - ll.sum(0) + priors.kappa_shape * (log_kp - log_k) - lam * (kappa_p - kappa)
    # The link code is validated for kappa <= KAPPA_MAX (1e4); larger proposals are rejected.
    acc_kappa = (jnp.log(jax.random.uniform(kku, kappa.shape)) < log_r) & (kappa_p <= KAPPA_MAX)
    kappa = jnp.where(acc_kappa, kappa_p, kappa)
    ll = jnp.where(acc_kappa[None, :], ll_p, ll)

    # --- omega and tau: log-normal random walks ---------------------------------
    def rw(key_n, key_u, w, sd, x, shape, rate):
        lw = jnp.log(w)
        lw_p = lw + sd * jax.random.normal(key_n)
        log_r = log_post_scale(jnp.exp(lw_p), x, shape, rate) - log_post_scale(w, x, shape, rate) + lw_p - lw
        acc = jnp.log(jax.random.uniform(key_u)) < log_r
        return jnp.where(acc, jnp.exp(lw_p), w), acc

    omega, acc_omega = rw(ko, kou, omega, state.omega_sd, beta, priors.omega_shape, priors.omega_rate)
    tau, acc_tau = rw(kt, ktu, tau, state.tau_sd, jnp.concatenate([yes, no]), priors.tau_shape, priors.tau_rate)

    # --- lam: conjugate Gamma ---------------------------------------------------
    J = kappa.shape[0]
    lam = jax.random.gamma(kl, priors.kappa_shape * J + priors.lam_shape) / (priors.lam_rate + kappa.sum())

    new = state._replace(beta=beta, yes=yes, no=no, kappa=kappa, lam=lam, omega=omega, tau=tau, ll=ll)
    return new, SweepStats(info_b.accepted, info_y.accepted, info_n.accepted, acc_kappa, acc_omega, acc_tau, ll.sum())


class Accum(NamedTuple):
    """Running sums over recorded sweeps."""

    lse: jax.Array  # (I, J) log sum_s exp(ll^s)
    s1: jax.Array  # sum_s ll^s
    s2: jax.Array  # sum_s (ll^s)^2
    beta: jax.Array
    yes: jax.Array
    no: jax.Array
    kappa: jax.Array
    n: jax.Array


@dataclass
class Posterior:
    beta: np.ndarray  # (B, I, K+1) evenly spaced draws
    omega: np.ndarray  # (S,)
    tau: np.ndarray  # (S,)
    lam: np.ndarray  # (S,)
    beta_mean: np.ndarray  # (I, K+1) mean of the embedded coordinates (not normalised)
    yes_mean: np.ndarray
    no_mean: np.ndarray
    kappa_mean: np.ndarray
    prob_mean: np.ndarray  # (I, J) posterior mean of P(observed y_ij); NaN where missing
    loglik_trace: np.ndarray  # (burnin + S,)
    accept: dict
    criteria: dict  # lppd, DIC of the paper, WAIC, in-sample accuracy
    priors: dict
    config: dict
    seconds: float
    tuning: dict = field(default_factory=dict)
    final_state: dict = field(default_factory=dict)

    def save(self, path):
        d = asdict(self)
        state = d.pop("final_state")
        arrays = {k: v for k, v in d.items() if isinstance(v, np.ndarray)}
        meta = {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}
        arrays |= {f"state__{k}": np.asarray(v) for k, v in state.items()}
        np.savez_compressed(path, **arrays, meta=np.array(repr(meta)))

    @classmethod
    def load(cls, path):
        import ast

        z = np.load(path, allow_pickle=False)
        meta = ast.literal_eval(str(z["meta"]))
        state = {k.removeprefix("state__"): z[k] for k in z.files if k.startswith("state__")}
        arrays = {k: z[k] for k in z.files if k != "meta" and not k.startswith("state__")}
        return cls(**arrays, **meta, final_state=state)


def initial_state(I, J, K, rng, priors: Priors, group=None) -> dict:
    """Initial values following the original code.

    ``group``: optional length-I array of -1 / +1 / 0; the first angle of
    subjects in group -1 starts in (-pi/2, 0), group +1 in (0, pi/2) (the R code
    does this with Democrats and Republicans), which fixes the reflection of the
    first dimension.
    """
    k2 = np.arange(2, K + 1) ** 2

    def rest(n):  # angles 2..K: 2 phi_k ~ vonMises(0, k^2)
        return rng.vonmises(0.0, k2, (n, K - 1)) / 2

    phi1 = rng.vonmises(0.0, 2.0, I) if group is None else np.asarray(group) * rng.uniform(0, np.pi / 2, I)
    cart = lambda a: np.asarray(to_cartesian(jnp.asarray(a)))
    return dict(
        beta=cart(np.column_stack([phi1, rest(I)])),
        yes=cart(np.column_stack([rng.uniform(-np.pi, np.pi, J), rest(J)])),
        no=cart(np.column_stack([rng.uniform(-np.pi, np.pi, J), rest(J)])),
        kappa=np.full(J, 0.1),
        lam=priors.lam_shape / priors.lam_rate,
        omega=1.0,
        tau=1.0,
    )


def _adapt_scale(scale, acc_rate, low=0.6, high=0.9):
    scale = jnp.where(acc_rate < low, scale * 0.9, scale)
    scale = jnp.where(acc_rate > high, scale * 1.1, scale)
    return jnp.clip(scale, 0.02, 50.0)


def _adapt_sd(sd, acc, low=0.3, high=0.6, down=0.8, up=1.2):
    return jnp.where(acc < low, sd * down, jnp.where(acc > high, sd * up, sd))


class _P(NamedTuple):  # Priors as arrays, so that chains with different priors can share one vmapped call
    omega_shape: jax.Array
    omega_rate: jax.Array
    tau_shape: jax.Array
    tau_rate: jax.Array
    kappa_shape: jax.Array
    lam_shape: jax.Array
    lam_rate: jax.Array


def sample(y, K: int, priors: Priors = Priors(), config: SamplerConfig = SamplerConfig(), group=None, init: dict | None = None) -> Posterior:
    """Run one chain on an (I, J) response matrix (1 / 0 / NaN missing) with latent space S^K."""
    return sample_many([y], K, [priors], config, [config.seed], [group], [init])[0]


def sample_many(ys, K: int, priors=None, config: SamplerConfig = SamplerConfig(), seeds=None, groups=None, inits=None, labels=None) -> list[Posterior]:
    """Run C independent chains in one vectorised GPU program.

    ``ys`` is a list of C response matrices of the same shape (repeat a matrix
    to run several chains on it); ``priors``, ``seeds``, ``groups`` and
    ``inits`` are optional per-chain lists.  At these problem sizes the GPU is
    limited by per-call overhead, so C chains cost little more than one.
    """
    C = len(ys)
    priors = list(priors) if priors is not None else [Priors()] * C
    seeds = list(seeds) if seeds is not None else [config.seed + c for c in range(C)]
    groups = list(groups) if groups is not None else [None] * C
    inits = list(inits) if inits is not None else [None] * C
    labels = list(labels) if labels is not None else [str(c) for c in range(C)]
    datas = [make_data(y) for y in ys]
    data = Data(jnp.stack([d.sign for d in datas]), jnp.stack([d.mask for d in datas]))
    _, I, J = data.sign.shape
    cfg = config
    pri = _P(*(jnp.asarray([getattr(p, f) for p in priors], jnp.float64) for f in _P._fields))

    vals = []
    for c in range(C):
        v = initial_state(I, J, K, np.random.default_rng(seeds[c]), priors[c], groups[c])
        if inits[c]:
            v.update({k: x for k, x in inits[c].items() if k in v})
        vals.append(v)
    values = {k: jnp.asarray(np.stack([np.asarray(v[k], float) for v in vals])) for k in vals[0]}
    state = State(
        **values,
        ll=jax.vmap(loglik_matrix)(values["beta"], values["yes"], values["no"], values["kappa"], data),
        kappa_sd=jnp.full((C, J), cfg.kappa_prop_sd), omega_sd=jnp.full(C, cfg.omega_prop_sd), tau_sd=jnp.full(C, cfg.tau_prop_sd),
        beta_scale=jnp.ones((C, I)), yes_scale=jnp.ones((C, J)), no_scale=jnp.ones((C, J)),
    )
    keys = jax.vmap(jax.random.key)(jnp.asarray(seeds, jnp.uint32))
    block = cfg.jitter_every
    n_iter = cfg.burnin + cfg.n_samples
    leap_rng = np.random.default_rng([int(seeds[0]), 977])

    def draw_jitter(key, leap):
        ks = jax.random.split(key, 3)
        u = lambda k, n, r: jax.random.uniform(k, (n,), minval=r[0], maxval=r[1])
        return Jitter(u(ks[0], I, cfg.beta_step), u(ks[1], J, cfg.item_step), u(ks[2], J, cfg.item_step), leap)

    @partial(jax.jit, static_argnames=("n", "record"))
    def run_block(keys, state, data, acc, pri, leap, n, record):
        def one_chain(key, state, data, acc, pri):
            k_jit, k_run = jax.random.split(key)
            jitter = draw_jitter(k_jit, leap)

            def body(carry, key):
                state, acc = carry
                state, stats = _sweep(key, state, jitter, data, pri, cfg)
                if record:
                    ll = state.ll
                    acc = Accum(jnp.logaddexp(acc.lse, ll), acc.s1 + ll, acc.s2 + ll * ll, acc.beta + state.beta, acc.yes + state.yes, acc.no + state.no, acc.kappa + state.kappa, acc.n + 1)
                    draws = (state.beta.astype(jnp.float32), state.omega, state.tau, state.lam)
                else:
                    draws = ()
                return (state, acc), (stats, draws)

            (state, acc), (stats, draws) = jax.lax.scan(body, (state, acc), jax.random.split(k_run, n))
            mean = lambda a: a.astype(jnp.float64).mean(0)
            return state, acc, SweepStats(*(mean(s) for s in stats[:-1]), stats.loglik), draws

        return jax.vmap(one_chain)(keys, state, data, acc, pri)

    d = K + 1
    z = lambda *shape: jnp.zeros((C,) + shape)
    acc = Accum(jnp.full((C, I, J), -jnp.inf), z(I, J), z(I, J), z(I, d), z(J, d), z(J, d), z(J), jnp.zeros(C, jnp.int32))
    S = cfg.n_samples
    B = min(cfg.n_beta_draws, S)
    keep_index = (np.arange(B) * S) // B + (S // B - 1 if B else 0)
    keep_row = np.full(S, -1)
    keep_row[keep_index] = np.arange(B)
    out_beta = np.empty((C, B, I, d), np.float32)
    out = {"omega": np.empty((C, S)), "tau": np.empty((C, S)), "lam": np.empty((C, S))}
    trace = np.empty((C, n_iter))
    names = ("beta", "yes", "no", "kappa", "omega", "tau")
    acc_sum = {k: np.zeros(C) for k in names}
    it = filled = 0
    t0 = time.perf_counter()
    last_print, last_time = 0, t0
    while it < n_iter:
        n = min(block, n_iter - it)
        record = it >= cfg.burnin
        if not record and it + n > cfg.burnin:
            n = cfg.burnin - it
        ks = jax.vmap(lambda k: jax.random.split(k))(keys)
        keys, k_run = ks[:, 0], ks[:, 1]
        leap = jnp.asarray(leap_rng.integers(1, cfg.max_leap + 1), jnp.int32)
        state, acc, summary, draws = run_block(k_run, state, data, acc, pri, leap, n, record)
        trace[:, it : it + n] = np.asarray(summary.loglik)
        if record:
            rows = keep_row[filled : filled + n]
            out_beta[:, rows[rows >= 0]] = np.asarray(draws[0])[:, rows >= 0]
            for name, arr in zip(out, draws[1:]):
                out[name][:, filled : filled + n] = np.asarray(arr)
            filled += n
            for name, v in zip(names, summary[:6]):
                acc_sum[name] += np.asarray(v).reshape(C, -1).mean(1) * n
        else:  # burn-in tuning (kappa as in the R code)
            sd = jnp.where(summary.acc_kappa <= 0.3, state.kappa_sd * 0.95, state.kappa_sd)
            sd = jnp.where(summary.acc_kappa >= 0.6, sd * 1.05, sd)
            state = state._replace(kappa_sd=sd, omega_sd=_adapt_sd(state.omega_sd, summary.acc_omega), tau_sd=_adapt_sd(state.tau_sd, summary.acc_tau))
            if cfg.adapt_steps:
                state = state._replace(
                    beta_scale=_adapt_scale(state.beta_scale, summary.acc_beta),
                    yes_scale=_adapt_scale(state.yes_scale, summary.acc_yes),
                    no_scale=_adapt_scale(state.no_scale, summary.acc_no),
                )
        it += n
        if cfg.print_every and (it - last_print >= cfg.print_every or it == n_iter):
            now = time.perf_counter()
            rate = (it - last_print) / (now - last_time)
            last_print, last_time = it, now
            f = lambda a: " ".join(f"{v:.2f}" for v in np.asarray(a).reshape(C, -1).mean(1))
            print(
                f"K={K} iter {it:>6}/{n_iter} {rate:.0f} it/s | loglik {' '.join(f'{v:.0f}' for v in trace[:, it - 1])} | acc beta {f(summary.acc_beta)} | psi {f(summary.acc_yes)} "
                f"| kappa {f(summary.acc_kappa)} | omega {f(state.omega)} | tau {f(state.tau)}",
                flush=True,
            )

    seconds = time.perf_counter() - t0
    posts = []
    obs_all = np.asarray(data.mask)
    A = Accum(*(np.asarray(a) for a in acc))
    st = {k: np.asarray(getattr(state, k)) for k in State._fields if k != "ll"}
    for c in range(C):
        n_rec = max(int(A.n[c]), 1)
        obs = obs_all[c]
        log_mean_p = np.where(obs, A.lse[c] - np.log(n_rec), np.nan)
        lppd = float(np.nansum(log_mean_p))
        p_waic = float(np.sum((A.s2[c][obs] - A.s1[c][obs] ** 2 / n_rec) / (n_rec - 1))) if n_rec > 1 else float("nan")
        var_ll = float(np.var(trace[c, cfg.burnin :], ddof=1)) if S > 1 else float("nan")
        criteria = {
            # For Bernoulli data lppd = l(E[Theta | data]), the first term of the paper's DIC.
            "lppd": lppd,
            "var_loglik": var_ll,
            "dic": lppd - 2.0 * var_ll,  # the paper's definition (larger is better)
            "p_waic": p_waic,
            "elpd_waic": lppd - p_waic,
            "accuracy": float(np.mean(np.exp(log_mean_p[obs]) > 0.5)),
            "n_obs": int(obs.sum()),
        }
        posts.append(
            Posterior(
                beta=out_beta[c], omega=out["omega"][c], tau=out["tau"][c], lam=out["lam"][c],
                beta_mean=A.beta[c] / n_rec, yes_mean=A.yes[c] / n_rec, no_mean=A.no[c] / n_rec, kappa_mean=A.kappa[c] / n_rec,
                prob_mean=np.exp(log_mean_p), loglik_trace=trace[c], accept={k: float(v[c]) / max(S, 1) for k, v in acc_sum.items()},
                criteria=criteria, priors=asdict(priors[c]), config=asdict(cfg) | {"K": K, "seed": int(seeds[c]), "label": labels[c]}, seconds=seconds / C,
                tuning={k: float(np.mean(st[k][c])) for k in ("beta_scale", "yes_scale", "no_scale", "omega_sd", "tau_sd", "kappa_sd")},
                final_state={k: st[k][c] for k in ("beta", "yes", "no", "kappa", "lam", "omega", "tau")},
            )
        )
    return posts
