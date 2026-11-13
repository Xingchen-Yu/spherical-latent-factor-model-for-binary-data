# Code and method review (November 2026)

Scope: every file in this repository at commit `63f7042`, and the arXiv paper
(2008.05109v1) including its supplementary material. Reviewed by Claude at the
author's request before journal submission.

## Verdict

**The method and the code are correct.** No error was found that affects any
result in the paper.

* The gradient of the log prior and log Jacobian (supplement, eq. for
  `∇ log p_Hπ + ∇ log p_J`) is correct, and `gradient_pj` with the pre-built
  matrices from `jacobian_init` / `jacobian_init2` implements it exactly
  (agreement with automatic differentiation: 1e-13).
* The likelihood, its gradients, and all seven update steps were tested
  end to end and are correct.
* The proposed prior does what the paper says: `Var(θ_ij)` levels off as `K`
  grows, where the von Mises–Fisher prior collapses to a point mass.

What needs fixing before submission is in the **paper text**: four formulas
have typos or are stated too strongly (section 3). None changes a result.
Section 4 lists code robustness points, none of which affects correctness.

## 1. How it was checked

The machine had no R, so the C++ was tested directly: `build_harness.py` and
`build_geweke.py` copy the C++ text out of
`source/spherical_factor_model_wrapper.R` and `source/spherical_factor_model_omp.cpp`
**unchanged** (only the Rcpp headers are swapped for plain Armadillo and a
12-line stand-in for `Rcpp::List`) and compile it. Every number below comes
from the repository's own code. `./run_checks.sh` reproduces everything
(needs g++, curl, uv; about 10 minutes). Raw output is in `logs/`.

| # | Check | Script | Result |
|---|---|---|---|
| 1 | Hand derivation of the Hausdorff density and of the prior + Jacobian gradient; index-by-index comparison with `gradient_pj`, `jacobian_init`, `jacobian_init2` | section 2 | matches |
| 2 | `likeliC` and `gradient_pj` vs an independent reference built from the *angular* density divided by the surface element, differentiated with JAX autodiff. p = 3, 4, 5, 6, 8, 11; ω_k = k²ω, constant, and random; 720 random points | `test_gradients.py` | value 1e-6 (the Bessel polynomial), gradient **1.7e-13** |
| 3 | `likeli_beta/yes/no` and `gradient_beta/yes/no` vs a 40-digit reference (mpmath incomplete beta), κ from 0.05 to ~3000, log-probabilities down to −1238 | `test_likelihood.py` | value 4e-8, gradient **2e-9** |
| 4 | Geodesic HMC loop of `update_beta` with `likeliC` + `gradient_pj` as target: samples vs the exact von Mises marginals of every angle, p = 3 … 11, both ω schemes; plus von Mises–Fisher targets | `test_sampler.py`, `test_sampler_long.py` | all marginals match |
| 5 | **Geweke (2004) joint test of the whole sampler**: alternate "simulate data given parameters" with the repository's `update_beta`, `update_yes`, `update_no`, `update_kappa`, `update_omega` (for ω and τ) and the λ draw, in the order of the main script. If any density, gradient-based proposal, acceptance ratio or normalising constant were wrong, the parameters would drift away from the prior. p = 3, 4, 5, 6, 8; 22–52 statistics each | `test_geweke.py` | z-scores standard normal, none above 3 |
| 6 | Same test with the log-Jacobian deleted from `likeliC` (to show the test can fail) | `logs/geweke_p4_mutation_no_jacobian.txt` | fails clearly: 12 of 28 statistics beyond 3, max 8.9 |
| 7 | Prior variance of θ_ij against K = 1 … 50, vMF vs SvM(ω, 2²ω, …, K²ω) | `test_prior_var.py` | section 2.4 |
| 8 | Limit statements and the Euclidean footnote in the paper | `test_claims.py` | section 3 |

Not run: the full R main script on the Senate data (no R here). The JAX port
in `python/` is checked against the same references and is used to rerun the
simulations.

## 2. The mathematics, step by step

### 2.1 Two coordinate conventions (worth one sentence in the paper)

The paper uses `x_1 = cos φ_1 ∏cos φ_k`, `x_2 = sin φ_1 ∏cos φ_k`,
`x_{k+1} = sin φ_k ∏_{l>k} cos φ_l`, with φ_1 ∈ [−π, π], φ_k ∈ [−π/2, π/2].
The code (`getcord_auto`) uses the standard `x_1 = sin a_1 ∏ sin a_k`,
`x_2 = cos a_1 ∏ sin a_k`, `x_{k+1} = cos a_k ∏_{l>k} sin a_l`, with
a_1 = φ_1 + π and a_k = φ_k + π/2 (as the README says). So

    x_code = −(x_2, x_1, x_3, …, x_{K+1})_paper,

an orthogonal map: distances, likelihood and prior are identical. The only
place it shows is the first-angle term: the paper's `exp{ω_1 x_1/√(x_1²+x_2²)}`
is `exp{−ω_1 x_2/√(x_1²+x_2²)}` in the code, and the paper's gradient term
`ω_1 x_2 (x_1²+x_2²)^{-3/2} (x_2, −x_1)` is `ω_1 x_1 (x_1²+x_2²)^{-3/2} (x_2, −x_1)`
in `gradient_pj`. Each is right in its own convention.

### 2.2 Hausdorff density (eq. `newpriorHausdorff`) — correct

The surface element is `dA = ∏_{k=2}^K |cos φ_k|^{k−1} dφ`, and
`cos² φ_k = S_k / S_{k+1}` with `S_k = Σ_{t≤k} x_t²`. The product telescopes:

    ∏_{k=2}^K (S_k/S_{k+1})^{(k−1)/2} = ∏_{k=2}^K S_k^{1/2}     (using S_{K+1} = 1),

so `p_H(x) = p(φ) / ∏_{k=1}^{K−1} √S_{k+1}`. The paper writes the product to
`k = K`; the extra factor is `√S_{K+1} = 1`, so both are the same on the sphere.
`cos 2φ_k = 1 − 2 x_{k+1}²/S_{k+1}` gives the exponent as printed. `likeliC`
implements exactly this, including the normalising constants
`1/(2π I_0(ω_1))` and `1/(π I_0(ω_k))` that the ω update needs.

### 2.3 Gradient of log prior + log Jacobian — correct, and the code matches

With `x_{K+1}` eliminated by the constraint (`S_{K+1} = 1`, `x_{K+1}² = 1 − S_K`):

| Term of log p | Derivative w.r.t. x_t, t ≤ K | Paper | Code (`gradient_pj`) |
|---|---|---|---|
| k = K: `−ω_K(1 − 2 S_K)` | `4 ω_K x_t` | 1st term | `4 * omega_vec(p-2) * temp_x` |
| first angle | see 2.1 | 2nd term | `pow(kobe(1),-1.5) * omega_vec(0) * x(0) * {x(1), -x(0)}` |
| 2 ≤ k ≤ K−1: `−2ω_k x_{k+1}²/S_{k+1}` | `4 x_t Σ_{k ≥ max(t−1,2)} ω_k x_{k+1}²/S_{k+1}²` | `4 x ∘ Σ_{-2}[…]` | `4 * temp_x % (jac_mat2 * (omega % x² / S²))` |
| same term, t = k+1 | `−4 ω_{t−1} x_t / S_t`, t = 3 … K | 4th term | `- temp` with `temp(2..p-2) = omega(1..p-3) / kobe(2..p-2)` |
| Jacobian `−½ Σ_{k=1}^{K−1} log S_{k+1}` | `−x_t Σ_{c ≥ max(t,2)}^{K} 1/S_c` | `−x ∘ Σ_{-1}[1/S_2 … 1/S_K]` | `- jac_mat * (1/kobe(1..p-2)) % temp_x` |

`jacobian_init(p)` returns the K × (K−1) matrix with entry (t, c) = 1 for
t ≤ c, c = 2 … K, which is the paper's Σ_{-1}; `jacobian_init2(p)` returns the
K × (K−2) matrix for c = 3 … K, which is Σ_{-2}. All index ranges were checked
for p = 3 (the `else` branch), p = 4 (the smallest case of the general branch)
and larger p.

Treating `x_{K+1}` as dependent is valid: the resulting vector differs from
the unconstrained gradient by a multiple of `x`, which the tangent projection
removes (confirmed numerically, row 2 of the table: 1.7e-14).

### 2.4 The prior penalises extra dimensions — confirmed

`Var(θ_ij)` by Monte Carlo (200,000 draws each; a point mass at ½ has variance 0):

| ω, τ, κ | prior | K=1 | 2 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|---|---|
| 0.5, 0.5, 1 | vMF | .041 | .024 | .017 | .010 | .005 | .003 | .001 |
| | SvM k² | .041 | .027 | .023 | .020 | .017 | .016 | .016 |
| 2, 2, 50 | vMF | .116 | .139 | .142 | .136 | .113 | .083 | .047 |
| | SvM k² | .116 | .115 | .114 | .113 | .113 | .112 | .112 |
| 10, 0.2, 75 | vMF | .190 | .185 | .177 | .161 | .135 | .103 | .062 |
| | SvM k² | .190 | .185 | .179 | .174 | .170 | .168 | .167 |

Under the paper's hyperpriors the variance at K = 10 is 0.122
(Beta(½, ½): 0.125), as intended.

### 2.5 Everything else in the code — correct

* **Link and likelihood.** `betaInc_log`, `betaInc_log_lower` and the four
  branches of the score `g/G`, `−g/(1−G)` in `gradient_beta/yes/no` are right
  and stay accurate deep in the tails.
* **Likelihood gradients** match the supplement's `e'_{i,j,t}` for β, ψ and ζ.
* **GHMC** (`update_beta/yes/no`): momentum projection, half step, geodesic
  flow, second half step and acceptance ratio follow Byrne & Girolami (2013).
* **κ_j**: log-normal random walk with the correct Jacobian term and
  Gamma(kappa_a, λ) prior. **λ**: correct conjugate Gamma draw.
* **ω, τ** (`update_omega`): correct, including the Bessel normalising constants
  and the factor k² (`dummy`). The Jacobian is rightly left out (free of ω).
* **Hyperparameters** in the main script equal those in the paper
  (a_ω = a_τ = 1, b_ω = 1/10, b_τ = 5, a_λ = 2, b_λ = 150, c = 1).
* **DIC**: the code computes `Σ log(mean_s p(y_ij | θ^s)) − 2 Var(ℓ)`. For
  Bernoulli data the first term equals `ℓ(E[Θ | data])` exactly, so it agrees
  with the formula in the paper.
* **Missing votes** are imputed from their full conditional, which is valid;
  the log-likelihood and DIC use observed cells only.
* **Data preparation** (`ymat_spit`, `readKH2`): vote codes and the 40% rule
  are as described in the paper.

## 3. Paper: corrections to make

Numerical evidence for 1–3 is in `logs/paper_claims.txt`.

1. **Eq. (linklimit), Gaussian limit of the link.** The exponent is missing a
   factor 2. The scaled Beta(κ, κ) has variance `π⁴/(2κ+1)`, so the Gaussian is
   `√((2κ+1)/(2π⁵)) exp{−(2κ+1) z²/(2π⁴)}`. As printed, the ratio at one
   standard deviation tends to 1.65, not 1; with the 2 it is 1.0000.
2. **vMF limit (section 3.2).** The denominator should be
   `(ω/2π)^{(d−1)/2}`, not `(ω/2π)^{d/2}` (the limit is a Gaussian on the
   (d−1)-dimensional tangent space). As printed the ratio tends to 0.
3. **Footnote 1 and supplement 7.1 (Euclidean prior).** The variance
   calculation is right (it tends to 1), but the conclusion is not: the
   central limit theorem does not apply because the terms `α_k β_k` have
   variances decaying like 1/k², so the first few dominate. `z_ij` converges
   to a non-Gaussian limit (excess kurtosis 0.6 analytically, 0.57 simulated),
   and `θ_ij = Φ(z_ij)` is not uniform (KS test p = 4e-15 with 200,000 draws;
   variance 0.081 vs 0.083). Suggested wording: "the prior variance of
   `z_ij` converges to 1, so the induced prior on θ_ij converges to a
   non-degenerate distribution". Also in 7.1: divide by the standard
   deviation, not the variance; `E` is missing in "(z_ij(K)) = 0"; the finite
   sum should be a limit.
4. **All "lim ratio = 1" statements** (eqs. linklimit, limitprior, the vMF
   one) hold at the mode or on the 1/√precision scale, not pointwise at a
   fixed argument (there the ratio tends to 0 or ∞). Safer: "X, rescaled by
   √precision, converges in distribution to a Gaussian".
5. **Typos.**
   * Eq. (sph_coord): products for x_1, x_2, x_3 should run to `cos φ_K`
     (printed `K−1`).
   * After eq. (limitprior): covariance is `diag{1/ω_1, 1/(4ω_2), …, 1/(4ω_K)}`.
   * "mean direction **0**" for the vMF priors: a mean direction has unit
     norm; say "mean direction (1, 0, …, 0)" or "all angles zero".
   * Section 4, GHMC steps: momentum is γ in steps 1–2 and 3(a), then φ and ν
     in 3(b)–(c) ("ν = ‖φ‖" should be ‖γ‖), while φ also denotes the angles.
     `N(0, I_d)`: d is not defined (it is K+1).
   * Section 4: λ has a **Gamma** full conditional, not inverse-Gamma.
   * Section 4: acceptance target for the random walks: text says 40%, code
     adapts towards 30–60%.
   * Supplement 7.3.2, ψ block: the density is headed `p(x_ζj | …)` and has ψ
     and ζ swapped inside G (it should be `{arccos(x_ζ'x_β)}² − {arccos(x_ψ'x_β)}²`
     as in e_ij just below).
   * Eqs. (finalpriors): subscripts `ψ_i` and `β_j` should be `ψ_j`, `β_i`;
     "β_1, …, β_J" in the first sentence of 3.2 should be β_I.
   * Scenario 1 coverage: text says 98%, the figure caption 99%. The interval
     quoted for 68% (0.578, 0.782) is wider than a normal interval
     (0.589, 0.771) or Clopper–Pearson (0.579, 0.770); please recheck.
   * Section 5 step sizes (0.01–0.03 or 0.05; 0.01–0.07 or 0.105) differ from
     the repository defaults (0.01–0.04; 0.01–0.07). State which were used.
   * Code link is still `github.com/forreviewonly/review/`.
6. **Worth a sentence.** The SvM density with respect to surface measure is
   unbounded at the coordinate poles (`S_k → 0`, factor `S_k^{-1/2}`). It is
   integrable and harmless when ω_k is large, but τ has prior mean 0.2, so
   ψ_j and ζ_j can sit near the pole of the second angle where the gradient is
   large. The sampler stays exact (the accept step corrects), only less
   efficient there.

## 4. Code: robustness and housekeeping (no effect on correctness)

1. **NaN is accepted.** `if(accept < as_scalar(arma::randu(1)))` rejects only
   when the comparison is true; if `h_new` is NaN (for example `acos` of a dot
   product that rounds to 1 + 1e-16) the proposal is *accepted*. Use
   `if(!(accept >= u))`, and clamp the dot products to [−1, 1].
2. **Division by `x_{K+1}`** in `gradient_beta/yes/no`. The prior concentrates
   the last angle where `x_{K+1} ≈ 0`, so a large multiple of `x` is added and
   then projected away. Exact in theory, loses digits in practice. The
   unconstrained gradient avoids it, and the paper already notes the two are
   equivalent after projection.
3. **Imputation runs every 50 iterations** (inside the progress block). Still
   a valid sampler. Dropping missing cells from the likelihood gives the same
   posterior with no imputation step; the JAX port does this.
4. **`test = T` is the default**, which hides 5% of the votes. If the paper's
   DIC and accuracy were computed on all observed votes, set the default to
   `F` or say so in the README.
5. `jacobian_init(3)` loops over `4:3` (R counts down) and returns a wrong
   matrix; it is never called for p = 3, but a guard would be safer.
6. `threads = 6` is hard-coded in the OpenMP file; `core = 10` in the main script.
7. `spit_omega_true` uses an undefined global `c_shrink`; `x_to_theta`,
   `waic_compute`, `atan2_kobe` are unused in the main script.
8. `data/.Rhistory` is a scratch file with local paths; remove it from the repo.
9. The Euclidean probit code and the principal-nested-spheres code used for
   the comparison figures are not in the repository. Reviewers will ask for
   them.

## 5. Files in this folder

    REVIEW.md              this document
    run_checks.sh          builds and runs everything
    build_harness.py       extracts the C++ density/gradient functions, adds harness_main.cpp
    build_geweke.py        extracts all C++ (both source files), adds geweke_main.cpp
    refs.py                independent reference densities (angles + surface element) in JAX
    test_gradients.py      check 2        test_likelihood.py   check 3
    test_sampler*.py       check 4        test_geweke.py       checks 5–6
    test_prior_var.py      check 7        test_claims.py       check 8
    logs/                  output of the runs reported above
