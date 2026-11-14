# Paper: things to fix or state before submission

Substantive items only (no typo list; those are in `REVIEW.md`, section 3, item 5). Each item says what to change and
where the evidence is. Numbers from the JAX reproduction are in `python/README.md` and `python/figures/criteria_table.csv`.

## A. Formulas that are wrong as printed

1. **Gaussian limit of the link (eq. `linklimit`).** The exponent is missing a factor 2. It should be
   `sqrt((2κ+1)/(2π⁵)) · exp{−(2κ+1) z² / (2π⁴)}` (the scaled Beta(κ, κ) has variance π⁴/(2κ+1)).
   Evidence: `REVIEW.md` 3.1, `logs/paper_claims.txt`.
2. **vMF limit (Section 3.2).** The normalising power should be `(ω/2π)^{(d−1)/2}`, not `(ω/2π)^{d/2}`.
   Evidence: `REVIEW.md` 3.2.
3. **Footnote 1 and Supplement 7.1 (prior of the Euclidean model as K grows).** The variance calculation is right, the
   conclusion is not: the central limit theorem does not apply (the terms have variances decaying like 1/k², so the
   first few dominate), the limit of z_ij is not Gaussian and θ_ij is not uniform. Replace with: "the prior variance
   of z_ij converges, so the induced prior on θ_ij converges to a non-degenerate distribution".
   Evidence: `REVIEW.md` 3.3.
4. **Statements of the form "lim ratio = 1"** (eqs. `linklimit`, `limitprior`, vMF). They hold on the 1/√precision
   scale, not pointwise. Rephrase as convergence in distribution after rescaling. Evidence: `REVIEW.md` 3.4.
5. **Supplement 7.3.2, full conditional of ψ_j.** The density is headed with ζ and has ψ and ζ swapped inside G.
   The code is right; the formula as printed is the conditional of the other parameter.
6. **Full conditional of λ is Gamma**, not inverse-Gamma (Section 4). The code samples a Gamma.

## B. Things the paper does not say but the results depend on

7. **Priors of the Euclidean comparison model.** State them: (μ_j, α_j) ~ N(0, I) and β_ik ~ N(0, 1/k²). The k²
   shrinkage is what lets DIC penalise extra Euclidean dimensions and makes the comparison with the spherical model
   fair. Without it (β_i ~ N(0, I)) the Euclidean model wins on DIC in Scenarios 2 and 3 and the paper's conclusion
   reverses. Evidence: `python/README.md`, "What the reproduction shows".
8. **Treatment of missing votes in DIC.** State that the log-likelihood (both the lppd term and the variance
   penalty) is computed over observed votes only, for both models. The original Euclidean script summed the penalty
   over imputed cells as well, which lowered the Euclidean DIC by 340 to 2,700 on the Senate data; this is now
   fixed in `main_script/Euclidean_nd.R`, and the Senate figure should be regenerated with the fixed version.
9. **How κ_j was generated in the simulations.** The paper gives the positions' distribution but not κ_j. The stored
   truth for Scenario 1 (`data/o1_2_sphdata.Rdata`) has κ_j ~ Gamma(50, rate 1/2), mean 100.
10. **Which principal-nested-spheres variant.** The figures used great subspheres (`shapes::pns(..., sphere.type =
    "great")`) on the normalised posterior mean. Say so; small subspheres give different percentages.
11. **MCMC settings actually used.** Section 5 quotes step-size ranges that differ from the repository defaults, the
    text says the random walks target 40% acceptance while the code adapts towards 30–60%, and run lengths differ
    between scripts. State the settings used for the reported results.
12. **Why DIC and not WAIC.** One sentence is enough: pointwise WAIC barely penalises the extra, heavily shrunk
    dimensions, so it keeps improving slowly with K and does not identify the dimension (in every dataset its
    maximum is at K = 8 to 10), whereas the penalty 2·Var(log-likelihood) grows with K. Evidence:
    `python/figures/criteria_dic_waic.png`. Worth adding that WAIC agrees with DIC on spherical vs Euclidean at the
    selected dimensions.
13. **One-dimensional Euclidean fits come from different code** (`Euclidean_1d.R` in the circular-model repository;
    `Euclidean_nd.R` does not run for one dimension). Its prior is β_i ~ N(0, 1), consistent with item 7. Mention
    the file in the code README.

## C. Claims to recheck against the reproduction

14. **102nd Senate: "a Euclidean model of dimension 3 provides the best fit overall".** With missing votes treated
    the same way in both models, the Euclidean and spherical K = 3 models are indistinguishable (DIC −16,464 vs
    −16,488; the two Euclidean chains differ by 73). Suggested wording: the two geometries fit equally well.
15. **Scenario 1 coverage (98% / 99% in the text and caption, 68%).** On the original data and truth the JAX fit
    gives 89% for dimension 1 and 64% for dimension 2 with axis reflections only (94% / 58% after a Procrustes
    rotation). The first number depends on how the posterior is aligned with the truth; say how. The under-coverage
    in dimension 2 is systematic, not noise: the posterior medians are more spread out than the truth
    (`python/figures/sim_recovery_sim_2_reflection.png`), i.e. the scale of the second dimension is not recovered.
    A referee will ask; the likely reason (data generated with equal precisions, fitted with k² precisions, and scale
    partly traded off against κ_j) deserves a sentence.
16. **DIC is noisy.** The penalty is the variance of one chain's log-likelihood trace. Between two chains DIC differs
    by 10 to 80 at the selected dimensions and by several hundred at some larger K. The reported differences between
    the two geometries (500 to 1,500 in Scenarios 1 to 3, about 600 for the 115th Senate) are well outside that;
    the Scenario 4 and 102nd Senate differences (under 100) are not. Report DIC from more than one chain, or give
    the chain-to-chain range.
17. **Principal-nested-spheres percentages quoted for the Senates** (">95%", "close to 99%", "less than 70%").
    These reproduce with great subspheres (96%, 99%, 67%); with small subspheres the 102nd Senate figure is 88%, so
    the "less than 70%" statement depends on item 10.

## D. Worth one sentence each

18. **Two coordinate conventions.** The paper's spherical coordinates and the code's differ by a fixed relabelling
    (`REVIEW.md` 2.1). The README says so; the paper's computation section should too.
19. **The SvM density is unbounded at the coordinate poles** with respect to surface measure (factor S_k^{-1/2}).
    Integrable and harmless, but it is where the sampler is least efficient when τ is small (`REVIEW.md` 3.6).
20. **Code availability.** The link in the paper is still the anonymised review URL. The repository now has the
    Euclidean code; the simulation driver and figure scripts used for the submitted figures are not in it yet.
