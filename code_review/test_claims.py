import numpy as np
from scipy import stats, special
rng=np.random.default_rng(5)
print("(a) link limit, eq. (linklimit): ratio g_kappa(z)/Gaussian at z = 1 and 2 standard deviations (sd = pi^2/sqrt(2k+1))")
for k in [50,500,5000,50000]:
    sd=np.pi**2/np.sqrt(2*k+1)
    for c in (1,2):
        z=c*sd; g=stats.beta.pdf((z+np.pi**2)/(2*np.pi**2),k,k)/(2*np.pi**2)
        paper=np.sqrt((2*k+1)/(2*np.pi**5))*np.exp(-(2*k+1)/np.pi**4*z*z); fixed=np.sqrt((2*k+1)/(2*np.pi**5))*np.exp(-(2*k+1)/(2*np.pi**4)*z*z)
        print(f"   kappa={k:6d} z={c}sd: as printed {g/paper:8.4f}   with exponent (2k+1)z^2/(2 pi^4) {g/fixed:8.4f}")
print("(b) vMF vs Gaussian normalising constants (ratio of the two as omega grows), d = 3 and 6")
for d in (3,6):
    for w in [10,100,1000]:
        logc=(d/2-1)*np.log(w)-(d/2)*np.log(2*np.pi)-(np.log(special.ive(d/2-1,w))+w) + w   # C_d(w) e^{w}: density at the mode
        print(f"   d={d} omega={w:5d}: vs (w/2pi)^(d/2) as printed {np.exp(logc-(d/2)*np.log(w/(2*np.pi))):7.4f}   vs (w/2pi)^((d-1)/2) {np.exp(logc-((d-1)/2)*np.log(w/(2*np.pi))):7.4f}")
print("(c) Euclidean footnote: z = mu + sum_k alpha_k beta_k, K = 3000")
n=200000; K=3000; z=rng.normal(0,np.sqrt(.5),n)
for k in range(1,K+1): z+=rng.normal(0,np.sqrt(.5),n)*rng.normal(0,np.sqrt(6)/(np.pi*k),n)
th=stats.norm.cdf(z)
print(f"   Var(z)={z.var():.4f} (claimed 1)   excess kurtosis={stats.kurtosis(z):.3f} (normal: 0; analytic limit 0.6)")
print(f"   KS test of theta vs Uniform(0,1): D={stats.kstest(th,'uniform').statistic:.4f}, p={stats.kstest(th,'uniform').pvalue:.1e};  Var(theta)={th.var():.4f} (uniform: 0.0833)")
print("(d) SvM Gaussian limit (eq. limitprior) normalising constants, K=4, omega_k = w*k^2")
for w in [5,50,500]:
    om=w*np.arange(1,5)**2; K=4
    num=-K*np.log(2*np.pi)+(K-1)*np.log(2)-np.sum(np.log(special.ive(0,om))+om)+om.sum()
    den=-(K/2)*np.log(2*np.pi)+(K-1)*np.log(2)+0.5*np.sum(np.log(om))
    print(f"   w={w:4d}: ratio at phi=0 = {np.exp(num-den):.4f}")
