# Prior variance of theta_ij as the dimension K grows: von Mises-Fisher vs the proposed SvM with k^2 scaling.
import numpy as np
from scipy import stats, special
from common import getcord_auto
rng=np.random.default_rng(3); N=200000
def theta(b,ps,ze,kap):
    e=np.arccos(np.clip((ze*b).sum(1),-1,1))**2-np.arccos(np.clip((ps*b).sum(1),-1,1))**2
    return special.betainc(kap,kap,(e+np.pi**2)/(2*np.pi**2))
def svm(n,om):   # code convention: a1 ~ vM(pi,om1), a_k ~ vM(pi,om_k)/2
    a=np.column_stack([rng.vonmises(0,om[0],n)+np.pi]+[(rng.vonmises(0,o,n)+np.pi)/2 for o in om[1:]]); return getcord_auto(a)
def vmf(n,d,w):
    mu=np.zeros(d); mu[0]=1
    return stats.vonmises_fisher(mu,w,seed=rng).rvs(n) if d>2 else (lambda t: np.column_stack([np.cos(t),np.sin(t)]))(rng.vonmises(0,w,n))
Ks=[1,2,3,4,5,6,8,10,15,20,30,50]
print("Var(theta_ij) by K  (theta mean is 0.5; a point mass at 0.5 has variance 0; MC s.e. about 0.0005)")
print("K:".ljust(44)+' '.join(f"{k:6d}" for k in Ks))
for om,tau,kap in [(0.5,0.5,1.0),(2,2,1.0),(10,10,1.0),(2,2,50.0),(10,2,50.0),(10,0.2,75.0)]:
    v1=[theta(vmf(N,K+1,om),vmf(N,K+1,tau),vmf(N,K+1,tau),kap).var() for K in Ks]
    v2=[theta(svm(N,om*np.arange(1,K+1)**2),svm(N,tau*np.arange(1,K+1)**2),svm(N,tau*np.arange(1,K+1)**2),kap).var() for K in Ks]
    lab=f"omega={om}, tau={tau}, kappa={kap}"
    print(f"vMF          {lab:30s} "+' '.join(f"{v:6.4f}" for v in v1))
    print(f"SvM k^2      {lab:30s} "+' '.join(f"{v:6.4f}" for v in v2))
# the full hierarchical prior used in the paper/code, K = 10 and 30
for K in (2,10,30):
    om=rng.gamma(1,10,N); tau=rng.gamma(1,1/5,N); lam=rng.gamma(2,1/150,N); kap=rng.gamma(1,1/lam)
    k2=np.arange(1,K+1)**2
    def svm_h(w):
        a=np.column_stack([rng.vonmises(0,w)+np.pi]+[(rng.vonmises(0,w*c)+np.pi)/2 for c in k2[1:]]); return getcord_auto(a)
    th=theta(svm_h(om),svm_h(tau),svm_h(tau),kap)
    print(f"hierarchical prior (paper hyperparameters), K={K:2d}: Var(theta)={th.var():.4f} [Beta(1/2,1/2) has 0.125], P(theta<0.05)={np.mean(th<0.05):.3f}, P(0.45<theta<0.55)={np.mean(abs(th-.5)<.05):.3f}")
