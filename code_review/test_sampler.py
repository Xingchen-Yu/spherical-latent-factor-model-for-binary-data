# GHMC (loop copied from update_beta) driven by the repo's likeliC + gradient_pj, target = SvM prior. Compare with exact angle marginals.
import numpy as np, sys
from scipy import stats, special, integrate
from common import *
def to_angles(X):
    a1 = np.arctan2(X[:,0],X[:,1]) % (2*np.pi); S=np.cumsum(X**2,axis=1)
    return np.column_stack([a1]+[np.arctan2(np.sqrt(S[:,k-1]),X[:,k]) for k in range(2,X.shape[1])])
def bm_z(v, target, nb=50):          # batch-means z score
    b = v[:len(v)//nb*nb].reshape(nb,-1).mean(1); return (b.mean()-target)/(b.std(ddof=1)/np.sqrt(nb))
A = lambda w: special.ive(1,w)/special.ive(0,w)
print(f"{'target':34s} {'acc':>5s}  max|z| mean-resultant   min KS p (thinned)   angles tested")
for p,scheme,w,rngs in [(3,'k2',2.0,(0.02,0.10)),(4,'k2',2.0,(0.02,0.08)),(6,'k2',2.0,(0.01,0.05)),(11,'k2',2.0,(0.01,0.03)),
                        (11,'k2',10.0,(0.004,0.012)),(4,'flat',2.0,(0.02,0.10)),(6,'flat',2.0,(0.02,0.10)),(11,'flat',5.0,(0.02,0.08))]:
    K=p-1; om = w*np.arange(1,K+1)**2 if scheme=='k2' else np.full(K,w)
    a0 = np.concatenate([[np.pi+0.3],np.full(K-1,np.pi/2+0.1)]); x0=getcord_auto(a0[None,:])[0]
    iters=600000; thin=10
    out = run(header('sample',p,om)+fmt(x0)+f"\n{iters} {thin} {rngs[0]} {rngs[1]} 10 0 0 {11+p}\n").strip().split('\n')
    acc=float(out[-1].split()[1]); X=np.array([l.split()[1:] for l in out[:-1]],float)[2000:]
    a=to_angles(X); zs=[bm_z(np.cos(a[:,0]),-A(om[0]))]+[bm_z(np.cos(2*a[:,k]),-A(om[k])) for k in range(1,K)]
    ks=[]
    for k in range(K):
        v=a[::60,k]
        if k==0: cdf=lambda t: stats.vonmises.cdf(t-np.pi,om[0])-stats.vonmises.cdf(-np.pi,om[0])
        else:    cdf=lambda t,k=k: stats.vonmises.cdf(2*t-np.pi,om[k])-stats.vonmises.cdf(-np.pi,om[k])
        ks.append(stats.kstest(v,cdf).pvalue)
    print(f"SvM p={p:2d} {scheme:4s} omega={w:<5} K={K:2d}     {acc:5.2f}  {np.max(np.abs(zs)):8.2f}              {min(ks):8.3f}            {K}")
# von Mises-Fisher sanity check of the integrator copy
for p,kv in [(5,8.0),(11,20.0)]:
    x0=np.ones(p)/np.sqrt(p)
    out = run(header('sample',p,np.ones(p-1))+fmt(x0)+f"\n300000 10 0.05 0.2 10 1 {kv} 5\n").strip().split('\n')
    X=np.array([l.split()[1:] for l in out[:-1]],float)[1000:]; t=X[:,-1]
    dens=lambda s: np.exp(kv*(s-1))*(1-s*s)**((p-3)/2); Z=integrate.quad(dens,-1,1)[0]
    cdf=np.vectorize(lambda s: integrate.quad(dens,-1,s)[0]/Z); m=integrate.quad(lambda s:s*dens(s),-1,1)[0]/Z
    print(f"vMF p={p:2d} kappa={kv:<5}                 {float(out[-1].split()[1]):5.2f}  {abs(bm_z(t,m)):8.2f}              {stats.kstest(t[::40],cdf).pvalue:8.3f}            mu'x")
