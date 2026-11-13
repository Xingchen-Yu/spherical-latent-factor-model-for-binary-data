import numpy as np
from scipy import stats, special
from common import *
from sampler_fns import to_angles, bm_z, A
for p,scheme,w,rngs,seed in [(11,'flat',5.0,(0.02,0.08),101),(11,'flat',5.0,(0.02,0.08),202),(11,'k2',10.0,(0.004,0.012),303),(11,'k2',10.0,(0.004,0.012),404)]:
    K=p-1; om = w*np.arange(1,K+1)**2 if scheme=='k2' else np.full(K,w)
    a0=np.concatenate([[np.pi+0.3],np.full(K-1,np.pi/2+0.1)]); x0=getcord_auto(a0[None,:])[0]
    out=run(header('sample',p,om)+fmt(x0)+f"\n3000000 20 {rngs[0]} {rngs[1]} 10 0 0 {seed}\n").strip().split('\n')
    X=np.array([l.split()[1:] for l in out[:-1]],float)[2000:]; a=to_angles(X)
    zs=np.array([bm_z(np.cos(a[:,0]),-A(om[0]))]+[bm_z(np.cos(2*a[:,k]),-A(om[k])) for k in range(1,K)])
    ks=[]
    for k in range(K):
        cdf=(lambda t: stats.vonmises.cdf(t-np.pi,om[0])-stats.vonmises.cdf(-np.pi,om[0])) if k==0 else (lambda t,k=k: stats.vonmises.cdf(2*t-np.pi,om[k])-stats.vonmises.cdf(-np.pi,om[k]))
        ks.append(stats.kstest(a[::100,k],cdf).pvalue)
    print(f"p={p} {scheme} omega={w} seed={seed}: z-scores {np.round(zs,2)}  min KS p={min(ks):.3f}")
