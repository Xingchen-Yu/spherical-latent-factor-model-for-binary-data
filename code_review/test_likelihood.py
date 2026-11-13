# Likelihood value/gradient vs a high-precision reference (mpmath incomplete beta, 40 digits), incl. far tails.
import numpy as np, jax, jax.numpy as jnp, mpmath as mp
jax.config.update("jax_enable_x64", True)
from common import *
from refs import ref_logprior, constrained
mp.mp.dps = 40
rng = np.random.default_rng(7)
PI2 = mp.pi**2
def logG_and_score(e,k,y):          # log P(y) and d/de log P(y), exact
    x = (mp.mpf(float(e))+PI2)/(2*PI2); k = mp.mpf(float(k))
    low = lambda t: mp.betainc(k,k,0,t,regularized=True) if t<=0.5 else 1-mp.betainc(k,k,0,1-t,regularized=True)
    Gv = low(x); Gc = low(1-x)
    g = x**(k-1)*(1-x)**(k-1)/mp.beta(k,k)/(2*PI2)
    return (mp.log(Gv), g/Gv) if y==1 else (mp.log(Gc), -g/Gc)
def e_beta(x,A,B): return jnp.arccos(B@x)**2 - jnp.arccos(A@x)**2
def e_yes(x,A,z):  return jnp.arccos(A@z)**2 - jnp.arccos(A@x)**2
def e_no(x,A,z):   return jnp.arccos(A@x)**2 - jnp.arccos(A@z)**2
worst={}; 
def upd(k,v): worst[k]=max(worst.get(k,0.0),float(v))
minlog = 0
for p in [3,4,6,11]:
  K=p-1; om = 0.7*np.arange(1,K+1)**2
  for regime in ['kappa~1','kappa~75 (prior mean)','kappa~500']:
    T=15; n=20; inp = header('grad',p,om)+f"{T}\n"; cases=[]
    for t in range(T):
        rs = lambda m: (lambda v: v/np.linalg.norm(v,axis=1,keepdims=True))(rng.normal(size=(m,p)))
        x=rs(1)[0]; A=rs(n); B=rs(n); z=rs(1)[0]
        sc = {'kappa~1':1.0,'kappa~75 (prior mean)':75.0,'kappa~500':500.0}[regime]
        kap = rng.exponential(sc,n)+0.05; shape=float(rng.exponential(sc)+0.05); y=rng.integers(0,2,n).astype(float)
        cases.append((x,A,B,kap,y,z,shape)); inp += f"{n}\n"+'\n'.join(fmt(v) for v in (x,A,B,kap,y,z))+f"\n{shape!r}\n"
    it = iter([l.split() for l in run(inp).strip().split('\n')])
    for (x,A,B,kap,y,z,shape) in cases:
        r={}
        for _ in range(8): l=next(it); r[l[0]]=np.array(l[1:],float)
        xj=jnp.array(x); P=np.eye(p)-np.outer(x,x); lpr=float(ref_logprior(xj,jnp.array(om))); 
        gpr=np.array(constrained(ref_logprior)(xj,jnp.array(om)))
        for name,lk,gk,ef,args,kv in [('beta','likeli_beta','gradient_beta',e_beta,(A,B),kap),
                                      ('yes(psi)','likeli_yes','gradient_yes',e_yes,(A,z),np.full(n,shape)),
                                      ('no(zeta)','likeli_no','gradient_no',e_no,(A,z),np.full(n,shape))]:
            e=np.array(ef(xj,*args)); res=[logG_and_score(e[i],kv[i],y[i]) for i in range(n)]
            val=float(sum(v for v,_ in res))+lpr; w=np.array([float(s) for _,s in res]); minlog=min(minlog,min(float(v) for v,_ in res))
            if not np.all(np.isfinite(w)): upd(f'{name}: cases with non-finite exact score (skipped)',1); continue
            # constrained Jacobian of e wrt x_{1:K}
            s=np.sign(x[-1]); eh=lambda u: ef(jnp.concatenate([u,s*jnp.sqrt(1-jnp.sum(u**2))[None]]),*args)
            J=np.array(jax.jacobian(eh)(xj[:-1])); gc=np.concatenate([J.T@w,[0]])+gpr
            upd(f'{name}: rel err of log target value', abs(val-r[lk][0])/abs(val))
            upd(f'{name}: rel err of gradient (paper constraint)', np.linalg.norm(gc-r[gk])/np.linalg.norm(gc))
            if not np.all(np.isfinite(r[gk])): upd(f'{name}: non-finite code gradient',1)
print("worst case over p in {3,4,6,11} x 3 kappa regimes x 15 random configurations (n=20 terms each):")
for k,v in worst.items(): print(f"  {k:55s} {v:.2e}")
print("  most extreme single log-probability exercised:", f"{minlog:.0f}")
