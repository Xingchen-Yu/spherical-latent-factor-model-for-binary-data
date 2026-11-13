import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.scipy.special import betainc, i0e
from common import *
rng = np.random.default_rng(1)

def angles(x):                       # code convention: x1 = sin a1 * prod sin, x2 = cos a1 * prod sin, x_{k+1} = cos a_k * prod_{l>k} sin a_l
    a1 = jnp.arctan2(x[0], x[1])
    S = jnp.cumsum(x**2)
    ak = jnp.arctan2(jnp.sqrt(S[1:-1]), x[2:])     # k = 2..K
    return a1, ak
def ref_logprior(x, om):             # log density w.r.t. surface measure, built from the ANGULAR density / surface element
    a1, ak = angles(x); K = x.shape[0]-1
    lp = -jnp.log(2*jnp.pi) - (jnp.log(i0e(om[0]))+om[0]) - om[0]*jnp.cos(a1)
    lp += jnp.sum(-jnp.log(jnp.pi) - (jnp.log(i0e(om[1:]))+om[1:]) - om[1:]*jnp.cos(2*ak))
    lp -= jnp.sum(jnp.arange(1,K)*jnp.log(jnp.sin(ak)))        # dA = prod_{k=2}^K sin^{k-1}(a_k) da
    return lp
def G(e,k): return betainc(k,k,(e+jnp.pi**2)/(2*jnp.pi**2))
def ll(e,k,y): 
    Gv = G(e,k); return jnp.sum(jnp.where(y==1, jnp.log(Gv), jnp.log1p(-Gv)))
def ref_beta(x, om, yes, no, kap, y): return ll(jnp.arccos(no@x)**2 - jnp.arccos(yes@x)**2, kap, y) + ref_logprior(x,om)
def ref_yes(x, om, B, z, k, y):  return ll(jnp.arccos(B@z)**2 - jnp.arccos(B@x)**2, k, y) + ref_logprior(x,om)
def ref_no(x, om, B, z, k, y):   return ll(jnp.arccos(B@x)**2 - jnp.arccos(B@z)**2, k, y) + ref_logprior(x,om)
def constrained(f):                  # treat x_1..x_K free, x_{K+1} = sign*sqrt(1-sum) as in the paper; gradient padded with 0
    def g(x,*a):
        s = jnp.sign(x[-1])
        fh = lambda u: f(jnp.concatenate([u, s*jnp.sqrt(1-jnp.sum(u**2))[None]]),*a)
        return jnp.concatenate([jax.grad(fh)(x[:-1]), jnp.zeros(1)])
    return g

worst = {}
def upd(k,v): worst[k] = max(worst.get(k,0.0), float(v))
for p in [3,4,5,6,8,11]:
  K = p-1
  for scheme in ['k2','flat','random']:
    om = {'k2': 0.7*np.arange(1,K+1)**2, 'flat': np.full(K,2.0), 'random': rng.gamma(2,2,K)}[scheme]
    T = 40; n = 25; inp = header('grad',p,om) + f"{T}\n"; cases=[]
    for t in range(T):
        def rs(m): 
            v = rng.normal(size=(m,p)); return v/np.linalg.norm(v,axis=1,keepdims=True)
        x = rs(1)[0]; A = rs(n); B = rs(n); z = rs(1)[0]
        kap = rng.gamma(2,20,n) if t%2 else rng.gamma(1,1,n)+0.05
        y = rng.integers(0,2,n).astype(float); shape = float(rng.gamma(2,10))
        cases.append((x,A,B,kap,y,z,shape))
        inp += f"{n}\n" + '\n'.join(fmt(v) for v in (x,A,B,kap,y,z)) + f"\n{shape!r}\n"
    out = [l.split() for l in run(inp).strip().split('\n')]
    it = iter(out)
    for (x,A,B,kap,y,z,shape) in cases:
        r = {}
        for _ in range(8): l = next(it); r[l[0]] = np.array(l[1:],float)
        P = np.eye(p)-np.outer(x,x); omj = jnp.array(om)
        tests = [('prior+Jacobian', 'likeliC','gradient_pj', ref_logprior, (omj,))]   # likelihood blocks: see test_likelihood.py (needs a high-precision reference)
        for name,lk,gk,f,args in tests:
            val = float(f(jnp.array(x),*args)); gu = np.array(jax.grad(f)(jnp.array(x),*args)); gc = np.array(constrained(f)(jnp.array(x),*args))
            upd(f'{name}: |value diff|', abs(val-r[lk][0]))
            upd(f'{name}: rel err, code grad vs autodiff under paper constraint', np.linalg.norm(gc-r[gk])/np.linalg.norm(gc))
            upd(f'{name}: rel err, tangent projection vs unconstrained autodiff', np.linalg.norm(P@gu-P@r[gk])/np.linalg.norm(P@gu))
            upd(f'{name}: last component of code grad (should be 0)', abs(r[gk][-1]))
print("worst case over p in {3,4,5,6,8,11}, 3 omega schemes, 40 random points each:")
for k,v in worst.items(): print(f"  {k:75s} {v:.2e}")
