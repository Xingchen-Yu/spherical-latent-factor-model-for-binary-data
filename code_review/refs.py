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

