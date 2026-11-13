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
