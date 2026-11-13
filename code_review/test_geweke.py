import numpy as np, subprocess, sys
from scipy import special
from common import *
from sampler_fns import to_angles
import os; STEP=os.environ.get("STEP","0.05 0.25 0.05 0.25")
hyp = dict(a=6.0,b=2.0,a_tau=6.0,b_tau=2.0,kappa_a=2.0,ccc_a=6.0,ccc_b=40.0)
def prior_draw(rng,n,p):
    K=p-1; k2=np.arange(1,K+1)**2
    om=rng.gamma(hyp['a'],1/hyp['b'],n); tau=rng.gamma(hyp['a_tau'],1/hyp['b_tau'],n); lam=rng.gamma(hyp['ccc_a'],1/hyp['ccc_b'],n); kap=rng.gamma(hyp['kappa_a'],1/lam)
    ang=lambda w: np.column_stack([rng.vonmises(0,w)+np.pi]+[(rng.vonmises(0,w*c)+np.pi)/2 for c in k2[1:]])
    return om,tau,lam,kap,ang(om),ang(tau)
def stats_of(om,tau,lam,kap,ab,ay,an):
    cols={'omega':om,'log omega':np.log(om),'tau':tau,'tau^2':tau**2,'lambda':lam,'log kappa_1':np.log(kap),'sqrt kappa_1':np.sqrt(kap)}
    for nm,a in (('beta_1',ab),('psi_1',ay),('zeta_J',an)):
        cols[f'{nm}: cos a1']=np.cos(a[:,0]); cols[f'{nm}: sin a1']=np.sin(a[:,0]); cols[f'{nm}: cos 2a1']=np.cos(2*a[:,0])
        for k in range(1,a.shape[1]): cols[f'{nm}: cos 2a{k+1}']=np.cos(2*a[:,k]); cols[f'{nm}: cos a{k+1}']=np.cos(a[:,k])
    return cols
def run_one(p,I,J,sweeps,seed):
    rng=np.random.default_rng(seed); om,tau,lam,kap,ab,ay=prior_draw(rng,max(I,J),p)
    _,_,_,_,_,an=prior_draw(rng,J,p)   # independent start for zeta
    inp=f"{p} {I} {J} {sweeps} 4 {seed}\n"+' '.join(str(hyp[k]) for k in ['a','b','a_tau','b_tau','kappa_a','ccc_a','ccc_b'])+f" {STEP}\n"
    if p>=4:
        for m in (jacobian_init(p),jacobian_init2(p)): inp+=f"{m.shape[0]} {m.shape[1]}\n{fmt(m)}\n"
    inp+='\n'.join(fmt(v) for v in (getcord_auto(ab[:I]),getcord_auto(ay[:J]),getcord_auto(an[:J]),kap[:J]))+f"\n{om[0]} {tau[0]} {lam[0]}\n"
    r=subprocess.run(['./geweke'],input=inp,capture_output=True,text=True,check=True)
    D=np.loadtxt(r.stdout.splitlines()); D=D[len(D)//20:]
    ch=stats_of(D[:,0],D[:,1],D[:,2],D[:,3],to_angles(D[:,4:4+p]),to_angles(D[:,4+p:4+2*p]),to_angles(D[:,4+2*p:4+3*p]))
    rng2=np.random.default_rng(seed+1); n=3000000; om,tau,lam,kap,ab,ay=prior_draw(rng2,n,p); _,_,_,_,_,an=prior_draw(rng2,n,p)
    pr=stats_of(om,tau,lam,kap,ab,ay,an)
    zs={}
    for k in ch:
        v=ch[k]; nb=60; bmeans=v[:len(v)//nb*nb].reshape(nb,-1).mean(1); se=np.sqrt(bmeans.var(ddof=1)/nb+pr[k].var()/n)
        zs[k]=(v.mean()-pr[k].mean())/se
    return r.stderr.strip(), zs, ch, pr
if __name__=='__main__':
    p,I,J,sweeps,seed=[int(v) for v in sys.argv[1:6]]
    acc,zs,ch,pr=run_one(p,I,J,sweeps,seed)
    z=np.array(list(zs.values())); worst=max(zs,key=lambda k:abs(zs[k]))
    print(f"p={p} (K={p-1}) I={I} J={J} sweeps={sweeps}: {len(z)} statistics, mean z={z.mean():+.2f}, sd z={z.std():.2f}, max|z|={abs(z).max():.2f} ({worst}); |z|>3: {int((abs(z)>3).sum())}\n   {acc}")
    for k in ['omega','tau','lambda','log kappa_1','beta_1: cos a1','beta_1: cos 2a2','psi_1: cos a1','psi_1: cos 2a2',f'beta_1: cos 2a{p-1}',f'zeta_J: cos 2a{p-1}']:
        print(f"      {k:22s} sampler mean {ch[k].mean():+.4f}   prior mean {pr[k].mean():+.4f}   z={zs[k]:+.2f}")
