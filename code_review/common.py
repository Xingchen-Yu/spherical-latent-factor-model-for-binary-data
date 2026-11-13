import numpy as np, subprocess
def jacobian_init(p):            # port of the R function
    k = np.ones((2,1))
    for i in range(4,p+1): k = np.hstack([np.vstack([k,np.zeros((1,k.shape[1]))]), np.ones((k.shape[0]+1,1))])
    return k
def jacobian_init2(p):
    k = np.ones((3,1))
    for i in range(5,p+1): k = np.hstack([np.vstack([k,np.zeros((1,k.shape[1]))]), np.ones((k.shape[0]+1,1))])
    return k
def getcord_auto(a):             # port of the R function: angles (n,K) -> x (n,K+1)
    sup = np.ones((a.shape[0],1))
    for i in range(a.shape[1]): sup = np.hstack([sup*np.sin(a[:,[i]]), np.cos(a[:,[i]])])
    return sup
def fmt(a): return ' '.join(repr(float(v)) for v in np.asarray(a).ravel())
def header(mode,p,omega):
    s = f"{mode} {p}\n"
    if p>=4:
        for m in (jacobian_init(p), jacobian_init2(p)): s += f"{m.shape[0]} {m.shape[1]}\n{fmt(m)}\n"
    return s + fmt(omega) + "\n"
def run(inp): return subprocess.run(['./harness'],input=inp,capture_output=True,text=True,check=True).stdout
