"""Calibrate horizons that cross the scheduled midday break and model the observed reopening jump."""
import numpy as np
import experiment_dense_path_skill_calibration_v1 as path
from prepare_dense_path_skill_reference_v1 import OUT as REFERENCE

onset,rel = path.onset,path.rel
OUT = REFERENCE.parent/"dense_midday_expiry_v1"
ARMS = ("forecast_beyond_break","forecast_opening_jump","observed_opening_surprise")
H = (5,20,60)
OPEN = 2401


def center(x):
    return x-x.mean(1,keepdims=True)


def inputs(raw,grid,p):
    with np.load(onset.settled.GEO) as f:
        k,coef,missing = float(f["k"]),f["coeff"],f["row"]
    obs = ~np.all(raw==missing,axis=-1)
    step = np.log1p(k*(raw[...,0].astype(float)-.00136))*coef[2,1]/k*obs
    level = np.cumsum(step,axis=0)
    t = np.flatnonzero((grid>=OPEN-60)&(grid<OPEN))
    cov = [np.ones((len(t),50))]
    for lag in (5,20,60,240,1200):
        delta = level[t]-level[(t-lag).clip(0)]
        cov.append(np.tanh(center(delta)/(.10*np.sqrt(lag))))
    cov.extend(np.tanh(center(p[t,:,j].astype(float))*2) for j in range(3))
    x = np.stack(cov,axis=-1)
    value = np.log1p(k*(p.astype(float)-coef[:,0])/coef[:,1])*coef[2,1]/k
    return t,x,step,value,obs,coef


def collect():
    OUT.mkdir(exist_ok=True)
    for d in range(4,16):
        raw,grid = [onset.settled.load(d,k) for k in ("x","time")]
        p = (np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else
             onset.settled.head.ranks.band.pairs.net.future.forecast(d))
        t,x,step,_,obs,_ = inputs(raw,grid,p)
        oi = np.flatnonzero(grid==OPEN)[0]
        ok = obs[oi]&obs[oi-1]
        tgt = step[oi]-(step[oi]*ok).sum()/ok.sum().clip(1)
        # Each stock/date has one reopening jump; its 60 forecast origins share the same total weight.
        valid = obs[t]&ok[None]
        weight = valid/valid.sum(0,keepdims=True).clip(1)
        z = np.concatenate((x,np.broadcast_to(tgt.clip(-3,3)[None,:,None],(*x.shape[:2],1))),axis=2)
        flat = (z*np.sqrt(weight[...,None])).reshape(-1,z.shape[-1])
        np.save(OUT/f"gram_{d:02d}.npy",flat.T@flat/50)


def train(end):
    rows = np.stack([np.load(OUT/f"gram_{d:02d}.npy") for d in range(4,end)])
    g = rows.mean(0)
    sd = np.sqrt(g.diagonal()[:-1]).clip(.03)
    grad = rows[:,:-1,-1]/sd
    mu = grad.mean(0)
    avg = mu*mu**2/(mu**2+grad.var(0,ddof=1)/len(grad)+1e-6)
    return np.linalg.solve(g[:-1,:-1]/np.outer(sd,sd)+np.eye(len(sd)),avg)/sd


def design(raw,grid,code,p,bank):
    t,x,step,value,obs,coef = inputs(raw,grid,p)
    z = np.zeros((*p.shape[:2],len(ARMS),3))
    left = OPEN-1-grid[t]
    edge = np.flatnonzero(grid==OPEN)
    for j,h in enumerate(H):
        active = left<h
        idx = t[active]
        remain = left[active]
        if len(idx):
            # Linearly interpolate the return expected to have accrued before the break.
            v = np.concatenate((np.zeros_like(value[idx,:,:1]),value[idx]),axis=-1)
            knots = np.array((0,*H))
            hi = np.searchsorted(knots,remain).clip(1)
            lo = hi-1
            mix = ((remain-knots[lo])/(knots[hi]-knots[lo]))[:,None]
            a = np.arange(len(idx))
            accrued = (1-mix)*v[a,:,lo]+mix*v[a,:,hi]
            z[idx,:,0,j] = center(value[idx,:,j]-accrued)*coef[j,1]/coef[2,1]
            z[idx,:,1,j] = center(x[active]@bank)*coef[j,1]/coef[2,1]
        if len(edge) and edge[0]>0 and grid[edge[0]-1]==OPEN-1:
            oi = edge[0]
            valid = obs[oi]&obs[oi-1]
            surprise = (step[oi]-value[oi-1,:,0]/5)*valid
            surprise = (surprise-(surprise*valid).sum()/valid.sum().clip(1))*valid
            later = np.flatnonzero((grid>=OPEN)&(grid<OPEN+3*h))
            z[later,:,2,j] = np.exp(-(grid[later,None]-OPEN)/h)*surprise.clip(-3,3)*coef[j,1]/coef[2,1]
    return z


def predict(raw,grid,p,params):
    return path.corrected(p,design(raw,grid,params["code"],p,params["bank"]),params)


if __name__=="__main__":
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
