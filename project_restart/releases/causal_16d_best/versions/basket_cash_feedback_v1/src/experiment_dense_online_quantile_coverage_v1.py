"""Recalibrate forecast quantile masses with the observed ranks of already expired stock forecasts."""
import numpy as np
import torch
from scipy.signal import lfilter
import experiment_dense_reflected_quantile_curves_v1 as curves

onset,rel,REFERENCE = curves.onset,curves.rel,curves.REFERENCE
OUT = REFERENCE.parent/"dense_online_quantile_coverage_v1"
ARMS = ("recent_quantile_coverage", "risk_specific_coverage", "asymmetric_wing_coverage")
NOMINAL = np.diff(np.r_[0,curves.TAUS.numpy().astype(float),1])


def train(end):
    if not (curves.OUT/f"fit_{end}.pt").exists():
        curves.train(end)
    data = [curves.reliable.samples(d) for d in range(4,end)]
    tables = np.ones((12,50,3,2))
    slopes = []
    for h in (1,2):
        by_h = []
        for d,(x,_,grid) in enumerate(data,4):
            with np.load(curves.basket.OUT/f"day_{d:02d}.npz") as f:
                by_h.append((x,f["y"][...,h].astype(float),grid))
        tables[:,:,h] = curves.reliable.risk(by_h)
    for d in range(4,end):
        p = np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else onset.settled.head.ranks.band.pairs.net.future.forecast(d)
        q = p.astype(float)-p.mean((0,1),dtype=float)
        y = onset.settled.load(d,"y")
        y = y.astype(float)-y.mean((0,1),dtype=float)
        slopes.append((q*y).sum((0,1))/(q*q).sum((0,1)))
    np.savez(OUT/f"fit_{end}.npz",risk=tables,slope=np.median(slopes,axis=0).clip(.2,4))
    return np.array([end])


@torch.inference_mode()
def proposal(raw,grid,code,p,end):
    pack = torch.load(curves.OUT/f"fit_{end}.pt",map_location="cpu",weights_only=True)
    net = curves.Curves(len(pack["unit"])).double().eval()
    net.load_state_dict(pack["state"])
    x = curves.basket.features(raw,grid,code,p,pack["rate"].numpy())
    cell = curves.reliable.cells(x,grid)
    sd = pack["sd"].numpy()[cell,np.arange(50)]
    parts = []
    for t in np.array_split(np.arange(len(x)),max(1,(len(x)+95)//96)):
        u = (torch.from_numpy(x[t]).double()/pack["unit"]).clamp(-6,6)
        sym = .5*(net(u)-net(-u).flip((-1,)))
        parts.append(sym.numpy()*sd[t,:,:,None])
    return np.concatenate(parts),cell


def stable_market(err,pars,mask):
    beta,sd = pars[...,0],pars[...,1]
    loc = (err*mask).sum(1)/mask.sum(1).clip(1)
    for _ in range(3):
        dist = abs(err-beta*loc[:,None])/sd
        w = mask*np.minimum(1,1.3/dist.clip(.001))/sd**2
        loc = (w*beta*err).sum(1)/(w*beta**2).sum(1).clip(.001)
    return loc


def corrections(quant,obs,valid,group):
    # Eight actual price regions with seven ordered boundaries.
    bins = (obs[...,None]>quant).sum(-1)
    surprise = np.eye(8)[bins]-NOMINAL
    def update(use):
        obs_count = use.sum(1)
        delta = (surprise*use[...,None]).sum(1)
        # One sample every 5 ticks. The regularizer limits the influence of session openings.
        def decay(z):
            return lfilter([1.],[1.,-np.exp(-5/220)],z,axis=0)
        return decay(delta)/(150+decay(obs_count))[:,None]
    market = update(valid)
    within = np.stack([update(valid&(group==k)) for k in (0,1)],axis=1)
    return market,within


def design(raw,grid,code,p,bank):
    end = int(bank[0])
    with np.load(OUT/f"fit_{end}.npz") as f:
        risk,slope = f["risk"],f["slope"]
    with np.load(onset.settled.GEO) as f:
        k,coef,missing = float(f["k"]),f["coeff"],f["row"]
    bands,cell = proposal(raw,grid,code,p,end)
    out = np.zeros((*p.shape[:2],3,3))
    for a,b in rel.fields.parent.sessions(grid):
        n = b-a
        path = np.log1p(k*(raw[a:b,:,0].astype(float)-.00136)).cumsum(0)
        gaps = np.vstack((np.zeros(50,int),np.all(raw[a:b]==missing,axis=-1).cumsum(0)))
        for h,lag in enumerate((5,20,60)):
            stops = np.arange(lag,n,5)
            if not len(stops):
                continue
            orig = stops-lag
            ok = gaps[stops+1]==gaps[orig]
            realized = np.expm1(path[stops]-path[orig])*coef[h,1]/k
            err = realized/slope[h]-p[a+orig,:,h]
            pars = risk[cell[a+orig],np.arange(50),h]
            quant = bands[a:b,:,h]
            width = quant[...,-2]-quant[...,1]
            high = width>=np.median(width,axis=1,keepdims=True)
            ticks = np.arange(lag,n)
            lookup = (ticks-lag)//5
            for side in (0,1):
                src,dst = np.arange(50)%2==side,np.arange(50)%2!=side
                if h==0:
                    location = (err[:,dst]*ok[:,dst]).sum(1)/ok[:,dst].sum(1).clip(1)
                    centered = err[:,src]-location[:,None]
                else:
                    location = stable_market(err[:,dst],pars[:,dst],ok[:,dst])
                    centered = err[:,src]-location[:,None]*pars[:,src,0]
                avg,by_group = corrections(quant[orig][:,src],centered,ok[:,src],high[orig][:,src])
                # Price the observed coverage deviations with the receiving stock's current interval values.
                v = quant[ticks][:,dst]
                value = np.concatenate(((1.5*v[...,0]-.5*v[...,1])[...,None],.5*(v[...,1:]+v[...,:-1]),
                                       (1.5*v[...,-1]-.5*v[...,-2])[...,None]),axis=-1)
                direct = avg[lookup,None,:]
                local = by_group[lookup[:,None],high[ticks][:,dst].astype(int)]
                # Isolate imbalances between mirrored sides of the two outer intervals.
                asym = .5*(direct-direct[...,::-1])
                asym[:,:,2:6] = 0
                corr = np.stack(((value*direct).sum(-1),(value*(local-direct)).sum(-1),(value*asym).sum(-1)),axis=-1)
                out[(a+ticks)[:,None],np.flatnonzero(dst),:,h] = corr
    return out-out.mean(1,keepdims=True)


def predict(raw,grid,p,params):
    z = design(raw,grid,params["code"],p,params["bank"])
    q = p.copy()
    for h in range(3):
        q[...,h] += (rel.relative.features(p,z[...,h])@params["w"][:,h]).astype(q.dtype)
    return q


if __name__ == "__main__":
    torch.set_num_threads(1)
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
