"""Estimate a distributed-lag price response to unexpected pressure reversals and subtract the response already forecast."""
import json
import numpy as np

import experiment_dense_pressure_survival_clock_v1 as clock
from prepare_dense_pressure_clock_reference_v1 import OUT as REFERENCE

onset, rel = clock.onset, clock.rel
OUT = REFERENCE.parent / "dense_switch_impulse_curve_v1"
ARMS = ("projected_price_response","late_price_response","response_beyond_forecast")
EDGES = (0,1,5,13,25,45,80,120)
HORIZONS = (5,20,60)


def shocks(v,s,start,visible,rates):
    _,key = clock.keys(v,s,start)
    chance = rates[key[:-1],0]
    event = s[1:]!=s[:-1]
    valid = visible[:-1,:,None]&visible[1:,:,None]&(start[:-1]>0)
    pulse = -s[:-1]*(event-chance)*valid
    # Treat a rare switch differently from one that would often occur in this state.
    scaled = pulse*np.tanh(-.5*np.log(chance.clip(.001)) - 1)
    x = np.concatenate((pulse,scaled),axis=2)
    return np.concatenate((np.zeros_like(x[:1]),x),axis=0) if len(x) else np.zeros((*s.shape[:2],10))


def lagged(x,t):
    sums = np.concatenate((np.zeros_like(x[:1]),x.cumsum(0)))
    return np.concatenate([(sums[(t+1-lo).clip(0)]-sums[(t+1-hi).clip(0)])/np.sqrt(hi-lo)
                           for lo,hi in zip(EDGES,EDGES[1:])],axis=2)


def collect():
    OUT.mkdir(exist_ok=True)
    code = rel.fields.parent.codec()
    with np.load(onset.settled.GEO) as f:
        k,coef,missing = float(f["k"]),f["coeff"][:,1],f["row"]
    for d in range(4,16):
        raw,grid,y = [onset.settled.load(d,n) for n in ("x","time","y")]
        p = (np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else onset.settled.head.ranks.band.pairs.net.future.forecast(d))
        q = p-p.mean(1,keepdims=True,dtype=float)
        actual = y-y.mean(1,keepdims=True,dtype=float)
        beta = ((q*actual).sum((0,1))/(q*q).sum((0,1))).clip(.2,4)
        q *= beta*coef[2]/coef
        rates = clock.rates(d)
        gram = cross = n = 0
        audit = np.zeros((3,5,2))
        for a,b,v,s,start,_ in onset.sessions(raw,grid,code):
            data = raw[a:b]
            visible = ~np.all(data==missing,axis=2)
            r = np.log1p(k*(data[:,:,0].astype(float)-.00136))*coef[2]/k
            r -= (r*visible).sum(1,keepdims=True)/visible.sum(1,keepdims=True).clip(1)
            r *= visible
            x = shocks(v,s,start,visible,rates)
            t = np.arange(30+d%3,b-a,3)
            lag = lagged(x,t)
            lag -= lag.mean(1,keepdims=True)
            xx = lag[visible[t]]
            resp = np.concatenate((r[t,:,None],q[a+t]),axis=2)[visible[t]]
            gram += xx.T@xx
            cross += xx.T@resp
            n += len(xx)
            cs = r.cumsum(0)
            gaps = np.vstack((np.zeros((1,50),int),(~visible).cumsum(0)))
            for j,h in enumerate(HORIZONS):
                ix = t[t+h<len(r)]
                good = gaps[ix+h+1]==gaps[ix]
                marked = x[ix,:,:5]*good[...,None]
                audit[j,:,0] += (marked*(cs[ix+h]-cs[ix])[...,None]).sum((0,1))
                audit[j,:,1] += (marked**2).sum((0,1))
        np.savez(OUT/f"day_{d:02d}.npz",g=gram/n,r=cross/n,audit=audit)


def train(end):
    gm,rr = [],[]
    for d in range(4,end):
        with np.load(OUT/f"day_{d:02d}.npz") as data:
            gm.append(data["g"]); rr.append(data["r"])
    g = np.mean(gm,0)
    unit = np.sqrt(g.diagonal()).clip(.01)
    eig,vec = np.linalg.eigh(g/np.outer(unit,unit))
    basis = vec/(unit[:,None]*np.sqrt(eig.clip(0)+.6))
    r = basis.T@np.stack(rr)
    mean = r.mean(0)
    w = basis@(mean*mean**2/(mean**2+r.var(0,ddof=1)/len(r)+1e-10))
    w = w.reshape(len(EDGES)-1,10,4)
    kernel = np.concatenate([np.repeat(w[j:j+1]/np.sqrt(hi-lo),hi-lo,axis=0)
                             for j,(lo,hi) in enumerate(zip(EDGES,EDGES[1:]))])
    # Forecast only returns AFTER the current tick. The same-tick impact has already happened.
    sums = np.concatenate((np.zeros_like(kernel[:1,:,0]),kernel[:,:,0].cumsum(0)),axis=0)
    t = np.arange(len(kernel))+1
    banks = []
    for j,h in enumerate(HORIZONS):
        total = sums[(t+h).clip(max=len(kernel))]-sums[t]
        late = sums[(t+h).clip(max=len(kernel))]-sums[(t+h//3).clip(max=len(kernel))]
        banks.append(np.stack((total,late,total-kernel[:,:,j+1]),axis=-1))
    kernel = np.stack(banks,axis=-1)
    np.savez(OUT/f"fit_{end}.npz",kernel=kernel,rate=clock.rates(end))
    return np.array([end])


def design(raw,grid,code,p,bank):
    with np.load(OUT/f"fit_{int(bank[0])}.npz") as f:
        kernel,rates = f["kernel"],f["rate"]
    with np.load(onset.settled.GEO) as f:
        missing = f["row"]
    from scipy.signal import fftconvolve
    out = np.empty((*p.shape[:2],3,3))
    for a,b,v,s,start,_ in onset.sessions(raw,grid,code):
        visible = ~np.all(raw[a:b]==missing,axis=2)
        x = shocks(v,s,start,visible,rates)
        # Truncate the causal convolution at the last observed tick.
        z = np.zeros((b-a,50,3,3))
        for h in range(3):
            for j in range(3):
                for f in range(x.shape[2]):
                    z[:,:,j,h] += fftconvolve(x[:,:,f],kernel[:,f,j,h,None],axes=0)[:len(x)]
        out[a:b] = z-z.mean(1,keepdims=True)
    return out


def predict(raw,grid,p,params):
    z = design(raw,grid,params["code"],p,params["bank"])
    q = p.copy()
    for h in range(3):
        q[:,:,h] += (rel.relative.features(p,z[...,h])@params["w"][:,h]).astype(q.dtype)
    return q


if __name__=="__main__":
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
