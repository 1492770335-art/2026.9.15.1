"""Price innovations at distinct price changes and integrate them over a fixed-tick forecasting horizon."""
from functools import lru_cache
import numpy as np
from scipy.signal import lfilter
from scipy.stats import binom
import torch
import experiment_dense_book_impulse_horizons_v1 as base

onset,rel,REFERENCE = base.onset,base.rel,base.REFERENCE
OUT = REFERENCE.parent/"dense_event_impulse_horizons_v1"
ARMS = ("direct_event_history","usual_event_count","live_event_count","count_revision")
TAPS,DIM = 24,base.DIM
RATES = np.linspace(0,1,41)


@lru_cache(maxsize=1)
def prior():
    observed = []
    for d in range(4):
        x = onset.settled.load(d,"x")
        observed.append((abs(x[...,0].astype(float)-.00136)>.001).mean(0))
    return np.mean(observed,axis=0)


def clock(raw):
    move = abs(raw[...,0].astype(float)-.00136)>.001
    count = move.cumsum(0)
    ids = np.zeros((len(raw)+2,50),dtype=np.int32)
    t,stock = np.where(move)
    ids[count[t,stock],stock] = t
    alpha = 1-np.exp(-1/120)
    rate = prior()+lfilter([alpha],[1,alpha-1],move.astype(float)-prior(),axis=0)
    return count,ids,np.clip(.8*rate+.2*prior(),0,1)


def inputs(e,t,count,ids,p):
    num = count[t,:,None]-np.arange(TAPS)
    stock = np.arange(50)[None,:,None]
    times = ids[num.clip(0),stock]
    x = e[times,stock]*(num>0)[...,None]
    return np.concatenate((p[t],x.reshape(len(t),50,-1)),axis=-1)


def collect():
    OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with np.load(onset.settled.GEO) as f:
        k,coef,missing = float(f["k"]),f["coeff"],f["row"]
    for d in range(4,16):
        raw,grid,y = [onset.settled.load(d,key) for key in ("x","time","y")]
        p = (np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else
             onset.settled.head.ranks.band.pairs.net.future.forecast(d))
        gram = 0.
        n = 0
        for a,b in rel.fields.parent.sessions(grid):
            section = raw[a:b]
            e = base.innovations(section,device)
            count,ids,_ = clock(section)
            gaps = np.vstack((np.zeros(50,int),np.all(section==missing,axis=-1).cumsum(0)))
            cash = np.log1p(k*(section[...,0].astype(float)-.00136))*coef[2,1]/k
            times = np.arange(80,b-a-60)
            ix = np.arange(len(times))
            sample = times[ix%8==(ix//24+d)%8]
            for start in range(0,len(sample),64):
                t = sample[start:start+64]
                stock = np.arange(50)[None,:]
                due = ids[count[t]+1,stock]
                valid = (due>t[:,None])&(gaps[t+61]==gaps[t])&(gaps[due+1,stock]==gaps[t])
                features = inputs(e,t,count,ids,p[a:b])
                labels = base.target(y[a:b],t,coef,k)
                labels[...,0] = cash[due,stock]
                z = np.concatenate((features,labels),axis=-1)
                z -= (z*valid[...,None]).sum(1,keepdims=True)/valid.sum(1)[:,None,None].clip(1)
                flat = torch.as_tensor(z[valid],device=device,dtype=torch.float32)
                gram += (flat.T@flat).cpu().numpy().astype(float)
                n += len(flat)
        np.save(OUT/f"gram_{d:02d}.npy",gram/n)


def response(days):
    g = days.mean(0)
    n = len(g)-4
    v = np.eye(n)[:,3:]
    ctrl = g[:3,:3]
    v[:3] = -np.linalg.solve(ctrl+.01*np.diag(ctrl.diagonal()),g[:3,3:n])
    cov = v.T@g[:n,:n]@v
    sd = np.sqrt(cov.diagonal().clip(.0001))
    ev,rot = np.linalg.eigh(cov/np.outer(sd,sd))
    axes = v@(rot/(sd[:,None]*np.sqrt(ev.clip(0)+1.)))
    grad = axes.T@days[:,:n,n:]
    avg = grad.mean(0)
    w = axes@(avg*avg**2/(avg**2+grad.var(0,ddof=1)/len(grad)+1e-8))
    kernel = w[3:,0].reshape(TAPS,DIM)
    return w,v,kernel


def train(end):
    days = np.stack([np.load(OUT/f"gram_{d:02d}.npy") for d in range(4,end)])
    w,v,kernel = response(days)
    bank = np.zeros((len(RATES),TAPS,DIM))
    for j in range(TAPS):
        # A past innovation contributes to the (j+1)th future event only if that event arrives within 60 ticks.
        prob = binom.sf(j,60,RATES)
        bank[:,:TAPS-j] += prob[:,None,None]*kernel[None,j:]
    integrated = (v@bank.reshape(len(RATES),-1).T).T
    np.savez(OUT/f"fit_{end}.npz",direct=w[:,1:].sum(1),bank=integrated)
    return np.array([end])


def design(raw,grid,code,p,bank):
    with np.load(OUT/f"fit_{int(bank[0])}.npz") as f:
        w,by_rate = f["direct"],f["bank"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.zeros((*p.shape[:2],len(ARMS),3))
    static = np.rint(prior()*40).astype(int)
    for a,b in rel.fields.parent.sessions(grid):
        e = base.innovations(raw[a:b],device)
        count,ids,rates = clock(raw[a:b])
        for start in range(0,b-a,64):
            t = np.arange(start,min(start+64,b-a))
            x = inputs(e,t,count,ids,p[a:b]).astype(float)
            usual = np.einsum("tsf,sf->ts",x,by_rate[static])
            dynamic = np.einsum("tsf,tsf->ts",x,by_rate[np.rint(rates[t]*40).astype(int)])
            pred = np.stack((x@w,usual,dynamic,dynamic-usual),axis=-1)
            z[a+t,:,:,2] = pred-pred.mean(1,keepdims=True)
    return z


def predict(raw,grid,p,params):
    q = p.copy()
    z = design(raw,grid,params["code"],p,params["bank"])
    q[...,2] += (rel.relative.features(p,z[...,2])@params["w"][:,2]).astype(q.dtype)
    return q


if __name__=="__main__":
    torch.set_num_threads(1)
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
