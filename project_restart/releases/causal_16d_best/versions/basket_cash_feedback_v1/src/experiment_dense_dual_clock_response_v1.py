"""Estimate a block's remaining payoff from both the number of price events and their actual elapsed ages."""
import numpy as np
import torch
import experiment_dense_event_impulse_horizons_v1 as event

base,onset,rel,REFERENCE = event.base,event.onset,event.rel,event.REFERENCE
OUT = REFERENCE.parent/"dense_dual_clock_response_v1"
ARMS = ("event_cohorts","unexpected_elapsed_time","elapsed_after_activity_adjustment")
BANDS = ((0,1),(1,3),(3,7),(7,15),(15,24))
WIDTH = len(BANDS)*base.DIM


def features(e,t,count,ids,rate,p):
    lags = np.arange(event.TAPS)
    num = count[t,:,None]-lags
    stock = np.arange(50)[None,:,None]
    occurred = ids[num.clip(0),stock]
    marks = e[occurred,stock]*(num>0)[...,None]
    elapsed = (t[:,None,None]-occurred).clip(0)
    static = event.prior()[None,:,None]
    live = rate[t,:,None].clip(.01,1)
    # Under Bernoulli arrivals the age of the most recent event may be zero.
    # This statistic also distinguishes bursts and long pauses at the same event count.
    def duration(prob):
        return np.tanh(np.log((1+elapsed)*prob/(lags+1)).clip(-4,4))
    weights = (np.ones_like(elapsed),duration(static),duration(live))
    sections = []
    for gate in weights:
        sections.append(np.concatenate([(marks[:,:,lo:hi]*gate[:,:,lo:hi,None]).mean(2) for lo,hi in BANDS],axis=-1))
    return np.concatenate((p[t],*sections),axis=-1)


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
            e = base.innovations(raw[a:b],device)
            count,ids,rate = event.clock(raw[a:b])
            gaps = np.vstack((np.zeros(50,int),np.all(raw[a:b]==missing,axis=-1).cumsum(0)))
            ticks = np.arange(80,b-a-60)
            ix = np.arange(len(ticks))
            sample = ticks[ix%8==(ix//24+d)%8]
            for start in range(0,len(sample),48):
                t = sample[start:start+48]
                ok = gaps[t+61]==gaps[t]
                x = features(e,t,count,ids,rate,p[a:b])
                z = np.concatenate((x,base.target(y[a:b],t,coef,k)[...,1:]),axis=-1)
                z -= (z*ok[...,None]).sum(1,keepdims=True)/ok.sum(1)[:,None,None].clip(1)
                flat = torch.as_tensor(z[ok],device=device,dtype=torch.float32)
                gram += (flat.T@flat).cpu().numpy().astype(float)
                n += len(flat)
        np.save(OUT/f"gram_{d:02d}.npy",gram/n)


def train(end,directory=OUT):
    rows = np.stack([np.load(directory/f"gram_{d:02d}.npy") for d in range(4,end)])
    g = rows.mean(0)
    n = len(g)-3
    heads = []
    for j in range(3):
        left = 3+j*WIDTH
        v = np.eye(n)[:,left:left+WIDTH]
        controls = np.arange(3 if not j else 3+WIDTH)
        cm = g[np.ix_(controls,controls)]
        v[controls] = -np.linalg.solve(cm+.16*np.diag(cm.diagonal().clip(.0001)),g[controls,left:left+WIDTH])
        cov = v.T@g[:n,:n]@v
        sd = np.sqrt(cov.diagonal().clip(.0001))
        eig,vec = np.linalg.eigh(cov/np.outer(sd,sd))
        axis = v@(vec/(sd[:,None]*np.sqrt(eig.clip(0)+1.)))
        grad = axis.T@rows[:,:n,n:]
        avg = grad.mean(0)
        heads.append(axis@(avg*avg**2/(avg**2+grad.var(0,ddof=1)/len(grad)+1e-8)))
    bank = np.stack(heads,axis=1)
    np.savez(directory/f"fit_{end}.npz",bank=bank)
    return np.array([end])


def design(raw,grid,code,p,bank):
    with np.load(OUT/f"fit_{int(bank[0])}.npz") as f:
        heads = f["bank"].sum(-1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.zeros((*p.shape[:2],len(ARMS),3))
    for a,b in rel.fields.parent.sessions(grid):
        e = base.innovations(raw[a:b],device)
        count,ids,rate = event.clock(raw[a:b])
        for start in range(0,b-a,48):
            t = np.arange(start,min(start+48,b-a))
            x = features(e,t,count,ids,rate,p[a:b])@heads
            z[a+t,:,:,2] = x-x.mean(1,keepdims=True)
    return z


def predict(raw,grid,p,params):
    out = p.copy()
    z = design(raw,grid,params["code"],p,params["bank"])
    out[...,2] += (rel.relative.features(p,z[...,2])@params["w"][:,2]).astype(out.dtype)
    return out


if __name__=="__main__":
    torch.set_num_threads(1)
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
