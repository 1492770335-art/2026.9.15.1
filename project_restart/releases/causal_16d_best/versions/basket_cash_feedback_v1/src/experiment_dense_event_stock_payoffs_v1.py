"""Estimate shared, stock-specific and partially pooled responses to unexpected price-event updates."""
import numpy as np
import torch
import experiment_dense_conditional_event_book_v1 as source
from prepare_dense_conditional_event_book_reference_v1 import OUT as REFERENCE

base,onset,rel = source.base,source.onset,source.rel
OUT = REFERENCE.parent/"dense_event_stock_payoffs_v1"
ARMS = ("shared_payoff","individual_payoff","partially_pooled_payoff")


def panels(raw,grid,p):
    for a,b,state,gaps,coef,k in source.states(raw,grid,p):
        marks,counts,ids = state
        yield a,b,(marks[...,2*base.DIM:],counts,ids),gaps


def features(t,state,p):
    marks,counts,ids = state
    num = counts[t,:,None]-np.arange(source.event.TAPS)
    stock = np.arange(50)[None,:,None]
    hist = marks[ids[num.clip(0),stock],stock]*(num>0)[...,None]
    z = np.concatenate((p[t],*(hist[:,:,lo:hi].mean(2) for lo,hi in source.spells.dual.BANDS)),axis=-1)
    return z-z.mean(1,keepdims=True)


def collect():
    OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for d in range(4,16):
        raw,grid,y = [onset.settled.load(d,key) for key in ("x","time","y")]
        p = (np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else
             onset.settled.head.ranks.band.pairs.net.future.forecast(d))
        gram = 0.
        n = 0
        for a,b,state,gaps in panels(raw,grid,p):
            times = np.arange(80+d%3,b-a-60,3)
            for i in range(0,len(times),64):
                t = times[i:i+64]
                ok = gaps[t+61]==gaps[t]
                x = features(t,state,p[a:b])
                yy = y[a+t, :, 2].astype(float)
                yy -= (yy*ok).sum(1,keepdims=True)/ok.sum(1,keepdims=True).clip(1)
                z = np.concatenate((x,yy[...,None]),axis=-1)*ok[...,None]
                u = torch.tensor(z.transpose(1,0,2),dtype=torch.float32,device=device)
                gram += (u.transpose(1,2)@u).cpu().numpy().astype(float)
                n += len(t)
        np.save(OUT/f"gram_{d:02d}.npy",(gram/n).astype(np.float32))


def parameters(rows):
    g = rows.mean((0,1))
    n = len(g)-1
    basis = np.eye(n)[:,3:]
    basis[:3] = -np.linalg.solve(g[:3,:3]+1e-7*np.eye(3),g[:3,3:n])
    cov = basis.T@g[:n,:n]@basis
    sd = cov.diagonal().clip(.0001)**.5
    val,u = np.linalg.eigh(cov/np.outer(sd,sd))
    axes = basis@(u/(sd[:,None]*np.sqrt(val.clip(0)+1)))
    # Remove the stock's own current forecast at the daily calibration slope before estimating returns.
    slope = (rows[:,:,:, -1][:,:,2]/rows[:,:,2,2].clip(.001)).clip(.25,4)
    grad = (rows[:,:,:n,-1]/slope[...,None]-rows[:,:,:n,2])@axes
    daily = grad.mean(1)
    mu = daily.mean(0)
    shared = mu*mu**2/(mu**2+daily.var(0,ddof=1)/len(rows)+1e-8)
    deviations = grad-daily[:,None]
    dev = deviations.mean(0)
    local = grad.mean(0)
    # Estimate the variance of persistent between-stock effects using independent-day cross products.
    left,right = deviations[::2].mean(0),deviations[1::2].mean(0)
    between = np.maximum((left*right).mean(0),0.)
    noise = deviations.var(0,ddof=1).mean(0)/len(rows)
    shrink = between/(between+noise+1e-8)
    return np.stack((np.broadcast_to(shared,dev.shape)@axes.T,local@axes.T,
                     (shared+shrink*dev)@axes.T),axis=-1),shrink


def train(end):
    rows = np.stack([np.load(OUT/f"gram_{d:02d}.npy") for d in range(4,end)]).astype(float)
    w,shrink = parameters(rows)
    np.savez(OUT/f"fit_{end}.npz",w=w,shrink=shrink)
    return np.array([end])


def design(raw,grid,code,p,bank):
    with np.load(OUT/f"fit_{int(bank[0])}.npz") as f:
        w = f["w"]
    out = np.zeros((*p.shape[:2],len(ARMS),3))
    for a,b,state,_ in panels(raw,grid,p):
        for start in range(0,b-a,64):
            t = np.arange(start,min(start+64,b-a))
            x = features(t,state,p[a:b])
            z = np.einsum("tsf,sfj->tsj",x,w,optimize=True)
            out[a+t,:,:,2] = z-z.mean(1,keepdims=True)
    return out


def predict(raw,grid,p,params):
    q = p.copy()
    z = design(raw,grid,params["code"],p,params["bank"])
    q[...,2] += (rel.relative.features(p,z[...,2])@params["w"][:,2]).astype(q.dtype)
    return q


if __name__=="__main__":
    torch.set_num_threads(1)
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
