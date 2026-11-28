"""Forecast second moments of independent factor errors and price them on the other stocks."""
from functools import lru_cache

import numpy as np
from scipy.signal import lfilter

import experiment_dense_refresh_payoff_instruments_v1 as own
from prepare_dense_stock_specific_refresh_reference_v1 import OUT as REFERENCE

onset,rel,book=own.onset,own.rel,own.book
OUT=REFERENCE.parent/"dense_factor_error_covariance_v1"
ARMS=("factor_error_variance","factor_error_covariance","all_second_moments")
H=(5,20,60)
K=4
I,J=np.triu_indices(K)
M=len(I)
C=1+K
W=C+3*M
GROUPS=np.sort(np.random.default_rng(4909).permutation(50).reshape(2,25),axis=1)


def learn_factors():
    weights=[];scales=[]
    for ids in GROUPS:
        cov=0.
        for d in range(4):
            y=onset.settled.load(d,"y")[::13,ids,1].astype(float)
            cov+=np.cov(y.T)/4
        market=np.ones(25)/5
        perp=np.eye(25)-np.outer(market,market)
        _,vec=np.linalg.eigh(perp@cov@perp)
        w=np.column_stack((market,vec[:,-3:]))
        weights.append(w)
        grams=[]
        for d in range(4):
            y=onset.settled.load(d,"y")[::13,ids].astype(float)
            projected=np.einsum("tsh,sk->thk",y,w)
            grams.append((projected**2).mean(0))
        scales.append(np.sqrt(np.mean(grams,axis=0)).clip(.1))
    np.savez(OUT/"factors.npz",weights=np.stack(weights),scales=np.stack(scales))
    factors.cache_clear()


@lru_cache(maxsize=1)
def factors():
    with np.load(OUT/"factors.npz") as f:return f["weights"],f["scales"]


def products(v):
    return (v[...,I]*v[...,J]).clip(-12,12)


def ema(x,span):
    alpha=1-np.exp(-1/span)
    return lfilter([alpha],[1,alpha-1],x,axis=0)


def panels(raw,grid,p,y=None):
    w,s=factors()
    with np.load(onset.settled.GEO) as f:
        k,co,missing=float(f["k"]),f["coeff"],f["row"]
    for a,b in rel.fields.parent.sessions(grid):
        data=raw[a:b]
        level=np.log1p(k*(data[:,:,0].astype(float)-.00136)).cumsum(0)
        gaps=np.vstack((np.zeros(50,int),np.all(data==missing,axis=2).cumsum(0)))
        for g,ids in enumerate(GROUPS):
            receivers=GROUPS[1-g]
            for j,h in enumerate(H):
                q=(p[a:b][:,ids,j]@w[g])/s[g,j]
                t=np.arange(h,b-a)
                mature=np.zeros((b-a,K))
                if len(t):
                    valid=gaps[t+1][:,ids]==gaps[t-h][:,ids]
                    move=np.expm1(level[t][:,ids]-level[t-h][:,ids])*co[j,1]/k+co[j,0]
                    error=(move-p[a+t-h][:,ids,j])*valid
                    mature[t]=error@w[g]/s[g,j]
                zz=products(mature)
                fast=ema(zz,max(48,2*h));slow=ema(zz,480)
                x=np.column_stack((np.ones(len(q)),q,products(q),fast,slow))
                future=None;err=None
                if y is not None:
                    outcome=y[a:b][:,ids,j]@w[g]/s[g,j]
                    diff=outcome-q
                    future=np.column_stack((diff,products(diff)))
                    outcome_y=y[a:b,:,j]
                    pred=p[a:b,:,j]
                    sy=outcome_y-outcome_y.mean(1,keepdims=True)
                    sp=pred-pred.mean(1,keepdims=True)
                    slope=((sy*sp).sum()/(sp*sp).sum()).clip(.3,3)
                    err=(sy/slope-sp)[:,receivers]
                yield a,b,g,j,receivers,x,future,err


def collect():
    OUT.mkdir(exist_ok=True)
    learn_factors()
    nall=W+K+M+25
    for d in range(4,16):
        raw,grid,y=[onset.settled.load(d,key) for key in ("x","time","y")]
        p=(np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else
           onset.settled.head.ranks.band.pairs.net.future.forecast(d))
        grams=np.zeros((2,3,nall,nall))
        counts=np.zeros((2,3))
        for a,b,g,j,ids,x,f,e in panels(raw,grid,p,y):
            t=np.arange(d%5,b-a-H[j],5)
            rows=np.column_stack((x[t],f[t],e[t]))
            grams[g,j]+=rows.T@rows;counts[g,j]+=len(rows)
        np.save(OUT/f"gram_{d:02d}.npy",grams/counts[:,:,None,None])


def train(end):
    days=np.stack([np.load(OUT/f"gram_{d:02d}.npy") for d in range(4,end)])
    bank=np.zeros((2,3,W,3,25))
    for g in range(2):
        for j in range(3):
            for parity in (0,1):
                dyn,pay=[days[k::2,g,j].mean(0) for k in (parity,1-parity)]
                # Target residuals are orthogonalized against realized linear factors and known forecasts.
                for arm,idx in enumerate((np.flatnonzero(I==J),np.flatnonzero(I!=J),np.arange(M))):
                    target=W+K+idx
                    ctrl=np.r_[np.arange(C),np.arange(W,W+K)]
                    cc=pay[np.ix_(ctrl,ctrl)]
                    beta=own.solve(cc,pay[np.ix_(ctrl,target)],.02)
                    tr=np.eye(len(pay))[:,target]
                    tr[ctrl]-=beta
                    # Remove effects of the mean and predicted factor direction at inference.
                    feat=np.eye(W)[:,C:]
                    feat[:C]=-own.solve(dyn[:C,:C],dyn[:C,C:W],.03)
                    cov=feat.T@dyn[:W,:W]@feat
                    bridge=own.solve(cov,feat.T@dyn[:W]@tr,.9)
                    price=own.solve(tr.T@pay@tr,tr.T@pay[:,-25:],.8)
                    bank[g,j,:,arm]+=.5*feat@bridge@price
    return bank


def design(raw,grid,code,p,bank):
    out=np.zeros((*p.shape[:2],3,3))
    for a,b,g,j,ids,x,_,_ in panels(raw,grid,p):
        z=(x@bank[g,j].reshape(W,-1)).reshape(b-a,3,25).transpose(0,2,1)
        out[a:b,:,:,j][:,ids,:]=z
    return out-out.mean(1,keepdims=True)


def predict(raw,grid,p,params):
    return book.path.corrected(p,design(raw,grid,params["code"],p,params["bank"]),params)


if __name__=="__main__":
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
