"""Estimate market-wide demand from the diffusion of newly switched pressure across disjoint stock baskets."""
import json
import numpy as np
from scipy.signal import lfilter

import experiment_dense_supervised_switch_direction_v1 as direction
from experiment_dense_shared_horizon_calibration_v1 import score

onset,clock,rel,REFERENCE=direction.onset,direction.clock,direction.rel,direction.REFERENCE
OUT=REFERENCE.parent/"dense_market_switch_front_v1"
ARMS=("expected_participation","unexpected_participation","independent_basket_confirmation")


def features(raw,grid,code,p,rate):
    blocks=[]
    for a,b,v,s,start,_ in onset.sessions(raw,grid,code):
        _,keys=clock.keys(v,s,start)
        pr=rate[keys]
        # Conditional probability that a name keeps its present sign up to each horizon.
        survivor=1-pr[...,:3].cumsum(-1)
        valid=(start>0)&(s!=0)
        t=np.arange(b-a)[:,None,None]
        age=t-start
        def mean(z,ids):
            mask=valid[:,ids]
            return (z[:,ids]*mask).sum(1)/mask.sum(1).clip(1)
        def smooth(z,tau):
            alpha=1-np.exp(-1/tau)
            return lfilter([alpha],[1,alpha-1],z-z[:1],axis=0)+z[:1]
        def basket(ids):
            expected=mean(s*survivor[...,1],ids)
            late=mean(s*survivor[...,2],ids)
            arrivals=mean(s*(age<5),ids)
            staying=mean(s*(age>=20),ids)
            pressure=mean(v,ids)
            ttl=mean(s,ids)
            return expected,late,arrivals,staying,pressure,ttl
        total=basket(slice(None))
        expected,late,arrivals,staying,pressure,ttl=total
        # Compare the realized survivors of an older cohort with what was forecast at issue.
        actual=np.zeros_like(ttl)
        for lag,h in ((5,0),(20,1)):
            survived=(start[lag:]==start[:-lag])&valid[lag:]
            use=valid[:-lag]
            surprise=((survived-survivor[:-lag,...,h])*use*s[:-lag]).sum(1)/use.sum(1).clip(1)
            actual[lag:]+=surprise
        x0=np.column_stack((p[a:b,:,0].std(1,dtype=float),p[a:b,:,1:].mean(1,dtype=float),ttl,pressure))
        x1=np.column_stack((expected,late,expected-smooth(expected,45)))
        x2=np.column_stack((actual,smooth(actual,20),arrivals*(1-abs(staying))))
        l=basket(slice(0,None,2));r=basket(slice(1,None,2))
        # Fresh pressure in one basket is supported when older pressure in the other has the same sign.
        def agree(x,y):
            return np.where(x*y>0,np.sign(x)*np.minimum(abs(x),abs(y)),0)
        flow=.5*(agree(l[2],r[3])+agree(r[2],l[3]))
        bridge=.5*(agree(l[0]-smooth(l[0],18),r[2])+agree(r[0]-smooth(r[0],18),l[2]))
        x3=np.column_stack((flow,bridge,agree(l[1],r[1])*(1-abs(ttl))))
        blocks.append(np.column_stack((x0,x1,x2,x3)))
    return np.concatenate(blocks)


def moments(x,p,y):
    # Keep exact pooled scoring statistics even though the candidate adds one value per timestamp.
    m=p.mean(1,dtype=float); ym=y.mean(1,dtype=float)
    z=np.column_stack((np.ones(len(x)),m,ym,x))
    g=z.T@z/len(z)
    g[1,1]=np.square(p,dtype=float).mean()
    g[2,2]=np.square(y,dtype=float).mean()
    g[1,2]=g[2,1]=np.multiply(p,y,dtype=float).mean()
    return g*len(x)*p.shape[1]


def fit(hist,groups):
    rows=np.stack([g/g[0,0]-np.outer(g[:,0],g[0])/g[0,0]**2 for g in hist])
    g=rows.mean(0)
    base=np.arange(3,3+groups[0])
    slopes=(rows[:,1,2]/rows[:,1,1]).clip(.2,4)
    banks=[]
    for lo,hi in zip(groups[:-1],groups[1:]):
        ids=np.arange(3+lo,3+hi)
        proj=np.eye(len(g))[:,ids]
        prior=g[np.ix_(base,base)]
        proj[base]=-np.linalg.solve(prior+.15*np.diag(prior.diagonal().clip(.0001)),g[np.ix_(base,ids)])
        gram=proj.T@g@proj
        sd=np.sqrt(gram.diagonal()).clip(.003)
        ev,u=np.linalg.eigh(gram/np.outer(sd,sd))
        axes=proj@(u/(sd[:,None]*np.sqrt(ev.clip(0)+.8)))
        grad=(rows[:,:,2]/slopes[:,None]-rows[:,:,1])@axes
        mu=grad.mean(0)
        w=.5*axes@(mu*mu**2/(mu**2+grad.var(0,ddof=1)/len(hist)+1e-8))
        w*=min(1,.035*np.sqrt(g[1,1])/np.sqrt(w@g@w).clip(1e-9))
        banks.append(w[3:])
    return banks


def run(out,arms,make,groups,make_bank=direction.table,reference=REFERENCE):
    out.mkdir(exist_ok=True)
    code=rel.fields.parent.codec()
    hist=[];checks={k:[] for k in arms}
    for d in range(4,16):
        if d in (4,6,8,12):rate=make_bank(d)
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        p=(np.load(reference/f"rolling_{d:02d}.npy") if d>=6 else onset.settled.head.ranks.band.pairs.net.future.forecast(d))
        x=make(raw,grid,code,p,rate)
        if d==8:
            np.testing.assert_allclose(x[:512],make(raw[:512],grid[:512],code,p[:512],rate),atol=1e-8,rtol=0)
        g=moments(x,p[...,2],y[...,2])
        if d in (8,12):weights=fit(hist,groups)
        if d>=8:
            for k,w in zip(arms,weights):
                proj=np.eye(len(g),4);proj[3:,3]=w
                checks[k].append(proj.T@g@proj)
        hist.append(g)
    report={k:score(v) for k,v in checks.items()}
    good=[k for k,v in report.items() if min(v["folds"]+v["omit_one"])>0 and sum(y>0 for y in v["days"])>=5]
    best=max(good,key=lambda k:min(report[k]["folds"])) if good else None
    (out/"rolling.json").write_text(json.dumps(dict(selected={"2":best} if best else {},results=report),indent=2))
    if best:
        np.savez(out/"weights.npz",w=fit(hist,groups)[arms.index(best)],rate=make_bank(16),code=code)
    print(best,{k:v["folds"] for k,v in report.items()},flush=True)


def predict(raw,grid,p,params):
    q=p.copy()
    q[...,2]+=(features(raw,grid,params["code"],p,params["rate"])@params["w"])[:,None].astype(q.dtype)
    return q


if __name__=="__main__":
    run(OUT,ARMS,features,(13,28,43,58))
