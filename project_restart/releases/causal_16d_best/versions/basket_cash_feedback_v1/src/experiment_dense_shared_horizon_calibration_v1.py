"""Separate repeatable multi-horizon book effects from correlated forecast noise across training dates."""
import json
import numpy as np
import experiment_dense_field_arrival_market_v1 as arrival
import experiment_dense_future_expert_value_v1 as future
from prepare_dense_payoff_reference_v1 import OUT as REFERENCE

rel=future.rel
OUT=REFERENCE.parent/"dense_shared_horizon_calibration_v1"
ARMS=("separate","shared_task_covariance","shared_leading_direction")


def features(raw,grid,p):
    q=p.astype(float)-p.mean(1,keepdims=True,dtype=float)
    for a,b in rel.fields.parent.sessions(grid):
        x=arrival.book(raw[a:b])
        x-=x.mean(1,keepdims=True)
        # The value, its innovation and its persistence can carry different return horizons.
        smooth=rel.fields.parent.smooth(x,32)
        lag=rel.fields.parent.smooth(x,160)
        vol=np.sqrt(.005+(x*x).mean(1,keepdims=True))
        fast=np.tanh((x-smooth)/vol)
        slow=np.tanh((smooth-lag)/vol)
        same=np.sign(fast)*np.minimum(np.abs(fast),np.abs(slow))*(fast*slow>0)
        opp=np.sign(fast)*np.minimum(np.abs(fast),np.abs(slow))*(fast*slow<0)
        terms=np.concatenate((fast,slow,same,opp),axis=-1)
        terms-=terms.mean(1,keepdims=True)
        yield a,b,np.concatenate((q[a:b],terms),axis=-1)


def collect():
    OUT.mkdir(exist_ok=True)
    short=rel.rolling_short_changes()
    for d in range(4,16):
        def load(k):return np.load(rel.ref.base.CACHE/f"day_{d:02d}_{k}.npy",mmap_mode="r")
        q=np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else future.forecast(d)
        if d in short:q[...,0]+=short[d][:,None]
        y=load("y")
        # The stored full-panel moments allow exact pooled Pearson scoring for any fitted linear readout.
        total=0
        for a,b,xx in features(load("x"),load("time").astype(int),q):
            for t in np.array_split(np.arange(b-a),6):
                z=np.concatenate((q[a+t],y[a+t],xx[t]),axis=-1).reshape(-1,xx.shape[-1]+6)
                zz=np.column_stack((np.ones(len(z)),z))
                total+=zz.T@zz
        np.save(OUT/f"moment_{d:02d}.npy",total)


def estimate(history):
    moments=np.stack(history)
    cov=moments[:,1:,1:]/moments[:,:1,:1]-(moments[:,1:,:1]/moments[:,:1,:1])@(moments[:,:1,1:]/moments[:,:1,:1])
    gram=cov[:,6:,6:].mean(0)
    proj=np.eye(len(gram))[:,3:]
    proj[:3]=-np.linalg.solve(gram[:3,:3]+.03*np.diag(gram.diagonal()[:3]),gram[:3,3:])
    g=proj.T@gram@proj
    sd=np.sqrt(g.diagonal()).clip(.015)
    eig,u=np.linalg.eigh(g/np.outer(sd,sd))
    basis=proj@(u/(sd[:,None]*np.sqrt(eig.clip(0)+.65)))
    slope=(np.diagonal(cov[:,:3,3:6],axis1=1,axis2=2)/cov[:,:3,:3].diagonal(axis1=1,axis2=2)).clip(.2,4.)
    gradient=basis.T@(cov[:,6:,3:6]/slope[:,None,:]-cov[:,6:,:3])
    unit=np.sqrt(cov[:,:3,:3].mean(0).diagonal()).clip(.002)
    gradient/=unit
    n=len(history)
    mu=gradient.mean(0)
    dev=gradient-mu
    # In groups of similarly identifiable book modes, estimate how often an effect observed for
    # one target reappears on another date for another target. Within-date noise is kept separate.
    groups=np.array_split(np.arange(len(mu)),6)
    reads=np.zeros((3,*mu.shape))
    signal_trace=noise_trace=0.
    for ix in groups:
        v=mu[ix]
        noise=np.einsum("dih,dik->hk",dev[:,ix],dev[:,ix])/(len(ix)*n*(n-1))
        between=v.T@v/len(ix)-noise
        val,rot=np.linalg.eigh(between)
        val=val.clip(0)
        clean=(rot*val)@rot.T
        floor=.15*np.trace(noise)/3+1e-7
        reads[0,ix]=v*(clean.diagonal()/(clean.diagonal()+noise.diagonal()+floor))
        for k,signal in ((1,clean),(2,np.outer(rot[:,-1],rot[:,-1])*val[-1])):
            reads[k,ix]=v@np.linalg.solve(signal+noise+floor*np.eye(3),signal)
        signal_trace+=clean.trace();noise_trace+=noise.trace()
    weights=basis@(reads*unit)
    risk=(weights*(gram@weights)).sum(1).clip(0)**.5/unit
    weights*=np.minimum(.75,.055/risk.clip(1e-8))[:,None,:]
    return weights,dict(signal_trace=float(signal_trace),mean_noise_trace=float(noise_trace))


def metric(moment,w,j):
    # Fold the reference, outcome, and proposed correction out of the same full-panel sufficient statistics.
    transform=np.zeros((len(moment),4))
    transform[0,0]=transform[1+j,1]=transform[4+j,2]=1.
    transform[7:,3]=w
    return transform.T@moment@transform


def score(rows):
    folds=(rows[:4],rows[4:])
    return dict(folds=[rel.ref.market.gain(sum(f)) for f in folds],days=[rel.ref.market.gain(v) for v in rows],
                omit_one=[rel.ref.market.gain(sum(f)-v) for f in folds for v in f])


def run():
    hist=[];rows={(j,k):[] for j in range(3) for k in ARMS}
    for d in range(4,16):
        moment=np.load(OUT/f"moment_{d:02d}.npy")
        if d in (8,12):weights,_=estimate(hist)
        if d>=8:
            for i,k in enumerate(ARMS):
                for j in range(3):rows[j,k].append(metric(moment,weights[i,:,j],j))
        hist.append(moment)
    report={str(j):{k:score(rows[j,k]) for k in ARMS} for j in range(3)}
    selected={}
    for j in range(3):
        arms=report[str(j)]
        ok=[k for k,v in arms.items() if min(v["folds"]+v["omit_one"])>0 and sum(s>0 for s in v["days"])>=5]
        if ok:selected[str(j)]=max(ok,key=lambda k:min(arms[k]["folds"]))
    bank,diag=estimate(hist)
    w=np.zeros_like(bank[0])
    for j,k in selected.items():w[:,int(j)]=bank[ARMS.index(k),:,int(j)]
    (OUT/"rolling.json").write_text(json.dumps(dict(selected=selected,results=report,diagnostics=diag),indent=2))
    if selected:np.save(OUT/"weights.npy",w)
    print(selected,{j:{k:v["folds"] for k,v in arms.items()} for j,arms in report.items()},flush=True)


def predict(raw,grid,p,weights):
    result=p.copy()
    for a,b,x in features(raw,grid,p):
        result[a:b]+=(x@weights).astype(p.dtype)
    return result


if __name__=="__main__":
    collect()
    run()
