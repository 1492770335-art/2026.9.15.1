"""Condition the original pairwise margin calibration on the trading speeds of the two names."""
import json
import numpy as np
from scipy.signal import lfilter
from experiment_dense_release_spectral_readout_v1 import TAPE, base, released, gain, metric, center
from experiment_dense_release_temporal_readout_v1 import summary

OUT=base.ROOT/"artifacts/dense_release_pair_activity_v1"
ARMS=("signed_pair_votes","distal_reliability")


def sessions(grid):
    cuts=np.flatnonzero((np.diff(grid)!=1)|((grid[:-1]<2401)&(grid[1:]>=2401)))+1
    return zip(np.r_[0,cuts],np.r_[cuts,len(grid)])


def ticks(raw,p,grid):
    for a,b in sessions(grid):
        changed=np.abs(raw[a:b,:,0].astype(float)-.00136)>.0001
        alpha=1-np.exp(-1/32)
        speed=lfilter([alpha],[1,alpha-1],changed-.30,axis=0)+.30
        starts=np.arange(0,b-a,4)
        for start in range(0,len(starts),64):
            t=starts[start:start+64]
            u=p[a+t,:,1]
            margin=u[:,:,None]-u[:,None,:]
            s=np.sign(margin)
            difference=s*(speed[t,:,None]-speed[t,None,:])
            side=np.digitize(difference,[-.12,0,.12])
            group=np.digitize(abs(margin),[.10,.25,.50,.90])*4+side
            yield a,b,a+t,s,np.tanh(abs(margin)/.25),group


def learn(raw,p,grid,y):
    sums=np.zeros((20,3))
    for a,b,t,s,fw,group in ticks(raw,p,grid):
        keep=t+20<b
        target=y[t[keep],:,1].astype(float)
        diff=target[:,:,None]-target[:,None,:]
        pred_mass=fw[keep]
        mass=pred_mass*np.tanh(abs(diff)/.4)
        for j,col in enumerate((mass*np.sign(diff)*s[keep],mass,pred_mass)):
            sums[:,j]+=np.bincount(group[keep].ravel(),weights=col.ravel(),minlength=20)
    return sums/sums[:,-1].sum()


def table(hist):
    g=np.mean(hist,axis=0).reshape(5,4,3)
    pool=g.sum(1,keepdims=True)
    a=g+.06*pool
    return (a[...,0]/a[...,1].clip(1e-9)-pool[...,0]/pool[...,1].clip(1e-9)).ravel()


def design(raw,p,grid,state):
    row=np.zeros((*p.shape[:2],2),np.float32)
    for a,b,t,s,fw,group in ticks(raw,p,grid):
        v=fw*state[group]
        row[t,:,0]=(v*s).sum(2)/49
        row[t,:,1]=v.sum(2)/49
    out=np.zeros_like(row)
    for a,b in sessions(grid):
        marks=np.arange(a,b,4)
        ix=(np.arange(a,b)-a)//4
        data=row[marks]
        mean=lfilter([1-np.exp(-4/96)],[1,-np.exp(-4/96)],data,axis=0)
        # Use freshness-dependent deviations from learned marginal correctness.
        out[a:b,:,0]=center((.5*data[:,:,0]+.5*mean[:,:,0])[ix])
        distal=center(p[a:b,:,2]-base.UNITS[1]*p[a:b,:,1])
        out[a:b,:,1]=center(center(mean[ix,:,1])*distal)
    return out


def weight(hist):
    moments=np.stack(hist)
    n=moments[:,0,0]
    cov=moments[:,1:,1:]/n[:,None,None]-moments[:,0,1:,None]*moments[:,0,None,1:]/n[:,None,None]**2
    slope=(cov[:,0,1]/cov[:,0,0]).clip(.35,3.)
    grad=cov[:,1,2]/slope-cov[:,0,2]
    v=cov[:,2,2]-cov[:,0,2]**2/cov[:,0,0]
    w=grad.mean()/(1.5*v.mean()+1e-10)*abs(grad.mean())/(abs(grad.mean())+grad.std()/np.sqrt(len(hist))+1e-9)
    limit=.06*np.sqrt(cov[:,0,0].mean()/max(v.mean(),1e-10))
    return float(np.clip(w,-limit,limit))


def rolling():
    OUT.mkdir(exist_ok=True)
    hist,cal,rows=[],[[],[]],[[],[]]
    for d in range(4,16):
        raw=np.load(base.CACHE/f"day_{d:02d}_x.npy",mmap_mode="r")
        grid=np.load(base.CACHE/f"day_{d:02d}_time.npy").astype(int)
        p=np.load(TAPE/f"forecast_{d:02d}.npy")
        y=np.load(base.CACHE/f"day_{d:02d}_y.npy",mmap_mode="r")
        if d in (6,8,12):
            state=table(hist)
            if d>=8:w=[weight(h) for h in cal]
        if d>=6:
            z=design(raw,p,grid,state)
            if d==8:
                n=2461
                np.testing.assert_array_equal(z[:n],design(raw[:n],p[:n],grid[:n],state))
            for j,h in enumerate(cal):
                h.append(metric(p[...,2],y[...,2],z[...,j]))
                if d>=8:
                    delta=w[j]*z[...,j]
                    rows[j].append((metric(p[...,2],y[...,2],delta),metric(p[...,2],y[...,2],np.roll(delta,13,axis=1))))
        hist.append(learn(raw,p,grid,y))
    selected=None
    report={}
    for arm,rr in zip(ARMS,rows):
        daily=[gain(t[0]) for t in rr]
        folds=[[gain(g) for g in np.sum(fold,axis=0)] for fold in (rr[:4],rr[4:])]
        report[arm]=dict(folds=folds,daily=daily)
        if selected is None and all(g>max(0,c) for g,c in folds) and sum(d>0 for d in daily)>=5:selected=arm
    print(selected,{k:v["folds"] for k,v in report.items()},flush=True)
    (OUT/"rolling.json").write_text(json.dumps(dict(selected=selected,results=report),indent=2))
    if selected:
        idx=ARMS.index(selected)
        np.savez(OUT/"model.npz",state=table(hist),index=idx,weight=weight(cal[idx]))
    return selected


def outer():
    results=[]
    for d in range(16,20):
        raw=np.load(base.CACHE/f"day_{d:02d}_x.npy",mmap_mode="r")
        grid=np.load(base.CACHE/f"day_{d:02d}_time.npy").astype(int)
        p=released(d,raw)
        with np.load(OUT/"model.npz") as f:
            z=design(raw,p,grid,f["state"])
            n=2416
            np.testing.assert_array_equal(z[:n],design(raw[:n],p[:n],grid[:n],f["state"]))
            q=p.copy()
            q[...,2]+=z[...,int(f["index"])]*float(f["weight"])
        results.append((p,np.load(base.CACHE/f"day_{d:02d}_y.npy",mmap_mode="r"),q))
    stat=summary(results)
    assert stat["rows"]==936250 and abs(stat["baseline_score"]-.37095867568143737)<1e-8
    (OUT/"outer.json").write_text(json.dumps(dict(pooled=stat,daily=[summary([r]) for r in results]),indent=2))
    print(stat,flush=True)


if __name__=="__main__" and rolling():outer()
