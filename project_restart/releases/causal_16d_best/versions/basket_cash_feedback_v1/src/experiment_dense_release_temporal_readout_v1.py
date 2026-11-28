"""Walk-forward study of fast and slow components of the three incumbent forecast intervals."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import lfilter

import experiment_dense_release_spectral_readout_v1 as parent

base = parent.base
OUT = base.ROOT / "artifacts/dense_release_temporal_readout_v1"
ARMS = ("fast", "slow", "fast_and_slow")
HALF_LIVES = (4, 20, 80)


def design(p, grid):
    b = parent.bands(p)
    interval = np.flatnonzero((np.diff(grid)!=1) | ((grid[:-1]<2401) & (grid[1:]>=2401)))+1
    rows=[]
    for lo,hi in zip(np.r_[0,interval],np.r_[interval,len(p)]):
        block=b[lo:hi]
        smooth=[]
        for time in HALF_LIVES:
            alpha=1-np.exp(-np.log(2)/time)
            smooth.append(lfilter([alpha],[1,alpha-1],block-block[:1],axis=0)+block[:1])
        # The original signal is retained, so the fit can change how quickly a revision enters the forecast.
        rows.append(np.concatenate((block-smooth[0],smooth[0]-smooth[1],smooth[1]-smooth[2],smooth[2]),2))
    return np.concatenate(rows)


def stat(x,p,y):
    a=p.reshape(-1,3).astype(float)
    b=y.reshape(-1,3).astype(float)
    a-=a.mean(0)
    b-=b.mean(0)
    var=(a*a).mean(0)
    beta=((a*b).mean(0)/var).clip(.35,3.)
    res=y/beta-p
    m=x.mean(1)
    em=res.mean(1)
    out=[]
    for reg, lbl in ((x-m[:,None],res-em[:,None]),(m,em)):
        reg=reg.reshape(-1,12)
        reg=reg-reg.mean(0)
        rhs=lbl.reshape(-1,3)
        out.append((reg.T@reg/len(reg),reg.T@rhs/len(reg),var))
    return out


def fit(hist,side,arm):
    g,r,var=[np.stack(v) for v in zip(*(rec[side] for rec in hist))]
    cols={"fast":np.arange(6),"slow":np.arange(6,12),"fast_and_slow":np.arange(12)}[arm]
    gram=g.mean(0)[np.ix_(cols,cols)]
    grad=r.mean(0)[cols]
    noise=r[:,cols].std(0,ddof=1)/np.sqrt(len(hist))
    grad=grad*abs(grad)/(abs(grad)+noise+1e-9)
    w=np.linalg.solve(gram+.85*np.diag(gram.diagonal().clip(1e-9)),grad)
    speed=np.sqrt((w*(gram@w)).sum(0)/var.mean(0))
    full=np.zeros((12,3))
    full[cols]=w*np.minimum(1.,.065/speed.clip(1e-8))
    return full


def evaluate_group(p,y,z,day):
    # Sufficient statistics allow pooling all rows without averaging correlations.
    ref_idx=np.arange(len(p))%5==day%5
    return np.stack([[parent.metric(p[mask,:,j],y[mask,:,j],z[mask,:,j]) for j in range(3)]
                    for mask in (ref_idx,~ref_idx)])


def apply(x,w,side):
    m=x.mean(1)
    if side==0:
        return (x-m[:,None])@w
    return np.broadcast_to((m@w)[:,None],(*x.shape[:2],3))


def screen():
    OUT.mkdir(exist_ok=True)
    hist,results=[],{}
    for d in range(4,16):
        p=np.load(parent.TAPE/f"forecast_{d:02d}.npy")
        grid=np.load(base.CACHE/f"day_{d:02d}_time.npy").astype(int)
        y=np.load(base.CACHE/f"day_{d:02d}_y.npy",mmap_mode="r")
        x=design(p,grid)
        if d==8:
            for cut in (341,2412):
                np.testing.assert_array_equal(x[:cut],design(p[:cut],grid[:cut]))
        if d in (8,12):
            weights={(side,arm):fit(hist,side,arm) for side in range(2) for arm in ARMS}
        if d>=8:
            for (side,arm),w in weights.items():
                z=apply(x,w,side)
                key=f"{side}_{arm}"
                results.setdefault(key,[]).append(evaluate_group(p,y,z,d))
        hist.append(stat(x,p,y))
    params={}
    report={}
    for j in range(3):
        for side in range(2):
            key=f"{'stock' if side==0 else 'market'}_Y{j+1}"
            scores={}
            for arm in ARMS:
                rows=np.stack(results[f"{side}_{arm}"])[:,:,j]
                wins=[]
                for fold in (rows[:4],rows[4:]):
                    phase=fold.sum(0)
                    wins.append(dict(full=parent.gain(phase.sum(0)),sampled=parent.gain(phase[0]),
                                     complement=parent.gain(phase[1]),daily=[parent.gain(r.sum(0)) for r in fold]))
                scores[arm]=wins
            eligible=[a for a,folds in scores.items() if all(min(f["sampled"],f["complement"])>0 for f in folds)
                      and sum(s>0 for f in folds for s in f["daily"])>=5]
            best=max(eligible,key=lambda a:min(f["full"] for f in scores[a])) if eligible else None
            if best:
                params[key]=fit(hist,side,best)[:,j]
            report[key]=dict(arm=best,scores=scores)
            print(key,best, {a:[round(f["full"],6) for f in v] for a,v in scores.items()},flush=True)
    np.savez(OUT/"weights.npz",**params)
    (OUT/"rolling.json").write_text(json.dumps(report,indent=2))


def summary(rows):
    p,y,q=(np.concatenate(col) for col in zip(*rows))
    def corr(v):
        return [float(np.corrcoef(v[...,j].ravel(),y[...,j].ravel())[0,1]) for j in range(3)]
    a,b=corr(p),corr(q)
    return dict(rows=y.shape[0]*50,base=a,candidate=b,baseline_score=float(np.dot(a,[.2,.3,.5])),
                score=float(np.dot(b,[.2,.3,.5])),target_gains=np.subtract(b,a).tolist())


def outer():
    with np.load(OUT/"weights.npz") as f:
        if not f.files:return
    rows=[]
    for d in range(16,20):
        raw=np.load(base.CACHE/f"day_{d:02d}_x.npy",mmap_mode="r")
        p=parent.released(d,raw)
        grid=np.load(base.CACHE/f"day_{d:02d}_time.npy").astype(int)
        def infer(p,grid):
            x=design(p,grid)
            m=x.mean(1)
            q=p.copy()
            with np.load(OUT/"weights.npz") as f:
                for name in f:
                    group,task=name.split("_")
                    j=int(task[1:])-1
                    z=(x-m[:,None])@f[name] if group=="stock" else (m@f[name])[:,None]
                    q[...,j]+=z
            return q
        q=infer(p,grid)
        for cut in (215,2462):
            np.testing.assert_array_equal(q[:cut],infer(p[:cut],grid[:cut]))
        rows.append((p,np.load(base.CACHE/f"day_{d:02d}_y.npy",mmap_mode="r"),q))
    scores=summary(rows)
    assert scores["rows"]==936250 and abs(scores["baseline_score"]-.37095867568143737)<1e-8
    (OUT/"outer.json").write_text(json.dumps(dict(pooled=scores, daily=[summary([r]) for r in rows]),indent=2))
    print(scores,flush=True)


if __name__=="__main__":
    screen()
    outer()
