"""Decode six logarithmic integer grids and measure their incremental value in the incumbent model."""
import json
import numpy as np
from scipy.signal import lfilter
from experiment_dense_session_slope_v1 import base,TAPE,released,sessions,metric,gain
from experiment_dense_release_temporal_readout_v1 import summary

OUT=base.ROOT/"artifacts/dense_decoded_counts_v1"
ARMS=("log_grid","decoded_increment")
COLS=np.arange(77,83)


def codec():
    raw=np.load(base.CACHE/"day_00_x.npy",mmap_mode="r")
    rank=np.column_stack((np.ones(32),np.log1p(3*np.arange(32))))
    return np.stack([np.linalg.lstsq(rank,np.unique(raw[:,:,j])[:32].astype(float),rcond=None)[0] for j in COLS])


def decode(s,code):
    return np.expm1((s.astype(float)-code[:,0])/code[:,1])/3


def history(x,tau):
    alpha=1-np.exp(-1/tau)
    return lfilter([alpha],[1,alpha-1],x,axis=0)


def views(raw,grid,code):
    for a,b in sessions(grid):
        v=raw[a:b,:,77:83].astype(float)
        approx=decode(v,code)
        n=np.rint(approx).clip(0,256)
        log=np.log1p(3*n)
        price=np.tanh((raw[a:b,:,0].astype(float)-.00136))
        cols=[]
        for xi in (log,n):
            # Chronological histories are sampled before the current field arrives.
            signed=xi[:,:,::2]-xi[:,:,1::2]
            total=xi[:,:,::2]+xi[:,:,1::2]
            slow_total=history(total,160)
            slow_signed=history(signed,160)
            prev=lambda x:np.concatenate((np.zeros_like(x[:1]),x[:-1]))
            expected=prev(slow_signed)/(2.5+prev(slow_total))
            innovation=(signed-total*expected)/np.sqrt(2+total+.04*total**2)
            burst=np.tanh(np.log((2+total)/(2+prev(slow_total))))
            paid=price[:,:,None]*np.sign(innovation)
            signals=np.concatenate((np.tanh(innovation),np.tanh(innovation)*burst,
                                    np.tanh(innovation)*paid,
                                    np.tanh(innovation)*np.tanh(signed/(2+total))),2)
            f=history(signals,8);s=history(signals,64)
            cols.append(np.concatenate((signals-f,f-s,s),2))
        yield a,b,np.concatenate(cols,2).astype("float32")


def matrices(p,y,v):
    full=p.astype(float);truth=y.astype(float)
    basis=np.concatenate((full,v),2)
    m=basis.mean(1)
    ym=truth.mean(1)
    rows=[]
    for b,z in ((basis-m[:,None],truth-ym[:,None]),(m,ym)):
        mat=np.concatenate((b,z),axis=-1)
        flat=mat.reshape(-1,mat.shape[-1])
        flat=flat-flat.mean(0)
        rows.append(flat.T@flat/len(flat))
    return np.stack(rows)


def fit(hist,side,arm):
    c=np.stack([m[side] for m in hist])
    n=c.shape[1]-3
    split=3+(n-3)//2
    start,stop=(3,split) if arm==ARMS[0] else (split,n)
    proj=np.eye(n)[:,start:stop]
    g=c.mean(0)
    def ridge(m,w):return m+w*np.diag(m.diagonal().clip(1e-7))
    proj[:start]=-np.linalg.solve(ridge(g[:start,:start],.16),g[:start,start:stop])
    h=proj.T@g[:n,:n]@proj
    slopes=np.stack([(c[:,j,n+j]/c[:,j,j].clip(1e-8)).clip(.35,3.) for j in range(3)],axis=1)
    rhs=proj.T@(c[:,:n,n:]/slopes[:,None]-c[:,:n,:3])
    avg=rhs.mean(0)
    avg*=abs(avg)/(abs(avg)+rhs.std(0)/np.sqrt(len(rhs))+1e-7)
    w=np.linalg.solve(ridge(h,.9),avg)
    risk=np.sqrt((w*(h@w)).sum(0).clip(0)/g.diagonal()[:3].clip(1e-8))
    return proj@(w*np.minimum(1.,.055/risk.clip(1e-8)))


def features(raw,grid,p,code):
    return np.concatenate([np.concatenate((p[a:b],v),2) for a,b,v in views(raw,grid,code)])


def run():
    OUT.mkdir(exist_ok=True)
    code=codec()
    hist=[];rows={};accuracy=[]
    for d in range(4,16):
        raw=np.load(base.CACHE/f"day_{d:02d}_x.npy",mmap_mode="r")
        grid=np.load(base.CACHE/f"day_{d:02d}_time.npy").astype(int)
        p=np.load(TAPE/f"forecast_{d:02d}.npy");y=np.load(base.CACHE/f"day_{d:02d}_y.npy")
        value=decode(raw[::7,:,77:83],code)
        use=value<50
        accuracy.append(dict(max_distance=float(abs(value[use]-np.rint(value[use])).max()),coverage=float(use.mean())))
        g=[];sizes=[]
        if d in (8,12):weights={(s,arm):fit(hist,s,arm) for s in range(2) for arm in ARMS}
        for a,b,v in views(raw,grid,code):
            # Session moments are given proportional weight within a date.
            g.append(matrices(p[a:b],y[a:b],v));sizes.append(b-a)
        if d>=8:
            f=features(raw,grid,p,code);m=f.mean(1,dtype=float)
            if d==8:
                k=2478
                np.testing.assert_array_equal(f[:k],features(raw[:k],grid[:k],p[:k],code))
            for (side,arm),w in weights.items():
                if side==0:
                    z=f@w-(m@w)[:,None]
                else:z=np.broadcast_to((m@w)[:,None],p.shape)
                for j in range(3):
                    key=f"{side}_{arm}_Y{j+1}"
                    rows.setdefault(key,[]).append(metric(p[...,j],y[...,j],z[...,j]))
        hist.append(np.average(g,weights=sizes,axis=0))
    report={};wanted={}
    for k,h in rows.items():
        report[k]=dict(folds=[gain(sum(v)) for v in (h[:4],h[4:])],daily=[gain(m) for m in h])
    for side in range(2):
        for j in range(3):
            key=f"{side}_{ARMS[1]}_Y{j+1}"
            r=report[key]
            if min(r["folds"])>0 and sum(v>0 for v in r["daily"])>=5:
                wanted[key]=fit(hist,side,ARMS[1])[:,j]
    (OUT/"rolling.json").write_text(json.dumps(dict(selected=list(wanted),results=report,geometry=code.tolist(),grid_checks=accuracy),indent=2))
    print(list(wanted),{k:v["folds"] for k,v in report.items() if "decoded" in k},flush=True)
    if not wanted:return
    np.savez(OUT/"model.npz",codec=code,**wanted)
    rr=[]
    for d in range(16,20):
        raw=np.load(base.CACHE/f"day_{d:02d}_x.npy",mmap_mode="r")
        grid=np.load(base.CACHE/f"day_{d:02d}_time.npy").astype(int)
        p=released(d,raw)
        def predict(raw,grid,p):
            q=p.copy()
            with np.load(OUT/"model.npz") as mod:
                f=features(raw,grid,p,mod["codec"]);m=f.mean(1,dtype=float)
                for key in wanted:
                    s,j=int(key[0]),int(key[-1])-1
                    w=mod[key]
                    q[...,j]+=f@w-(m@w)[:,None] if s==0 else (m@w)[:,None]
            return q
        q=predict(raw,grid,p)
        n=2443
        np.testing.assert_allclose(q[:n],predict(raw[:n],grid[:n],p[:n]),atol=1e-7,rtol=0)
        rr.append((p,np.load(base.CACHE/f"day_{d:02d}_y.npy"),q))
    result=summary(rr)
    assert result["rows"]==936250 and abs(result["baseline_score"]-.37095867568143737)<1e-8
    (OUT/"outer.json").write_text(json.dumps(dict(pooled=result,daily=[summary([r]) for r in rr]),indent=2))
    print(result,flush=True)


if __name__=="__main__":run()
