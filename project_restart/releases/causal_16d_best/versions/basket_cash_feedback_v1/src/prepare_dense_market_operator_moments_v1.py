"""Cache daily training sufficient statistics and causal market states for operator comparisons."""
import json
import numpy as np
import prepare_dense_market_reliability_v1 as tape

market=tape.market
base=market.base
OUT=base.ROOT/"artifacts/dense_market_operator_moments_v1"


def main():
    OUT.mkdir(exist_ok=True)
    code=market.parent.codec()
    for day in range(16):
        load=lambda k:np.load(base.CACHE/f"day_{day:02d}_{k}.npy",mmap_mode="r")
        raw,grid,y=load("x"),load("time").astype(int),load("y")
        x,av=market.parent.state(raw,grid,code,with_market=True)
        m=market.market_state(av,grid)
        np.savez(OUT/f"day_{day:02d}.npz",stock=market.parent.moment(x,grid,y),market=market.record(m,grid,y),state=m)
    print("saved 16 strictly dated training covariances",flush=True)


def run(out,arms,operator):
    out.mkdir(exist_ok=True)
    predictions={}
    stock=[];common=[];cal=[];rows={(k,h):[] for k in arms for h in (1,2)}
    def fit(hist,k,target):
        mats=np.stack(hist);g=mats.mean(0);n=len(g)-3;col=5+arms.index(k)
        control=np.arange(5)
        v=np.zeros(n);v[col]=1
        hh=g[np.ix_(control,control)]
        v[control]=-np.linalg.solve(hh+.035*np.diag(hh.diagonal().clip(1e-8)),g[control,col])
        slope=(mats[:,target,n+target]/mats[:,target,target]).clip(.25,4)
        daily=(mats[:,:n,n+target]/slope[:,None]-mats[:,:n,target])@v
        mu=daily.mean();mu*=abs(mu)/(abs(mu)+daily.std()/np.sqrt(len(daily))+1e-8)
        w=v*mu/(1.8*(v@g[:n,:n]@v)+1e-8)
        risk=np.sqrt(w@g[:n,:n]@w/g[target,target])
        return w*min(1,.055/max(risk,1e-8))
    for d in range(16):
        with np.load(OUT/f"day_{d:02d}.npz") as f:st,mk,m=f["stock"],f["market"],f["state"]
        if d>=6:
            if d in (6,8,12):
                bank=operator(stock,common)
                # The existing operator used days 1-4 until its day 8 refit.
                old=market.operators(stock[:4],common[:4]) if d==6 else market.operators(stock,common)
            with np.load(tape.OUT/f"day_{d:02d}.npz") as ref:q=ref["reference"];delta=ref["change"]
            y=np.load(base.CACHE/f"day_{d:02d}_y.npy",mmap_mode="r")
            x=np.column_stack((q.mean(1,dtype=float),delta,m@old[:,2],m@bank))
            if d in (8,12):weights={key:fit(cal,*key) for key in rows}
            if d>=8:
                for (k,h),v in rows.items():
                    signal=x@weights[k,h]
                    predictions[d,k,h]=signal.astype("float32")
                    z=np.broadcast_to(signal[:,None],q.shape[:2])
                    v.append(market.metric(q[...,h],y[...,h],z))
            zz=np.column_stack((x,y.mean(1,dtype=float)));zz-=zz.mean(0)
            cal.append(zz.T@zz/len(zz))
        stock.append(st);common.append(mk)
    report={f"{k}_{h}":dict(folds=[market.gain(sum(v[:4])),market.gain(sum(v[4:]))],days=[market.gain(g) for g in v]) for (k,h),v in rows.items()}
    picks={}
    for h in (1,2):
        ok=[k for k in arms if min(report[f"{k}_{h}"]["folds"])>0 and sum(t>0 for t in report[f"{k}_{h}"]["days"])>=5]
        if ok:picks[str(h)]=max(ok,key=lambda k:min(report[f"{k}_{h}"]["folds"]))
    (out/"rolling.json").write_text(json.dumps(dict(selected=picks,results=report),indent=2));print(picks,{k:v["folds"] for k,v in report.items()},flush=True)
    if not picks:return
    for d in range(8,16):
        shape=next(v.shape for (day,_,_),v in predictions.items() if day==d)
        update=np.zeros((shape[0],3),dtype="float32")
        for h,k in picks.items():update[:,int(h)]=predictions[d,k,int(h)]
        np.save(out/f"oof_{d:02d}.npy",update)
    w=np.zeros((5+len(arms),3))
    for h,k in picks.items():w[:,int(h)]=fit(cal,k,int(h))
    np.savez(out/"weights.npz",bank=operator(stock,common),w=w)
    outer=[]
    for d in range(16,20):
        grid,r,y=tape.outer(d);q=r["reference"]
        raw=np.load(base.CACHE/f"day_{d:02d}_x.npy",mmap_mode="r")
        with np.load(out/"weights.npz") as f,np.load(market.parent.OUT/"weights.npz") as st:
            _,av=market.parent.state(raw,grid,st["code"],with_market=True)
            m=market.market_state(av,grid)
            xx=np.column_stack((q.mean(1,dtype=float),r["change"],r["directions"][:,2],m@f["bank"]))
            adj=(xx@f["w"]).astype(q.dtype)
            pred=q+adj[:,None]
        outer.append((q,y,pred))
    score=market.summary(outer)
    assert score["rows"]==936250 and abs(score["baseline_score"]-json.loads((market.OUT/"outer.json").read_text())["pooled"]["score"])<1e-8
    (out/"outer.json").write_text(json.dumps(dict(pooled=score,daily=[market.summary([r]) for r in outer]),indent=2))
    print(score,flush=True)


if __name__=="__main__":main()
