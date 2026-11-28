"""Horizon-specific calibration of the released forecasts from realized price changes since the open."""
import json
import numpy as np
from version_paths import GEO
from experiment_dense_release_spectral_readout_v1 import base,TAPE,released,metric,gain
from experiment_dense_release_pair_activity_v1 import sessions
from experiment_dense_release_temporal_readout_v1 import summary

OUT=base.ROOT/"artifacts/dense_session_slope_v1"
HORIZONS=(5,20,60)
ARMS=("risk","correlation","joint","intraday_clock")
EDGES=np.array([0,120,360,800,1600,2401,2761,3401,6000])


def stats(raw,p,grid):
    with np.load(GEO) as f:
        k,unit,row=float(f["k"]),f["coeff"][:,1],f["row"]
    n=len(p)
    total=np.zeros((n,3,2,6))
    ret=raw[:,:,0].astype(float)-.00136
    for a,b in sessions(grid):
        path=np.log1p(k*ret[a:b]).cumsum(0)
        invalid=np.vstack((np.zeros((1,50),int),np.all(raw[a:b]==row,axis=2).cumsum(0)))
        for j,h in enumerate(HORIZONS):
            if b-a<=h:continue
            issue=np.arange(b-a-h)
            stop=issue+h
            y=np.expm1(path[stop]-path[issue])*unit[j]/k
            ok=invalid[stop+1]==invalid[issue]
            w=ok.sum(1).clip(1)
            u=p[a+issue,:,j].astype(float)
            pm=(u*ok).sum(1)/w
            ym=(y*ok).sum(1)/w
            xx=u-pm[:,None]
            yy=y-ym[:,None]
            gm=np.stack((np.ones_like(w),pm,ym,pm**2,ym**2,pm*ym),1)
            zg=np.zeros_like(pm)
            gs=np.stack((np.ones_like(w),zg,zg,(xx**2*ok).sum(1)/w,(yy**2*ok).sum(1)/w,(xx*yy*ok).sum(1)/w),1)
            total[a+stop,j]=np.stack((gm,gs),1)*(w>=35)[:,None,None]
    return total


def ref(history):
    summed=np.stack(history).sum(0)
    return summed/summed[...,:1].clip(1)


def state(obs,prior):
    cs=obs.cumsum(0)
    t=np.arange(len(obs))
    # Include the morning's price evidence in afternoon calibration. Estimates
    # use only outcomes already completed within one continuous session.
    past=cs-np.where((t>=1440)[:,None,None,None],cs[(t-1440).clip(0)],0)
    def get(g):
        g=g[...,1:]/g[...,:1].clip(1e-8)
        x,y,xx,yy,xy=np.moveaxis(g,-1,0)
        vp=(xx-x*x).clip(.0001);vy=(yy-y*y).clip(.0001)
        return .5*np.log(vy/vp),np.arctanh(((xy-x*y)/np.sqrt(vy*vp)).clip(-.94,.94))
    # More pseudo-ticks for the market, which has one payoff per tick compared with 50 stock payoffs.
    ps=prior*np.array([650.,260.])[None,:,None]
    risk,skill=get(ps+past)
    r0,c0=get(ps)
    return (risk-r0).clip(-1.5,1.5),(skill-c0).clip(-1.5,1.5)


def design(p,obs,prior,grid):
    r,c=state(obs,prior)
    slot=np.searchsorted(EDGES[1:],grid,side="right").clip(0,len(EDGES)-2)
    clock=np.eye(len(EDGES)-1)[slot]
    pred=p[:,:,2].astype(float)
    mean=pred.mean(1,keepdims=True)
    components=(np.broadcast_to(mean,pred.shape),pred-mean)
    return [[np.concatenate((r[:,:,side],c[:,:,side]),1),clock] for side in range(2)],components


def basis_parts(features,arm):
    return {"risk":features[0][:,:3],"correlation":features[0][:,3:],"joint":features[0],"intraday_clock":features[1]}[arm]


def day_moment(p,y,x,src):
    # Exact daily sufficient statistics without materializing T x 50 x feature_count.
    pp=p[...,2].astype(float);yy=y[...,2].astype(float)
    g=src.mean(1)
    top=np.array([[pp.size,pp.sum(),yy.sum()],[pp.sum(),(pp*pp).sum(),(pp*yy).sum()],
                  [yy.sum(),(pp*yy).sum(),(yy*yy).sum()]])
    cross=50*np.stack([g@x,(src*pp).mean(1)@x,(src*yy).mean(1)@x])
    gg=50*x.T@((src*src).mean(1)[:,None]*x)
    return np.block([[top,cross],[cross.T,gg]])


def fit(records):
    # Estimate gradients inside two-day intervals issued by the same original checkpoint.
    pairs=[sum(records[i:i+2]) for i in range(0,len(records),2)]
    h=[];g=[]
    for mat in pairs:
        cov=mat[1:,1:]/mat[0,0]-np.outer(mat[0,1:],mat[0,1:])/mat[0,0]**2
        pred,label=cov[:2,:2].diagonal()
        slope=(cov[0,1]/pred).clip(.30,3.)
        h.append(cov[2:,2:]-np.outer(cov[2:,0],cov[2:,0])/pred)
        g.append(cov[2:,1]/slope-cov[2:,0])
    gram=np.mean(h,0)
    rhs=np.mean(g,0)
    rhs*=abs(rhs)/(abs(rhs)+np.std(g,axis=0)/np.sqrt(len(g))+.00001)
    return np.clip(np.linalg.solve(gram+.95*np.diag(gram.diagonal().clip(.0002)),rhs),-.6,.6)


def diagnose(p,y):
    def describe(s):
        a=p[s,:,2].astype(float);b=y[s,:,2].astype(float)
        m=a.mean(1);v=b.mean(1)
        return [float(np.cov(v,m,bias=True)[0,1]/np.var(m)),float(((a-m[:,None])*(b-v[:,None])).mean()/((a-m[:,None])**2).mean())]
    return [describe(slice(0,480)),describe(slice(480,None))]


def run():
    OUT.mkdir(exist_ok=True)
    totals=[];hist={};results={};diagnostics=[]
    for d in range(4,16):
        raw=np.load(base.CACHE/f"day_{d:02d}_x.npy",mmap_mode="r")
        grid=np.load(base.CACHE/f"day_{d:02d}_time.npy").astype(int)
        p=np.load(TAPE/f"forecast_{d:02d}.npy");y=np.load(base.CACHE/f"day_{d:02d}_y.npy")
        obs=stats(raw,p,grid)
        if d in (6,8,12):prior=ref(totals)
        if d in (8,12):weights={key:fit(records) for key,records in hist.items()}
        if d>=6:
            xx,parts=design(p,obs,prior,grid)
            if d==8:
                for n in (183,2451):
                    small=stats(raw[:n],p[:n],grid[:n])
                    np.testing.assert_array_equal(small,obs[:n])
                    a,_=design(p[:n],small,prior,grid[:n])
                    np.testing.assert_array_equal(a[0][0],xx[0][0][:n])
            for side in range(2):
                for arm in ARMS:
                    key=f"{side}_{arm}"
                    x=basis_parts(xx[side],arm)
                    hist.setdefault(key,[]).append(day_moment(p,y,x,parts[side]))
                    if d>=8:
                        gamma=(x@weights[key]).clip(-.3,.3)
                        change=parts[side]*gamma[:,None]
                        results.setdefault(key,[]).append(metric(p[...,2],y[...,2],change))
        diagnostics.append(diagnose(p,y))
        totals.append(obs.sum(0))
    report={};selection=[]
    for key,records in results.items():
        folds=[gain(sum(chunk)) for chunk in (records[:4],records[4:])]
        pairs=[gain(sum(records[i:i+2])) for i in range(0,8,2)]
        report[key]=dict(folds=folds,same_checkpoint=pairs,daily=[gain(v) for v in records])
    for side in range(2):
        candidates=[k for k,r in report.items() if k.startswith(f"{side}_") and min(r["folds"])>0 and sum(p>0 for p in r["same_checkpoint"])>=3 and sum(v>0 for v in r["daily"])>=5]
        if candidates:selection.append(max(candidates,key=lambda k:min(report[k]["folds"])))
    diagnostics=np.array(diagnostics)
    stagewise=diagnostics[4:].copy()
    for i in range(0,len(stagewise),2):stagewise[i:i+2]-=stagewise[i:i+2].mean(0)
    corrs=[float(np.corrcoef(stagewise[:,:,s].T)[0,1]) for s in (0,1)]
    print(selection,{k:[round(v,6) for v in r["folds"]] for k,r in report.items()},corrs,flush=True)
    (OUT/"rolling.json").write_text(json.dumps(dict(selected=selection,results=report,opening_rest_slope_corr_within_pairs=corrs,opening_rest_slopes=diagnostics.tolist()),indent=2))
    if not selection:return
    np.savez(OUT/"model.npz",prior=ref(totals),**{k:fit(hist[k]) for k in selection})
    rows=[]
    for d in range(16,20):
        raw=np.load(base.CACHE/f"day_{d:02d}_x.npy",mmap_mode="r")
        grid=np.load(base.CACHE/f"day_{d:02d}_time.npy").astype(int)
        p=released(d,raw)
        def infer(raw,grid,p):
            with np.load(OUT/"model.npz") as m:
                xx,parts=design(p,stats(raw,p,grid),m["prior"],grid);q=p.copy()
                for key in selection:
                    i,arm=key.split("_",1);idx=int(i)
                    factor=(basis_parts(xx[idx],arm)@m[key]).clip(-.3,.3)
                    q[...,2]+=parts[idx]*factor[:,None]
            return q
        q=infer(raw,grid,p)
        np.testing.assert_allclose(q[:2437],infer(raw[:2437],grid[:2437],p[:2437]),atol=1e-7,rtol=0)
        rows.append((p,np.load(base.CACHE/f"day_{d:02d}_y.npy"),q))
    result=base.score(*(np.concatenate(c) for c in zip(*rows)))
    assert result["rows"]==936250 and abs(result["base_score"]-.37095867568143737)<1e-8
    (OUT/"outer.json").write_text(json.dumps(dict(pooled=result,daily=[base.score(*r) for r in rows]),indent=2))
    print(result,flush=True)


if __name__=="__main__":run()
