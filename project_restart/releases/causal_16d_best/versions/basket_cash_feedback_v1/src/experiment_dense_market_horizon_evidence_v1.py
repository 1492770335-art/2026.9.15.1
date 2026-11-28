"""Compare stock payoff horizon pooling with horizon weights learned from market response on independent dates."""
import json
import numpy as np
import experiment_dense_cross_scale_clocks_v1 as clock

market,tape,base,nonlinear,precision=clock.market,clock.tape,clock.base,clock.nonlinear,clock.precision
OUT=base.ROOT/"artifacts/dense_market_horizon_evidence_v1"
ARMS=("fixed_mixture","matched_clock_only","payoff_agreement","pooled_evidence")


def normalize(x):
    return x/x.sum(axis=0,keepdims=True).clip(1e-12)


def operator(hist):
    st,mk,sy,my,sv,mv=[np.stack(z) for z in zip(*hist)]
    sd=np.sqrt(np.diag(mk[:,0].mean(0))).clip(.006);ss=np.sqrt(np.diag(st[:,0].mean(0))).clip(.03)
    g=.5*(mk[:,0].mean(0)/np.outer(sd,sd)+st[:,0].mean(0)/np.outer(ss,ss))
    eig,u=np.linalg.eigh(g+.25*np.eye(len(g)));w=u/np.sqrt(eig)
    rm=w/sd[:,None];rs=w/ss[:,None]
    cov=(rm.T@mk[:,3:5].mean((0,1))@rm+rs.T@st[:,3:5].mean((0,1))@rs)*.5
    _,axis=np.linalg.eigh(.5*(cov+cov.T));rm=rm@axis;rs=rs@axis
    ms=rs.T@st@rs;mm=rm.T@mk@rm
    sp=(rs.T@sy)*np.sqrt(mv[:,2].mean()/sv.mean(0))
    mp=(rm.T@my)[:,:,2]
    lags=np.array(clock.clock.LAGS);horizons=np.array(clock.clock.HORIZONS)
    prior=normalize(np.exp(-.5*np.square(np.log(horizons/20)/.80)))[:,None]
    results=[];summary=[]
    for j in (0,1):
        train=slice(j,None,2);test=slice(1-j,None,2)
        sm=ms[train].mean(0);ma=mm[train].mean(0)
        src=sm.diagonal(axis1=1,axis2=2);dst=ma.diagonal(axis1=1,axis2=2)
        sr=src/src[:1].clip(.008);dr=dst/dst[:1].clip(.008)
        ac=np.stack([np.array([np.interp(h,lags,sr[:,i]) for i in range(sr.shape[1])]) for h in horizons])
        true=clock.select_weights(ac,dr[3],20,horizons)
        seq=sp[train]-mp[train,:,None]
        n=len(seq)
        # Across-day products estimate repeatable discrepancies. Shrink unstable local choices.
        noise=seq.var(0,ddof=1)/n
        loss=(seq.mean(0)**2-noise).clip(0)
        denominator=(noise.mean(1,keepdims=True)+np.median(noise)+1e-7)
        relative=(loss/denominator).clip(0,7).T
        evidence=normalize(true*np.exp(-.5*relative))
        group=normalize(prior*np.exp(-relative.mean(1,keepdims=True)))
        allpay=sp[test].mean(0)
        mixed=(sm[3]+sm[4]+ma[3]+ma[4])*.5
        op=(mixed+mixed.T)*.5
        delta=lambda weights,compare:rm@op@(.5*((weights-compare).T*allpay).sum(1))
        original=np.eye(5)[:,2:3]
        results.append(np.column_stack((delta(prior,original),delta(true,prior),delta(evidence,true),delta(group,prior))))
        summary.append(dict(clock=true.mean(1).tolist(),agreement=evidence.mean(1).tolist(),global_weights=group[:,0].tolist()))
    return np.mean(results,0),summary


def fit(hist,j,h,n_controls=6):
    c=np.stack(hist);g=c.mean(0);n=len(g)-3
    v=np.zeros(n);v[n_controls+j]=1
    ids=np.arange(n_controls);hh=g[np.ix_(ids,ids)]
    # Retain earlier forecasts and the existing clock-specific change as controls.
    v[ids]=-np.linalg.solve(hh+.045*np.diag(hh.diagonal().clip(1e-8)),g[ids,n_controls+j])
    slope=(c[:,h,n+h]/c[:,h,h]).clip(.25,4.)
    gradient=(c[:,:n,n+h]/slope[:,None]-c[:,:n,h])@v
    rhs=gradient.mean()
    rhs*=abs(rhs)/(abs(rhs)+gradient.std()/np.sqrt(len(c))+1e-8)
    w=v*rhs/(1.9*(v@g[:n,:n]@v)+1e-8)
    risk=np.sqrt(w@g[:n,:n]@w/g[h,h])
    return w*min(1,.045/max(risk,1e-8))


def latest(q,r,m):
    q=clock.inference_inputs(q,r,m)
    with np.load(clock.OUT/"weights.npz") as f:
        z=m@f["bank"]
        xx=np.column_stack((q.mean(1,dtype=float),r["change"],r["directions"][:,2],z))
        update=(xx@f["w"]).astype(q.dtype);q=q+update[:,None]
    return q,update


def run(output,arms,construct):
    output.mkdir(exist_ok=True)
    history=[];cal=[];rows={(k,h):[] for k in arms for h in (1,2)};diagnostics={}
    for day in range(16):
        with np.load(clock.clock.OUT/f"day_{day:02d}.npz") as f:v=[f[k] for k in ("stock","market","stock_pay","market_pay","stock_var","market_var")]
        if day>=6:
            if day in (6,8,12):bank,diagnostics[day]=construct(history)
            with np.load(clock.live.data.OUT/f"day_{day:02d}.npz") as f:m=f["state"]
            with np.load(tape.OUT/f"day_{day:02d}.npz") as f:q=f["reference"];dm=f["change"];raw=f["directions"][:,2]
            clock_change=np.zeros(len(q))
            if day>=8:
                for module in (precision,nonlinear,clock):
                    adj=np.load(module.OUT/f"oof_{day:02d}.npy");q+=adj[:,None]
                clock_change=adj[:,2]
            y=np.load(base.CACHE/f"day_{day:02d}_y.npy",mmap_mode="r")
            z=np.column_stack((dm,raw,clock_change,m@bank))
            if day in (8,12):weights={key:fit(cal,arms.index(key[0]),key[1]) for key in rows}
            if day>=8:
                xx=np.column_stack((q.mean(1,dtype=float),z))
                for key,val in rows.items():
                    k,h=key;dz=np.broadcast_to((xx@weights[key])[:,None],q.shape[:2])
                    val.append(market.metric(q[...,h],y[...,h],dz))
            cal.append(clock.live.stat(q,y,z))
        history.append(v)
    report={f"{k}_{h}":dict(folds=[market.gain(sum(v[:4])),market.gain(sum(v[4:]))],days=[market.gain(g) for g in v]) for (k,h),v in rows.items()}
    picks={}
    for h in (1,2):
        ok=[k for k in arms if min(report[f"{k}_{h}"]["folds"])>0 and sum(a>0 for a in report[f"{k}_{h}"]["days"])>=5]
        if ok:picks[str(h)]=max(ok,key=lambda k:min(report[f"{k}_{h}"]["folds"]))
    (output/"rolling.json").write_text(json.dumps(dict(selected=picks,results=report,fit_diagnostics=diagnostics),indent=2))
    print(picks,{k:v["folds"] for k,v in report.items()},flush=True)
    if not picks:return
    w=np.zeros((6+len(arms),3))
    for h,k in picks.items():w[:,int(h)]=fit(cal,arms.index(k),int(h))
    bank,_=construct(history);np.savez(output/"weights.npz",bank=bank,w=w)
    outer=[]
    for d in range(16,20):
        grid,r,y=tape.outer(d)
        raw=np.load(base.CACHE/f"day_{d:02d}_x.npy",mmap_mode="r")
        with np.load(market.parent.OUT/"weights.npz") as f:_,av=market.parent.state(raw,grid,f["code"],with_market=True)
        m=market.market_state(av,grid);q,change=latest(r["reference"],r,m)
        with np.load(output/"weights.npz") as f:
            xx=np.column_stack((q.mean(1,dtype=float),r["change"],r["directions"][:,2],change[:,2],m@f["bank"]))
            pred=q+(xx@f["w"]).astype(q.dtype)[:,None]
        outer.append((q,y,pred))
    score=market.summary(outer)
    assert score["rows"]==936250 and abs(score["baseline_score"]-json.loads((clock.OUT/"outer.json").read_text())["pooled"]["score"])<1e-8
    (output/"outer.json").write_text(json.dumps(dict(pooled=score,daily=[market.summary([r]) for r in outer]),indent=2));print(score,flush=True)


if __name__=="__main__":run(OUT,ARMS,operator)
