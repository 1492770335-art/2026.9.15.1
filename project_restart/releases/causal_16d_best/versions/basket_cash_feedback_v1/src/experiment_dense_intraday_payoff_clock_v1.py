"""Use market-state persistence observed earlier in the same session to adjust transferred payoff horizons."""
import json
import numpy as np
import experiment_dense_market_horizon_evidence_v1 as ref

clock=ref.clock
base=ref.base
OUT=base.ROOT/"artifacts/dense_intraday_payoff_clock_v1"
ARMS=("local_clock","common_speed","confirmed_speed")
HORIZONS=np.array(clock.clock.HORIZONS)
LAGS=np.array(clock.clock.LAGS)


def parameters(history):
    st,mk,sy,my,sv,mv=[np.stack(z) for z in zip(*history)]
    sd=np.sqrt(np.diag(mk[:,0].mean(0))).clip(.006)
    ss=np.sqrt(np.diag(st[:,0].mean(0))).clip(.03)
    g=.5*(mk[:,0].mean(0)/np.outer(sd,sd)+st[:,0].mean(0)/np.outer(ss,ss))
    val,u=np.linalg.eigh(g+.25*np.eye(len(g)));w=u/np.sqrt(val)
    rm=w/sd[:,None];rs=w/ss[:,None]
    both=.5*(rm.T@mk[:,3:5].mean((0,1))@rm+rs.T@st[:,3:5].mean((0,1))@rs)
    _,axes=np.linalg.eigh(.5*(both+both.T));rm=rm@axes;rs=rs@axes
    ms=rs.T@st@rs;mm=rm.T@mk@rm
    pay=(rs.T@sy)*np.sqrt(mv[:,2].mean()/sv.mean(0))
    src=[];fixed=[];variance=[];op=[];reward=[]
    for j in (0,1):
        s=ms[j::2].mean(0);m=mm[j::2].mean(0)
        d=s.diagonal(axis1=1,axis2=2)
        a=d/d[:1].clip(.008);source=np.stack([np.interp(HORIZONS,LAGS,c) for c in a.T],1)
        var=m[0].diagonal().clip(.008)
        target=(m[3].diagonal()/var).clip(-1,1)
        mixed=.5*(s[3]+s[4]+m[3]+m[4])
        src.append(source);fixed.append(target);variance.append(var)
        op.append(rm@(.5*(mixed+mixed.T)));reward.append(.5*pay[1-j::2].mean(0))
    return dict(transform=rm,source=np.stack(src),fixed=np.stack(fixed),variance=np.stack(variance),op=np.stack(op),pay=np.stack(reward))


def design(state,grid,f):
    cov=f["variance"];fixed=f["fixed"];source=f["source"];pay=f["pay"]
    baseline=[clock.select_weights(source[i],fixed[i],20,HORIZONS) for i in range(2)]
    result=np.zeros((len(grid),3))
    dt=48;decay=np.exp(-dt/560);pseudo=150.
    for a,b in ref.market.parent.sessions(grid):
        m=state[a:b];n=len(m)
        raw=m@f["transform"]
        dyn=np.clip(raw,-5*np.sqrt(cov.mean(0)),5*np.sqrt(cov.mean(0)))
        num=np.zeros_like(cov);den=np.zeros_like(cov)
        for t in range(0,n,dt):
            num*=decay;den*=decay
            ix=np.arange(max(20,t-dt),t,2)
            if len(ix):
                prev=dyn[ix-20];current=dyn[ix]
                k=t//dt%2
                num[k]+=(prev*current).sum(0);den[k]+=.5*(prev**2+current**2).sum(0)
            block=slice(t,min(n,t+dt))
            for i in (0,1):
                def estimate(x,y,mass):
                    return ((x-y*fixed[i])/(mass*cov[i]+y)).clip(-.30,.30)
                drift=estimate(num.sum(0),den.sum(0),pseudo)
                left=estimate(num[0],den[0],pseudo/2);right=estimate(num[1],den[1],pseudo/2)
                agree=np.where(left*right>0,np.sign(left)*np.minimum(abs(left),abs(right)),0)
                # Weight modes by sensitivity of the trained payoff to its horizon.
                importance=pay[i].std(1)*np.sqrt(cov[i])
                common=np.sum(importance*drift)/importance.sum().clip(1e-8)
                for j,delta in enumerate((drift,common,agree)):
                    select=clock.select_weights(source[i],fixed[i]+delta,20,HORIZONS)
                    coefficient=f["op"][i]@((select-baseline[i]).T*pay[i]).sum(1)
                    result[a+block.start:a+block.stop,j]+=.5*(m[block]@coefficient)
    return result


def run(out,arms,train,views,*,keep_oof=False,prior_head=None,prepare_inputs=None,targets=(1,2),check_leave_one_out=False):
    out.mkdir(exist_ok=True)
    parents=() if prior_head is None else prior_head if isinstance(prior_head,tuple) else (prior_head,)
    predictions={}
    history=[];cal=[];panels={(k,h):[] for k in arms for h in targets}
    for d in range(16):
        with np.load(clock.clock.OUT/f"day_{d:02d}.npz") as f:v=[f[k] for k in ("stock","market","stock_pay","market_pay","stock_var","market_var")]
        if d>=6:
            if d in (6,8,12):params=train(history)
            with np.load(clock.live.data.OUT/f"day_{d:02d}.npz") as f:m=f["state"]
            with np.load(ref.tape.OUT/f"day_{d:02d}.npz") as f:q=f["reference"];dm=f["change"];fix=f["directions"][:,2]
            prior=np.zeros(len(q))
            if d>=8:
                for mod in (ref.precision,ref.nonlinear,clock):
                    change=np.load(mod.OUT/f"oof_{d:02d}.npy");q+=change[:,None]
                prior=change[:,2]
            grid=np.load(base.CACHE/f"day_{d:02d}_time.npy").astype(int)
            y=np.load(base.CACHE/f"day_{d:02d}_y.npy",mmap_mode="r")
            controls=[dm,fix,prior]
            for head in parents:
                saved=np.load(head.OUT/f"oof_{d:02d}.npy") if d>=8 else np.zeros((len(m),3),np.float32)
                q+=saved[:,None]
                controls.append(saved[:,2])
            n_controls=3+len(controls)
            state=m
            if prepare_inputs is not None:
                raw=np.load(base.CACHE/f"day_{d:02d}_x.npy",mmap_mode="r")
                state=prepare_inputs(raw,grid,m,params)
            forecast=views(state,grid,params);z=np.column_stack((*controls,forecast))
            if d in (8,12):w={key:ref.fit(cal,arms.index(key[0]),key[1],n_controls) for key in panels}
            if d>=8:
                xx=np.column_stack((q.mean(1,dtype=float),z))
                for (k,h),rows in panels.items():
                    change=xx@w[k,h]
                    if keep_oof:predictions[d,k,h]=change.astype(q.dtype)
                    step=np.broadcast_to(change[:,None],q.shape[:2])
                    rows.append(ref.market.metric(q[...,h],y[...,h],step))
                if d==8:
                    for n in (333,2501):
                        prefix=m[:n] if prepare_inputs is None else prepare_inputs(raw[:n],grid[:n],m[:n],params)
                        np.testing.assert_allclose(forecast[:n],views(prefix,grid[:n],params),rtol=0,atol=1e-8)
            cal.append(clock.live.stat(q,y,z))
        history.append(v)
    report={f"{k}_{h}":dict(folds=[ref.market.gain(sum(v[:4])),ref.market.gain(sum(v[4:]))],days=[ref.market.gain(g) for g in v]) for (k,h),v in panels.items()}
    picks={}
    for h in targets:
        ok=[k for k in arms if min(report[f"{k}_{h}"]["folds"])>0 and sum(i>0 for i in report[f"{k}_{h}"]["days"])>=5]
        if check_leave_one_out:
            for k in arms:
                group=panels[k,h]
                drops=[ref.market.gain(sum(fold)-day) for fold in (group[:4],group[4:]) for day in fold]
                report[f"{k}_{h}"]["leave_one_date_out"]=drops
            ok=[k for k in ok if min(report[f"{k}_{h}"]["leave_one_date_out"])>0]
        if ok:picks[str(h)]=max(ok,key=lambda k:min(report[f"{k}_{h}"]["folds"]))
    (out/"rolling.json").write_text(json.dumps(dict(selected=picks,results=report),indent=2));print(picks,{k:v["folds"] for k,v in report.items()},flush=True)
    if not picks:return
    if keep_oof:
        for day in range(8,16):
            update=np.zeros((len(predictions[day,arms[0],targets[0]]),3),dtype=np.float32)
            for h,k in picks.items():update[:,int(h)]=predictions[day,k,int(h)]
            np.save(out/f"oof_{day:02d}.npy",update)
    w=np.zeros((n_controls+len(arms),3))
    for h,k in picks.items():w[:,int(h)]=ref.fit(cal,arms.index(k),int(h),n_controls)
    np.savez(out/"weights.npz",w=w,**train(history))
    outer=[]
    for d in range(16,20):
        grid,r,y=ref.tape.outer(d);raw=np.load(base.CACHE/f"day_{d:02d}_x.npy",mmap_mode="r")
        with np.load(ref.market.parent.OUT/"weights.npz") as f:_,av=ref.market.parent.state(raw,grid,f["code"],with_market=True)
        m=ref.market.market_state(av,grid);q,change=ref.latest(r["reference"],r,m)
        controls=[r["change"],r["directions"][:,2],change[:,2]]
        for head in parents:
            with np.load(head.OUT/"weights.npz") as pf:
                source=head.prepare_inputs(raw,grid,m,pf) if hasattr(head,"prepare_inputs") else m
                xx=np.column_stack((q.mean(1,dtype=float),*controls,head.views(source,grid,pf)))
                saved=(xx@pf["w"]).astype(q.dtype)
            q+=saved[:,None]
            controls.append(saved[:,2])
        with np.load(out/"weights.npz") as f:
            state=m if prepare_inputs is None else prepare_inputs(raw,grid,m,f)
            z=views(state,grid,f)
            xx=np.column_stack((q.mean(1,dtype=float),*controls,z))
            pp=q+(xx@f["w"]).astype(q.dtype)[:,None]
        outer.append((q,y,pp))
    score=ref.market.summary(outer)
    compared=parents[-1] if parents else clock
    assert score["rows"]==936250 and abs(score["baseline_score"]-json.loads((compared.OUT/"outer.json").read_text())["pooled"]["score"])<1e-8
    (out/"outer.json").write_text(json.dumps(dict(pooled=score,daily=[ref.market.summary([r]) for r in outer]),indent=2));print(score,flush=True)


if __name__=="__main__":run(OUT,ARMS,parameters,design)
