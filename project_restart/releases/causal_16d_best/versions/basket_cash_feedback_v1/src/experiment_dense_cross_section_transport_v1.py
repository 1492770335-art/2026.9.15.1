"""Split historical book moves into distribution shifts and changes to the stock's rank within that distribution."""
import json
import numpy as np
import experiment_dense_replicated_stock_portfolios_v1 as previous

panel,onset,rel,REFERENCE=previous.panel,previous.onset,previous.rel,previous.REFERENCE
OUT=REFERENCE.parent/"dense_cross_section_transport_v1"
ARMS=("common_distribution","relative_rank_changes","combined")
LAGS=(8,40,120)


def states(v):
    def vals(ids):
        return np.tanh(.7*np.arcsinh(v[:,:,ids].astype(float)))
    imbalance=vals([26,41,54])-vals([38,42,55])
    return np.concatenate((imbalance,vals([0,11,64])),axis=2)


def design(raw,grid,p):
    for a,b in rel.fields.parent.sessions(grid):
        u=panel.smooth(states(raw[a:b]),4)
        t=np.arange(len(u))
        order=np.argsort(u,axis=1,kind="stable")
        rank=np.argsort(order,axis=1)
        sorted_u=np.take_along_axis(u,order,axis=1)
        f=p[a:b].astype(float)
        market=[f[:,:,1:].mean(1)]
        local=[panel.centered(f)]
        for lag in LAGS:
            old=(t-lag).clip(0)
            shift=sorted_u-sorted_u[old]
            common=np.take_along_axis(shift,rank,axis=1)
            exchange=u-u[old]-common
            r=(rank-24.5)/25.
            # Different parts of the cross section can move in opposite directions.
            lower=shift[:,:12].mean(1)
            mid=shift[:,12:38].mean(1)
            upper=shift[:,38:].mean(1)
            market.extend((mid,lower+upper-2*mid,shift.mean(1)*np.mean(np.sign(shift),axis=1)**2,
                           np.mean(exchange*np.abs(r),axis=1),np.mean(exchange*np.tanh(3*u[old]*u),axis=1)))
            local.extend((common,exchange,exchange*r**2,exchange*np.tanh(3*common*u[old]),
                          common*np.tanh(3*exchange*u[old])))
        yield a,b,np.column_stack(market),panel.centered(np.concatenate(local,axis=2))


def read_p(d):
    return np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else panel.load_p(d)


def collect(designer=design, dest=OUT):
    dest.mkdir(exist_ok=True)
    for d in range(4,16):
        p=read_p(d)
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        f=p[...,2].astype(float)
        truth=y[...,2].astype(float)
        slope=np.cov(f.ravel(),truth.ravel(),bias=True)[0,1]/f.var()
        err=truth/slope-f
        far=truth-panel.ratio()*y[...,1]
        total=[0.,0.];n=[0,0]
        for a,b,market,local in designer(raw,grid,p):
            t=np.arange(d%5,b-a,5)
            label=np.stack((err[a+t],far[a+t]/slope),axis=2)
            for j,(v,z) in enumerate(((market[t],label.mean(1)),(local[t],panel.centered(label)))):
                u=np.concatenate((v,z),axis=-1).reshape(-1,v.shape[-1]+2)
                u-=u.mean(0)
                total[j]+=u.T@u;n[j]+=len(u)
        np.savez(dest/f"day_{d:02d}.npz",market=total[0]/n[0],relative=total[1]/n[1])


def fit(rows,controls):
    g=rows.mean(0);dim=len(g)-2
    tr=np.eye(dim)[:,controls:]
    c=g[:controls,:controls]
    tr[:controls]=-np.linalg.solve(c+.025*np.diag(c.diagonal().clip(1e-8)),g[:controls,controls:dim])
    m=tr.T@g[:dim,:dim]@tr
    white=previous.whitening(m,.85)
    axes=tr@white
    rhs=axes.T@rows[:,:dim,-2:]
    mean=rhs.mean(0)
    stable=mean*mean**2/(mean**2+rhs.var(0)/len(rhs)+1e-9)
    # Each supported direction must price both the full error and the distant interval.
    use=np.where(stable[:,0]*stable[:,1]>0,np.sign(stable[:,0])*np.minimum(abs(stable[:,0]),abs(stable[:,1])),0)
    return axes@use


def train(end, dest=OUT):
    days=[]
    for d in range(4,end):
        with np.load(dest/f"day_{d:02d}.npz") as f:
            days.append((f["market"],f["relative"]))
    c,r=[np.stack(col) for col in zip(*days)]
    return np.r_[fit(c,2),fit(r,3)]


def signals(raw,grid,p,bank,designer=design):
    out=np.zeros((*p.shape[:2],3))
    for a,b,market,local in designer(raw,grid,p):
        common=market@bank[:market.shape[1]]
        specific=local@bank[market.shape[1]:]
        out[a:b,:,0]=common[:,None]
        out[a:b,:,1]=specific
        out[a:b,:,2]=common[:,None]+specific
    return out


def summarize(rows):
    return {name:panel.score(v) for name,v in zip(("full","distal"),rows)}


def sizes(records):
    # Select a common validation-prediction scale on earlier dates; cost in variance is
    # included in both exact correlations. No sign reversal or held-out-date calibration.
    strengths=np.array([0.,.1,.25,.5,.8,1.2])
    selected=[]
    for j in range(len(records[0])):
        gains=[]
        for s in strengths:
            scale=np.diag([1.,1.,1.,s])
            day=np.array([[rel.ref.market.gain(scale@v@scale) for v in r[j]] for r in records])
            gains.append(np.min(day.mean(0)-.4*day.std(0)/len(day)**.5))
        selected.append(strengths[np.argmax(gains)])
    return np.array(selected)


def evaluate(p,y,z,b3,b2):
    far=y[...,2].astype(float)-panel.ratio()*y[...,1]
    pf=b3*p[...,2].astype(float)-panel.ratio()*b2*p[...,1]
    return [[rel.ref.market.metric(p[...,2],y[...,2],z[...,j]),rel.ref.market.metric(pf,far,b3*z[...,j])]
            for j in range(z.shape[-1])]


def rolling(model):
    cal=[panel.phase_stats(read_p(d),onset.settled.load(d,"y")) for d in (4,5)]
    hist=[];rows={k:[[],[]] for k in model.ARMS}
    for d in range(6,16):
        if d in (6,8,12):
            bank=model.train(d)
            b3,b2=panel.calibration(cal)
            if d>=8:
                strength=sizes(hist)
        p=read_p(d)
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        z=model.signals(raw,grid,p,bank)
        if d==8:
            for n in (240,2451):
                np.testing.assert_allclose(z[:n],model.signals(raw[:n],grid[:n],p[:n],bank),rtol=0,atol=2e-7)
        rec=evaluate(p,y,z,b3,b2)
        if d>=8:
            for j,key in enumerate(model.ARMS):
                scale=np.diag([1.,1.,1.,strength[j]])
                for dest,v in zip(rows[key],rec[j]):
                    dest.append(scale@v@scale)
        hist.append(rec);cal.append(panel.phase_stats(p,y))
    report={key:summarize(r) for key,r in rows.items()}
    final=sizes(hist)
    good=[k for j,k in enumerate(model.ARMS) if min(report[k]["full"]["folds"]+report[k]["full"]["omit_one"]+report[k]["distal"]["folds"])>1e-12
          and sum(v>1e-12 for v in report[k]["full"]["days"])>=5 and final[j]>0]
    pick=max(good,key=lambda k:min(report[k]["full"]["folds"])) if good else None
    (model.OUT/"rolling.json").write_text(json.dumps(dict(selected={"2":pick} if pick else {},results=report,
            final_strengths=final.tolist()),indent=2))
    if pick:
        j=model.ARMS.index(pick)
        np.savez(model.OUT/"weights.npz",bank=model.train(16),choice=j,strength=final[j])
    print(pick,{k:[r["full"]["folds"],r["distal"]["folds"]] for k,r in report.items()},flush=True)


def predict(raw,grid,p,params):
    z=signals(raw,grid,p,params["bank"])
    q=p.copy()
    q[...,2]+=(params["strength"]*z[:,:,int(params["choice"])]).astype(q.dtype)
    return q


if __name__=="__main__":
    import sys
    collect();rolling(sys.modules[__name__])
