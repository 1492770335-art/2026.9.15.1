"""Estimate the common rotation angle from settled cash increments of previously issued stock portfolios."""
import json
import numpy as np
import experiment_dense_distal_fixed_energy_rotation_v1 as base

onset,rel,REFERENCE,panel=base.onset,base.rel,base.REFERENCE,base.panel
OUT=REFERENCE.parent/"dense_distal_online_angles_v1"
ARMS=("settled_full_and_distal","agreement_between_stock_halves","agreement_with_historical_angle")
GROUPS=np.array_split(np.random.default_rng(38503).permutation(50),2)
BLOCK=40


def estimates(g):
    # The accounting blocks do not overlap. Each block's
    # cash increments enter once, even though the original forecasts overlap.
    results=np.zeros_like(g)
    mean=np.zeros_like(g[0]);sq=mean.copy();weight=weight2=0.
    decay=np.exp(-BLOCK/520)
    for i,block in enumerate(g):
        # The angle for a block sees only earlier *completed* blocks.
        if weight>2:
            avg=mean/weight
            se=np.sqrt((sq/weight-avg**2).clip(1e-12)*weight2/weight**2+1e-12)
            results[i]=np.tanh(.45*avg/se)*min(1.,weight/8)
        mean=decay*mean+block
        sq=decay*sq+block*block
        weight=decay*weight+1
        weight2=decay**2*weight2+1
    return results


def agree(a,b):
    return np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0.)


def angle_paths(raw,grid,z,prior):
    with np.load(onset.settled.GEO) as f:
        k,coef,row=float(f["k"]),f["coeff"][2,1],f["row"]
    angles=np.zeros((len(raw),3))
    for a,b in rel.fields.parent.sessions(grid):
        data=raw[a:b]
        cash=np.log1p(k*(data[:,:,0].astype(float)-.00136))*coef/k
        valid=~np.all(data==row,axis=2)
        cash=(cash-(cash*valid).sum(1,keepdims=True)/valid.sum(1,keepdims=True).clip(1))*valid
        vec=z[a:b]
        prefix=np.vstack((np.zeros(50),vec.cumsum(0)))
        t=np.arange(b-a)
        pnl=[]
        for left in (1,21):
            position=(prefix[(t-left+1).clip(0)]-prefix[(t-60).clip(0)])/(61-left)
            pnl.append(np.stack([(position[:,ids]*cash[:,ids]).mean(1) for ids in GROUPS],axis=1))
        pnl=np.stack(pnl,axis=1)
        block=np.add.reduceat(pnl,np.arange(0,len(t),BLOCK),axis=0)
        # The all-stock and two independently observed baskets use the same accounting clock.
        confidence=estimates(np.concatenate((block,block.mean(2,keepdims=True)),axis=2))[t//BLOCK]
        pooled=agree(confidence[:,0,2],confidence[:,1,2])
        both=agree(agree(confidence[:,0,0],confidence[:,1,0]),agree(confidence[:,0,1],confidence[:,1,1]))
        aligned=agree(pooled,np.full(len(t),np.sign(prior)))
        angles[a:b]=np.column_stack((.20*pooled,.20*both,prior*abs(aligned)))
    return angles


def infer(raw,grid,p,bank,prior):
    r,z=base.frame(raw,grid,p,bank)
    # This is the hedged book direction; the estimand is its rotation payoff.
    r,z=r[:,:,1:2],z[:,:,1:2]
    angles=angle_paths(raw,grid,z[:,:,0],prior)
    d=base.delta(r,z,angles[:,None,:])
    base.verify_geometry(p,d)
    return d,r,z


def predict(raw,grid,p,params):
    d,_,_=infer(raw,grid,p,params["bank"],float(params["prior"]))
    q=p.copy()
    q[...,2]+=d[:,:,int(params["selected"])].astype(q.dtype)
    return q


def rolling():
    OUT.mkdir(exist_ok=True)
    hist=[];cal=[];rows={k:[[],[]] for k in ARMS}
    activity=[]
    for day in range(6,16):
        if day in (6,8,12):
            bank=base.train(day)
        raw,grid,y=[onset.settled.load(day,k) for k in ("x","time","y")]
        p=panel.load_p(day)
        if day in (6,8,12):
            prior=0. if not hist else base.fit(hist)[0]
        d,r,z=infer(raw,grid,p,bank,prior)
        if day in (8,12):
            beta3,beta2=panel.calibration(cal)
            n=2453
            partial,_,_=infer(raw[:n],grid[:n],p[:n],bank,prior)
            np.testing.assert_allclose(d[:n],partial,atol=2e-7)
        if day>=8:
            far=panel.centered(y[...,2].astype(float)-panel.ratio()*y[...,1])
            f=panel.centered(beta3*p[...,2].astype(float)-panel.ratio()*beta2*p[...,1])
            for j,k in enumerate(ARMS):
                rows[k][0].append(rel.ref.market.metric(p[...,2],y[...,2],d[:,:,j]))
                rows[k][1].append(rel.ref.market.metric(f,far,beta3*d[:,:,j]))
            activity.append((np.abs(d).sum(1)>0).mean(0).tolist())
        hist.append(base.statistics(p,y,r,z))
        cal.append(panel.phase_stats(p,y))
    report={k:dict(full=panel.score(g),distal=panel.score(f)) for k,(g,f) in rows.items()}
    ok=[k for k,v in report.items() if min(v["full"]["folds"]+v["full"]["omit_one"]+v["distal"]["folds"])>0
        and sum(z>0 for z in v["full"]["days"])>=5]
    pick=max(ok,key=lambda k:min(report[k]["full"]["folds"])) if ok else None
    prior=base.fit(hist)[0]
    if pick==ARMS[2] and prior==0:
        pick=None
    (OUT/"rolling.json").write_text(json.dumps(dict(selected={"2":pick} if pick else {},results=report,
                                                    fraction_active=activity,final_prior=prior),indent=2))
    if pick:
        np.savez(OUT/"weights.npz",bank=base.train(16),selected=np.array(ARMS.index(pick)),prior=prior)
    print(pick,{k:[v["full"]["folds"],v["distal"]["folds"]] for k,v in report.items()},flush=True)


if __name__=="__main__":
    rolling()
