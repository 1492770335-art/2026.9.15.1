"""Rotate only the long-horizon stock component while preserving both official and distal Pearson denominators."""
import json
import numpy as np
import experiment_dense_distal_scenario_hedge_v1 as hedge
import experiment_dense_distal_overlap_gls_v1 as gls

panel,onset,rel,REFERENCE = hedge.panel,hedge.onset,hedge.rel,hedge.REFERENCE
OUT = REFERENCE.parent/"dense_distal_fixed_energy_rotation_v1"
ARMS=("direct_distal","near_scenario_neutral","stock_time_innovation")


def train(end):
    return np.column_stack((panel.train(end)[:,0],hedge.train(end),gls.train(end)[:,1]))


def remove(x,basis):
    gram=basis.swapaxes(1,2)@basis
    rhs=basis.swapaxes(1,2)@x
    return x-basis@(np.linalg.pinv(gram,hermitian=True,rcond=1e-11)@rhs)


def frame(raw,grid,p,bank):
    z=np.empty((*p.shape[:2],len(ARMS)))
    middle=bank[:,1:-1]
    for a,b,x,_ in panel.panels(raw,grid,p):
        vol=np.sqrt(.002+panel.smooth(np.tanh(raw[a:b,:,0].astype(float))**2,45))
        for t in range(0,b-a,128):
            u=x[t:t+128]
            near=u@middle[:,:hedge.HEDGES]
            risk=panel.centered(vol[t:t+len(u)])
            stressed=near.reshape(len(u),50,hedge.GROUPS,2).mean(2)*risk[:,:,None]
            z[a+t:a+t+len(u),:,0]=u@bank[:,0]
            z[a+t:a+t+len(u),:,1]=hedge.neutral(u,np.concatenate((u[:,:,:3],near,stressed),axis=2))@middle[:,-1]
            z[a+t:a+t+len(u),:,2]=hedge.neutral(u,u[:,:,:3])@bank[:,-1]
    p=panel.centered(p.astype(float))
    r=remove(p[:,:,2:3],p[:,:,:2])
    z=remove(panel.centered(z),p)
    sd=np.linalg.norm(z,axis=1,keepdims=True)
    valid=(sd>1e-8)
    risk=np.linalg.norm(r,axis=1,keepdims=True)
    return r*valid,valid*risk*z/sd.clip(1e-8)


def delta(r,z,theta):
    return r*(-2*np.sin(.5*theta)**2)+z*np.sin(theta)


def statistics(p,y,r,z):
    actual=panel.centered(y[...,2].astype(float))
    far=panel.centered(y[...,2]-panel.ratio()*y[...,1])
    return np.stack([np.stack([(v*label[:,:,None]).mean((0,1)) for v in (r,z)]) for label in (actual,far)])


def fit(rows):
    c=np.stack(rows)
    angles=np.arange(-.30,.301,.01)
    curve=(c[:,:,:,None,:]*np.stack((-2*np.sin(.5*angles)**2,np.sin(angles)))[None,None,:,:,None]).sum(2)
    # Both numerator gains must replicate. Shrinking by the cross-date standard
    # error also reduces rotation when the observed improvement comes from one date.
    mean=curve.mean(0)
    error=curve.std(0)/np.sqrt(len(c))
    value=(mean-.65*error).min(0)
    result=.6*angles[np.argmax(value,axis=0)]
    result[abs(result)<1e-8]=0.
    return result


def verify_geometry(p,d):
    u=p.astype(float)
    np.testing.assert_allclose(d.mean(1),0.,atol=2e-8)
    np.testing.assert_allclose((u[...,1:2]*d).mean(1),0.,atol=2e-8)
    np.testing.assert_allclose(((2*u[...,2:3]+d)*d).mean(1),0.,atol=3e-8)


def predict(raw,grid,p,params):
    r,z=frame(raw,grid,p,params["bank"])
    d=delta(r,z,params["angles"])
    verify_geometry(p,d)
    out=p.copy()
    out[...,2]+=d[:,:,int(params["selected"])].astype(out.dtype)
    return out


def rolling():
    OUT.mkdir(exist_ok=True)
    hist=[]; cal=[]
    rows={k:[[],[]] for k in ARMS}
    decisions={}
    for day in range(6,16):
        if day in (6,8,12):
            bank=train(day)
        p=panel.load_p(day)
        raw,grid,y=[onset.settled.load(day,k) for k in ("x","time","y")]
        r,z=frame(raw,grid,p,bank)
        if day in (8,12):
            angles=fit(hist)
            decisions[str(day+1)]=angles.tolist()
            beta3,beta2=panel.calibration(cal)
            n=2453
            rr,zz=frame(raw[:n],grid[:n],p[:n],bank)
            np.testing.assert_allclose(delta(rr,zz,angles),delta(r[:n],z[:n],angles),atol=2e-7)
        if day>=8:
            d=delta(r,z,angles)
            verify_geometry(p,d)
            f=panel.centered(beta3*p[...,2].astype(float)-panel.ratio()*beta2*p[...,1])
            yfar=panel.centered(y[...,2].astype(float)-panel.ratio()*y[...,1])
            for j,k in enumerate(ARMS):
                rows[k][0].append(rel.ref.market.metric(p[...,2],y[...,2],d[:,:,j]))
                rows[k][1].append(rel.ref.market.metric(f,yfar,beta3*d[:,:,j]))
        hist.append(statistics(p,y,r,z))
        cal.append(panel.phase_stats(p,y))
    report={k:dict(full=panel.score(g),distal=panel.score(f)) for k,(g,f) in rows.items()}
    ok=[k for k,v in report.items() if min(v["full"]["folds"]+v["full"]["omit_one"]+v["distal"]["folds"])>0
        and sum(z>0 for z in v["full"]["days"])>=5]
    pick=max(ok,key=lambda k:min(report[k]["full"]["folds"])) if ok else None
    angles=fit(hist)
    if pick and angles[ARMS.index(pick)]==0:
        pick=None
    (OUT/"rolling.json").write_text(json.dumps(dict(selected={"2":pick} if pick else {},results=report,
                                                    fitting_angles=decisions,final_angles=angles.tolist()),indent=2))
    if pick:
        np.savez(OUT/"weights.npz",bank=train(16),selected=np.array(ARMS.index(pick)),angles=angles)
    print(pick,{k:[v["full"]["folds"],v["distal"]["folds"]] for k,v in report.items()},flush=True)


if __name__=="__main__":
    rolling()
