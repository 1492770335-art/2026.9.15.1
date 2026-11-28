"""Predict multi-horizon stock-relative payoffs through transitions between quiet and active book states."""
import json
import numpy as np
import experiment_dense_near_matched_spreads_v1 as source
import experiment_dense_shared_horizon_calibration_v1 as calibration

rel,REFERENCE=calibration.rel,calibration.REFERENCE
OUT=REFERENCE.parent/"dense_activity_transition_curve_v1"
ARMS=("single_regime","switching_regime","switching_increment")
HORIZONS=(1,4,12)


def state(raw,grid,code,market_state=False):
    impulses=source.sources(raw,grid,code).astype(float)
    parts=[]
    for a,b in source.sessions(grid):
        current=impulses[a:b]
        price=raw[a:b,:,0].astype(float)-.00136
        speed=source.smooth((abs(price)>.0001).astype(float),24)
        prior=source.smooth((abs(price)>.0001).astype(float),240)
        depths=np.tanh(.65*np.arcsinh(raw[a:b][:,:,[7,10,21,39,50,72]].astype(float)))
        levels=np.concatenate((current,depths-source.smooth(depths,32)),axis=2)
        common=levels.mean(1,keepdims=True)
        x=levels-common
        activity=np.tanh(3*(speed-.5*(prior+speed.mean(1,keepdims=True))))
        # A gate may change before the forecast expires; the transition model advances both products.
        local=x*activity[:,:,None]
        market=common*activity[:,:,None]
        z=np.concatenate((x,local,market),axis=2)
        if market_state:
            # Track average signed levels and how much of each level originates from more active stocks.
            m=np.concatenate((common[:,0],local.mean(1),market.mean(1)),axis=1)
            parts.append(m-source.smooth(m,320))
        else:
            parts.append(z-z.mean(1,keepdims=True))
    return np.concatenate(parts)


def collect():
    OUT.mkdir(exist_ok=True)
    code=source.codec()
    for d in range(16):
        def load(k):return np.load(rel.ref.base.CACHE/f"day_{d:02d}_{k}.npy",mmap_mode="r")
        grid=load("time").astype(int)
        z=state(load("x"),grid,code)
        y=load("y")[:,:,0].astype(float)
        y-=y.mean(1,keepdims=True)
        g=lag5=lag20=paid=0;count=0
        for a,b in source.sessions(grid):
            t=np.arange(a,b-25,4)
            x=z[t].reshape(-1,z.shape[-1])
            g+=x.T@x
            lag5+=x.T@z[t+5].reshape(x.shape)
            lag20+=x.T@z[t+20].reshape(x.shape)
            paid+=x.T@y[t].ravel()
            count+=len(x)
        np.savez(OUT/f"day_{d:02d}.npz",g=g/count,lag5=lag5/count,lag20=lag20/count,paid=paid/count)


def train(end):
    data=[]
    for d in range(end):
        with np.load(OUT/f"day_{d:02d}.npz") as f:data.append(tuple(f[k] for k in ("g","lag5","lag20","paid")))
    grams,l5,l20,prices=map(np.stack,zip(*data))
    n=len(prices[0]);models=[]
    for dim in (n//3,n):
        sl=slice(0,dim)
        g=grams[:,sl,sl].mean(0)
        sd=np.sqrt(g.diagonal()).clip(.008)
        eig,v=np.linalg.eigh(g/np.outer(sd,sd))
        root=v/(sd[:,None]*np.sqrt(eig.clip(0)+.18))
        halves=[]
        for side in (0,1):
            f=root.T@l5[side::2,sl,sl].mean(0)@root
            coarse=root.T@l20[side::2,sl,sl].mean(0)@root
            reward=root.T@prices[1-side::2,sl].mean(0)
            # Four adjacent five-tick payoffs cover each 20-tick subinterval.
            twenty=(np.eye(dim)+f+f@f+f@f@f)@reward
            sixty=(np.eye(dim)+coarse+coarse@coarse)@twenty
            out=np.zeros((n,3));out[sl]=root@np.column_stack((reward,twenty,sixty))
            halves.append(out)
        models.append(np.mean(halves,0))
    return np.stack((models[0],models[1],models[1]-models[0]),axis=1)


def records(p,y,z,h):
    return rel.relative.stats(p,y,z[...,h])


def run():
    code=source.codec()
    short=rel.rolling_short_changes()
    rows={(h,k):[] for h in range(3) for k in ARMS};history=[[] for _ in range(3)]
    for d in range(4,16):
        if d in (4,8,12):bank=train(d)
        p=np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else calibration.future.forecast(d)
        if d in short:p[...,0]+=short[d][:,None]
        def load(k):return np.load(rel.ref.base.CACHE/f"day_{d:02d}_{k}.npy",mmap_mode="r")
        raw,grid,y=load("x"),load("time").astype(int),load("y")
        x=state(raw,grid,code)
        z=np.einsum("tif,fah->tiah",x,bank)
        if d==8:np.testing.assert_allclose(x[:512],state(raw[:512],grid[:512],code),atol=1e-8,rtol=0)
        if d in (8,12):w={(h,k):rel.relative.fit(history[h],j,h) for h in range(3) for j,k in enumerate(ARMS)}
        for h in range(3):
            if d>=8:
                xx=rel.relative.features(p,z[...,h])
                for k in ARMS:rows[h,k].append(rel.ref.market.metric(p[:,:,h],y[:,:,h],xx@w[h,k]))
            history[h].append(records(p,y,z,h))
    report={str(h):{k:calibration.score(rows[h,k]) for k in ARMS} for h in range(3)}
    picks={}
    for h in range(3):
        r=report[str(h)]
        ok=[k for k,v in r.items() if min(v["folds"]+v["omit_one"])>0 and sum(j>0 for j in v["days"])>=5]
        if ok:picks[str(h)]=max(ok,key=lambda k:min(r[k]["folds"]))
    (OUT/"rolling.json").write_text(json.dumps(dict(selected=picks,results=report),indent=2))
    print(picks,{h:{k:v["folds"] for k,v in r.items()} for h,r in report.items()},flush=True)
    if picks:
        bank=train(16)
        w=np.zeros((6,3))
        for key,k in picks.items():w[:,int(key)]=rel.relative.fit(history[int(key)],ARMS.index(k),int(key))
        np.savez(OUT/"weights.npz",code=code,bank=bank,w=w)


def predict(raw,grid,p,params):
    state_values=state(raw,grid,params["code"])
    z=np.einsum("tsf,fah->tsah",state_values,params["bank"])
    q=p.copy()
    for h in range(3):
        q[:,:,h]+=(rel.relative.features(p,z[...,h])@params["w"][:,h]).astype(q.dtype)
    return q


if __name__=="__main__":
    collect()
    run()
