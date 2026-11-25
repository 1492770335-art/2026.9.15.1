"""Screen stock-level additions against rolling Y3 and the stock-relative terminal twenty-tick return."""
import json
import numpy as np
import y3_overshoot_agreement_references as history
import cache_y3_market_agreement_reference as cache

trial=history.trial


def changes(z,p,w):
    return list(np.moveaxis(np.concatenate((trial.clock.center(p),z),2)@w,2,0))


def record(z,p,y,day):
    t=np.arange(90+day%7,len(y)-60,7)
    x=np.concatenate((trial.clock.center(p[t]),z[t]),2).reshape(len(t)*50,-1)
    q=trial.clock.center(p[t,:,2]).ravel()
    target=trial.clock.center(y[t,:,2]).ravel()
    beta=q@target/max(1e-7,q@q)
    return x.T@x/len(x),x.T@(target/beta.clip(.4,2.)-q)/len(x)


def projected_readouts(rows,specs):
    gs,bs=[np.stack(a).astype(float) for a in zip(*rows)]
    gram=gs.mean(0);heads=[]
    for cols,n,weight in specs:
        basis=np.eye(len(gram))[:,cols]
        basis[:n]=-trial.curve.solve(gram[:n,:n],gram[:n,cols],.055)
        days=bs@basis
        mu=days.mean(0)
        mu*=abs(mu)/(abs(mu)+days.std(0)/np.sqrt(len(rows))+1e-7)
        heads.append(weight*(basis@trial.curve.solve(basis.T@gram@basis,mu,.95)))
    return np.column_stack(heads).astype('float32')


def run(mod, references=history.references, outer=cache.outer):
    mod.OUT.mkdir(exist_ok=True)
    rows=[];folds=[];daily=[]
    coeff=trial.curve.settled.historical_geometry()[1];ratio=coeff[2,1]/coeff[1,1]
    for d in range(4,16):
        raw=trial.clock.book.rawday(d);y=trial.clock.labels(d).reshape(-1,50,3)
        refs=references(d,raw) if d>=8 else [history.market.prior.path.early.load(d,raw)]
        if d>=8 and d%2==0:w=mod.fit(rows);groups=[[],[]]
        rec=[]
        for j,p in enumerate(refs):
            z=mod.features(raw,p)
            rec.append(record(z,p,y,d))
            if d>=8:
                ds=changes(z,p,w)
                if d==8 and j==0:
                    zz=mod.features(raw[:311],p[:311])
                    for a,b in zip(ds,changes(zz,p[:311],w)):
                        np.testing.assert_allclose(a[:311],b,rtol=0,atol=4e-6)
                t=np.arange(len(raw)-60)
                q=np.zeros_like(p[t]);truth=np.zeros_like(q)
                q[:,:,2]=trial.clock.center(p[t,:,2]-ratio*p[t,:,1])
                truth[:,:,2]=ratio*trial.clock.center(y[t+40,:,1])
                stat=trial.scoring.audit.blockstats
                g=[stat(p,y,ds).sum(0),stat(q,truth,[a[t] for a in ds]).sum(0)]
                groups[j].append(g)
                if j==0:daily.append(trial.scoring.audit.pooled(g[0]))
        rows.append([np.mean(v,0) for v in zip(*rec)])
        if d>=8 and d%2:folds.append([[trial.scoring.audit.pooled(g) for g in np.sum(c,axis=0)] for c in groups])
    f=np.array(folds);d=np.array(daily)
    good=[]
    for j in mod.CANDIDATES:
        v=f[:,:,0,j]
        if ((v>0).sum(0).min()>=3 and v.mean(0).min()>0 and v[-1].min()>0 and (d[:,j]>0).sum()>=5
            and f[:,:,1,j].mean(0).min()>0 and all((v-f[:,:,0,k]).mean(0).min()>0 for k in mod.CONTROLS)):
            good.append(j)
    pick=max(good,key=lambda j:f[:,0,0,j].mean()) if good else None
    (mod.OUT/'results.json').write_text(json.dumps(dict(selected=pick,arms=mod.ARMS,folds=f.tolist(),daily=d.tolist()),indent=2))
    print(mod.OUT.name,'selected',pick,flush=True)
    if pick is not None:
        np.savez(mod.OUT/'model.npz',pick=pick,w=mod.fit(rows))
        outer(mod)


def correction(mod):
    with np.load(mod.OUT/'model.npz') as f:w,pick=f['w'],int(f['pick'])
    return lambda raw,p:changes(mod.features(raw,p),p,w)[pick]


def layer(mod,parent=history.incumbent):
    predict,update=parent.layer(),correction(mod)
    def apply(raw,p):
        q=predict(raw,p);q[:,:,2]+=update(raw,q)
        return q
    return apply
