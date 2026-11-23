"""Compose historical short-forecast transitions to estimate longer disjoint return sums."""
import json
import numpy as np
import experiment_online_forecast_transitions_v1 as online
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'horizon_composition_v1'
CONFIG=((0,5,4,.272164/.558942),(1,20,3,.160943/.272164))


def state(p,h):
    v=p.astype(float)-p.mean(1,keepdims=True)
    v=v.clip(-3,3)
    t=np.arange(len(p))
    return v,np.concatenate([v[(t-j*h).clip(0)] for j in range(3)],2)


def sample(p,setting):
    k,h,n,unit=setting
    v,x=state(p,h)
    pairs=[online.observed(x,v,h*j) for j in range(1,n)]
    hist=[(g.sum(0)/(len(p)-h*j),r.sum(0)/(len(p)-h*j)) for j,(g,r) in enumerate(pairs,1)]
    return x,pairs,hist


def prior(hist):
    out=[]
    for earlier in zip(*hist):
        gs,rs=map(np.stack,zip(*earlier))
        g=gs.mean(0);r=rs.mean(0)
        g=.65*g.mean(0)+.35*g;r=.65*r.mean(0)+.35*r
        penalty=.22*np.eye(9)[None]*g.diagonal(axis1=1,axis2=2).clip(.00025)[:,None,:]
        out.append((np.linalg.solve(g+penalty,r),g+penalty))
    return out


def dynamic(pairs,pr):
    out=[]
    for (g,r),(w,reference) in zip(pairs,pr):
        gram=online.past(g,56);err=online.past(r-g@w,56)
        ridge=360*reference+np.eye(9)*.3
        shared=np.linalg.solve(gram.mean(1)+ridge.mean(0),err.mean(1))
        local=np.linalg.solve(gram+ridge,err-gram@shared[:,None])
        out.append(w[None]+shared[:,None]+.60*local)
    return out


def evolution(x,matrices,k,iterate):
    current=x
    pred=np.zeros(x.shape[:2],np.float32)
    for i in range(len(matrices)):
        coef=matrices[0] if iterate else matrices[i]
        new=(current[:,:,:,None]*coef).sum(2).clip(-3,3)
        pred+=new[:,:,k]
        if iterate:
            current=np.concatenate((new,current[:,:,:6]),axis=2)
    return pred


def features(x,pairs,pr,setting):
    k,h,n,unit=setting
    adjusted=dynamic(pairs,pr)
    rows=[]
    for start in range(0,len(x),144):
        t=np.arange(start,min(len(x),start+144));b=t//online.BLOCK
        local=[v[b] for v in adjusted];old=[w[None] for w,g in pr]
        z=[]
        for it in (False,True):
            diff=unit*(evolution(x[t],local,k,it)-evolution(x[t],old,k,it))
            z.append(diff-diff.mean(1,keepdims=True))
        rows.append(np.stack(z,axis=2))
    return np.concatenate(rows)


def stats(p,y,x):
    a=p.astype(float);b=y.astype(float)
    a-=a.mean();b-=b.mean()
    slope=((a*b).mean()/(a*a).mean()).clip(.35,3)
    residual=b/slope-a
    residual-=residual.mean(1,keepdims=True)
    return (x*x).mean((0,1)),(x*residual[:,:,None]).mean((0,1))


def fit(hist):
    g,b=map(np.stack,zip(*hist))
    reg=.85*g.mean(0).clip(.00002)
    daily=b/(g+reg)
    w=.65*(.65*b.mean(0)/(g.mean(0)+reg)+.35*np.median(daily,0))*abs(np.sign(daily).mean(0))
    return w.clip(-.65,.65)


def load(carry_medium=0.0, future_medium=None):
    parent=online.best.load();st=np.load(OUT/'model.npz')
    arms=st['arms'];ws=st['ws']
    prs=[list(zip(st[f'w{i}'],st[f'g{i}'])) for i in range(2)];st.close()
    def predict(raw,p):
        a=parent(raw,p);q=a.copy()
        for i,setting in enumerate(CONFIG):
            if arms[i]<0:continue
            x,pairs,_=sample(a,setting)
            z=features(x,pairs,prs[i],setting)
            delta=ws[i,arms[i]]*z[:,:,arms[i]]
            q[:,:,i+1]+=delta
            if i==0 and carry_medium:
                q[:,:,2]+=carry_medium*delta
                if future_medium is not None:
                    q[:,:,2]+=carry_medium*future_medium(delta)
        return q
    return predict


def main():
    OUT.mkdir(exist_ok=True)
    history=[[],[]];rows=[[],[]];folds=[];daily=[]
    for end in (8,10,12,14):
        priors=[prior(v) for v in history] if history[0] else []
        ws=[fit(v) for v in rows] if rows[0] else []
        groups=[[[],[]] for _ in range(2)]
        for d in (end,end+1):
            for j,path in enumerate((online.best.refit.OUT/f'reference_{d}.npy',online.best.refit.core.OUT/f'pred_{d}.npy')):
                p=np.load(path,mmap_mode='r');y=labels(d).reshape(p.shape)
                t=np.arange(d%4,len(p),4);ts=t[t<len(p)-60]
                updates=[p[t].copy(),p[t].copy()]
                for i,setting in enumerate(CONFIG):
                    x,obs,m=sample(p,setting)
                    if j==0:history[i].append(m)
                    if not priors:continue
                    z=features(x,obs,priors[i],setting)
                    if j==0:
                        rows[i].append(stats(p[ts,:,i+1],y[ts,:,i+1],z[ts]))
                        if d==end:
                            xx,oo,_=sample(p[:193],setting)
                            zz=features(xx,oo,priors[i],setting)
                            np.testing.assert_allclose(zz,z[:193],rtol=0,atol=2e-6)
                    if ws:
                        for k,q in enumerate(updates):q[:,:,i+1]+=ws[i][k]*z[t,:,k]
                if ws:
                    day=[]
                    for k,q in enumerate(updates):
                        row=tuple(a.reshape(-1,3) for a in (p[t],q,y[t]))
                        groups[j][k].append(row)
                        s=measure([row]);day.append(np.subtract(s['candidate'][1:],s['base'][1:]))
                    if j==0:daily.append(day)
        if ws:
            folds.append([[measure(s) for s in pair] for pair in groups])
            print(end+1,[[np.subtract(s['candidate'][1:],s['base'][1:]).tolist() for s in pair] for pair in folds[-1]],flush=True)
    g=np.array([[[np.subtract(s['candidate'][1:],s['base'][1:]) for s in pair] for pair in f] for f in folds])
    ok=(g[:,0].min(0)>1e-6)&(g[:,1].mean(0)>0)&((np.array(daily)>1e-6).sum(0)>=3)
    iterate=ok[1]&(~ok[0]|((g[:,0,1]-g[:,0,0]).mean(0)>0))
    arms=np.where(iterate,1,np.where(ok[0],0,-1))
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=np.array(daily).tolist(),arms=arms.tolist(),
                    arm_names=['direct','iterated'],fold_starts=[13,15]),indent=2))
    print('Y2,Y3',arms,flush=True)
    if (arms>=0).any():
        model=dict(arms=arms,ws=np.stack([fit(v) for v in rows]))
        for i,hist in enumerate(history):
            w,g=zip(*prior(hist));model[f'w{i}']=np.stack(w);model[f'g{i}']=np.stack(g)
        np.savez(OUT/'model.npz',**model)
        replay=online.best.roll.gls.best.walk.replay
        result=replay(load());(OUT/'raw_replay.json').write_text(json.dumps(result,indent=2))
        prev=json.loads((online.best.OUT/'reload_verification.json').read_text())['pooled']
        print('official',result['pooled']['score'],result['pooled']['score']-prev['score'],flush=True)
        if result['pooled']['score']>prev['score']:
            check=replay(load())
            np.testing.assert_allclose(check['pooled']['candidate'],result['pooled']['candidate'],rtol=0,atol=1e-10)
            (OUT/'reload_verification.json').write_text(json.dumps(check,indent=2))


if __name__=='__main__':main()
