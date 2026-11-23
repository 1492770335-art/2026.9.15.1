"""Measure the online transition's incremental skill on settled forecast paths and return paths."""
import json
import numpy as np
import experiment_verified_near_carry_v1 as best
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

short=best.medium
online=short.online
OUT=ART/'transition_revision_skill_v1'
UNIT=short.CONFIG[0][3]


def calibration(z,prior,outcome,delay):
    n,stocks=z.shape
    if n<=delay:return np.zeros_like(z)
    num=np.zeros_like(z)
    den=np.zeros_like(z)
    origin=np.arange(n-delay)[:,None]
    mask=(origin+np.arange(stocks)[None,:])%5==0
    num[delay:]=mask*z[:-delay]*(outcome-prior[:-delay])
    den[delay:]=mask*z[:-delay]**2
    def rolling(v):
        cs=np.concatenate((np.zeros_like(v[:1]),v.cumsum(0,dtype=float)))
        end=np.arange(1,len(v)+1)
        return cs[end]-cs[(end-900).clip(0)]
    a,b=rolling(num),rolling(den)
    # One is the coefficient of the deployed unmodified transition difference.
    shared=(a.mean(1,keepdims=True)+.32)/(b.mean(1,keepdims=True)+.32)
    local=(a+.8*shared)/(b+.8)
    change=(.65*shared+.35*local-1.).clip(-1.5,1.5)
    q=.5*change*z
    return q-q.mean(1,keepdims=True)


def signal(raw,p,pr):
    v,x=short.state(p,5)
    g,r=online.observed(x,v,5)
    coef,=short.dynamic([(g,r)],[pr])
    w,_=pr
    count=max(1,len(p)-5)
    hist=[(g.sum(0)/count,r.sum(0)/count)]
    old=[];new=[]
    for start in range(0,len(x),128):
        t=np.arange(start,min(len(x),start+128))
        old.append(UNIT*short.evolution(x[t],[w[None]]*3,0,True))
        new.append(UNIT*short.evolution(x[t],[coef[t//online.BLOCK]]*3,0,True))
    a,b=map(np.concatenate,(old,new))
    a=a.astype(float)-a.mean(1,keepdims=True)
    b=b.astype(float)-b.mean(1,keepdims=True)
    z=b-a
    obs=UNIT*sum(v[k:k+max(0,len(p)-15),:,0] for k in (5,10,15))
    obs-=obs.mean(1,keepdims=True)
    price=(raw[:,:,0].astype(float)-.00136).cumsum(0)
    realized=.272164*(price[20:]-price[5:5+max(0,len(p)-20)])
    realized-=realized.mean(1,keepdims=True)
    f=calibration(z,a,obs,15)
    y=calibration(z,a,realized,20)
    return np.stack((z,f,y),2).astype('float32'),hist


def load():
    parent=best.load()
    with np.load(OUT/'model.npz') as st:
        pr=(st['w'],st['g'])
        weights=st['weights']
        arm=st['arms']
    def apply(raw,p):
        q=parent(raw,p)
        z,_=signal(raw,q,pr)
        out=q.copy()
        for i,k in enumerate(arm):
            if k>=0:out[:,:,i+1]+=weights[i,k]*z[:,:,k+1]
        return out
    return apply


def gain(rs):
    s=measure(rs)
    return np.subtract(s['candidate'][1:],s['base'][1:]).tolist()


def main():
    OUT.mkdir(exist_ok=True)
    history,records,folds,daily=[],[[],[]],[],[]
    for end in (8,10,12,14):
        pr=short.prior(history)[0] if history else None
        ws=np.stack([short.fit(s) for s in records]) if records[0] else None
        groups=[[[],[]],[[],[]]]
        for d in (end,end+1):
            raw=best.incumbent.segment.paths.book.rawday(d)
            for j,path in enumerate((online.best.refit.OUT/f'reference_{d}.npy',online.best.refit.core.OUT/f'pred_{d}.npy')):
                p=np.load(path,mmap_mode='r')
                y=labels(d).reshape(p.shape)
                if pr is None:
                    if not j:
                        _,_,hist=short.sample(p,best.SETTING)
                        history.append(hist[:1])
                    continue
                z,hist=signal(raw,p,pr)
                t=np.arange(d%4,len(p),4)
                tr=t[t<len(p)-60]
                if not j:
                    history.append(hist)
                    for i,rec in enumerate(records):
                        rec.append(short.stats(p[tr,:,i+1],y[tr,:,i+1],z[tr]))
                    if d==end:
                        zz,_=signal(raw[:221],p[:221],pr)
                        np.testing.assert_allclose(z[:221],zz,atol=3e-6,rtol=0)
                if ws is None:continue
                original=p[t].copy()
                delta=ws[0,0]*z[t,:,0]
                original[:,:,1]+=delta
                original[:,:,2]+=best.RATIO*delta
                for k,rs in enumerate(groups[j],1):
                    q=original.copy()
                    q[:,:,1:]+=z[t,:,k,None]*ws[:,k]
                    rs.append(tuple(a.reshape(-1,3) for a in (original,q,y[t])))
                if not j:daily.append([gain(rs[-1:]) for rs in groups[j]])
        if ws is not None:
            folds.append([[gain(rs) for rs in gs] for gs in groups])
            print(end+1,'by reference and settled observation',folds[-1],flush=True)
    g=np.array(folds)
    ok=(g[:,0].min(0)>0)&(g[:,1].mean(0)>0)&((np.array(daily)>0).sum(0)>=3)
    arms=np.where(ok[0]&(~ok[1]|(g[:,0,0].mean(0)>=g[:,0,1].mean(0))),0,np.where(ok[1],1,-1))
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=daily,arms=arms.tolist()),indent=2))
    if (arms>=0).any():
        w,g=short.prior(history)[0]
        np.savez(OUT/'model.npz',w=w,g=g,weights=np.stack([short.fit(s)[1:] for s in records]),arms=arms)
        base=json.loads((best.OUT/'reload_verification.json').read_text())['pooled']['score']
        for name in ('raw_replay','reload_verification'):
            res=online.best.roll.gls.best.walk.replay(load())
            if name=='reload_verification':
                last=json.loads((OUT/'raw_replay.json').read_text())
                np.testing.assert_allclose(res['pooled']['candidate'],last['pooled']['candidate'],atol=1e-10,rtol=0)
            (OUT/f'{name}.json').write_text(json.dumps(res,indent=2))
            print(name,res['pooled']['score'],flush=True)
            if res['pooled']['score']<=base:break


if __name__=='__main__':
    main()
