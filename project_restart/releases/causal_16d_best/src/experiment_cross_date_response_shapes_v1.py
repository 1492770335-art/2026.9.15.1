"""Estimate reproducible five-tick response shapes for order book and market features."""
import json
import numpy as np
import experiment_transition_revision_skill_v1 as best
import experiment_rolling_medium_reference_v1 as earlier
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'cross_date_response_shapes_v1'
book=best.best.incumbent.segment.paths.book
SCALE=np.array([.558942,.272164,.160943])
ENDS=(1,4,12)


def features(raw,t):
    state,v=book.state(raw)
    z=np.tanh(state[t])
    m=z.mean(1,keepdims=True)
    s=z.std(1,keepdims=True)
    common=np.concatenate((m,m*s,np.tanh(2*z).mean(1,keepdims=True),m*np.exp(-t[:,None,None]/350)),axis=2)
    a=z-m
    relative=np.concatenate((a,a*np.log(v[t,:,None].clip(.4,4)),a*m),axis=2)
    return relative,common


def record(day):
    raw=book.rawday(day)
    t=np.arange(day%5,len(raw)-61,5)
    y=labels(day).reshape(-1,50,3)[t]/SCALE
    price=(raw[:,:,0].astype(float)-.00136).cumsum(0)
    edges=np.stack([price[t+5*k] for k in range(13)],axis=2)
    inc=np.diff(edges,axis=2)
    # Preserve the official cumulative payoff for each of the three horizons.
    for start,end,j in ((0,1,0),(1,4,1),(4,12,2)):
        expected=y[:,:,j]-(y[:,:,j-1] if j else 0)
        inc[:,:,start:end]+=(expected-inc[:,:,start:end].sum(2))[:,:,None]/(end-start)
    np.testing.assert_allclose(inc.cumsum(2)[:,:,[0,3,11]],y,atol=1e-7,rtol=0)
    mean=inc.mean(1,keepdims=True)
    out=[]
    for x,pay in zip(features(raw,t),(inc-mean,mean)):
        a=x.reshape(-1,x.shape[-1]).astype(float)
        b=pay.reshape(-1,12)
        a-=a.mean(0)
        b-=b.mean(0)
        out.append((a.T@a/len(a),a.T@b/len(a),x.mean((0,1)).astype(float)))
    return out


def powers(matrix,p):
    eig,vec=np.linalg.eigh(matrix)
    return (vec*eig.clip(1e-10)**p)@vec.T


def mapping(daily,gram):
    # Whiten feature directions. Estimate the second moment of effects across leads
    # and subtract uncertainty of the estimated mean using independent trading dates.
    x=powers(gram,.5)@daily
    n=len(x)
    avg=x.mean(0)
    dev=x-avg
    moment=avg.T@avg
    noise=np.einsum('dfi,dfj->ij',dev,dev)/((n-1)*n)
    reg=.07*np.trace(moment)/len(moment)+1e-8
    root=powers(moment+np.eye(len(moment))*reg,.5)
    inv=powers(moment+np.eye(len(moment))*reg,-.5)
    reliability=inv@(moment-noise)@inv
    eig,vec=np.linalg.eigh(reliability)
    return inv@((vec*eig.clip(0,1))@vec.T)@root


def fit(records):
    fitted=[]
    bands=np.zeros((12,3))
    for k,(start,stop) in enumerate(((0,1),(1,4),(4,12))):
        bands[start:stop,k]=1/np.sqrt(stop-start)
    recover=np.array([1,np.sqrt(3),np.sqrt(8)])
    for rec in zip(*records):
        grams,rhs,centers=map(np.stack,zip(*rec))
        gram=grams.mean(0)
        ridge=.8*np.diag(gram.diagonal().clip(.0002))
        daily=np.linalg.solve(grams+ridge,rhs)
        w=.7*np.linalg.solve(gram+ridge,rhs.mean(0))+.3*np.median(daily,axis=0)
        reduced=w@bands@mapping(daily@bands,gram)  # fit directly on the three disjoint payoffs
        fine=w@mapping(daily,gram)@bands
        correction=((fine-reduced)*recover).cumsum(1)*SCALE
        fitted.append((centers.mean(0),correction))
    return fitted


def changes(raw,t,models):
    out=[]
    for x,(center,w) in zip(features(raw,t),models):
        delta=(x-center)@w
        if x.shape[1]>1:delta-=delta.mean(1,keepdims=True)
        out.append(delta.astype('float32'))
    return out


def load():
    parent=best.load()
    with np.load(OUT/'model.npz') as st:
        fit_params=[(st[f'c{k}'],st[f'w{k}']) for k in range(2)]
        mask=st['mask']
    def apply(raw,p):
        q=parent(raw,p).copy()
        for start in range(0,len(q),400):
            t=np.arange(start,min(start+400,len(q)))
            for z,k in zip(changes(raw,t,fit_params),mask):
                q[t]+=.15*z*k
        return q
    return apply


def gains(rows):
    s=measure(rows)
    return np.subtract(s['candidate'],s['base']).tolist()


def main():
    OUT.mkdir(exist_ok=True)
    stats=[]
    folds,daily=[],[]
    for end in (8,10,12,14):
        stats.extend(record(d) for d in range(len(stats),end))
        pars=fit(stats)
        rows=[[],[]]
        for d in (end,end+1):
            raw=book.rawday(d)
            t=np.arange(d%4,len(raw),4)
            p=np.load(best.online.best.refit.OUT/f'reference_{d}.npy',mmap_mode='r')[t].copy()
            p[:,:,1]=np.load(earlier.OUT/f'y2_day_{d}.npy',mmap_mode='r')[t]
            y=labels(d).reshape(-1,50,3)[t]
            sides=changes(raw,t,pars)
            if d==end:
                early=t[t<177]
                for a,b in zip(changes(raw[:177],early,pars),sides):
                    np.testing.assert_allclose(a,b[:len(early)],rtol=0,atol=3e-6)
            for g,z in zip(rows,sides):
                g.append(tuple(a.reshape(-1,3) for a in (p,p+.15*z,y)))
            daily.append([gains(g[-1:]) for g in rows])
        folds.append([gains(g) for g in rows])
        print(end+1,'local and market IC gains',folds[-1],flush=True)
    keep=(np.min(folds,axis=0)>0)&((np.array(daily)>0).sum(0)>=5)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=daily,retained=keep.tolist()),indent=2))
    if keep.any():
        stats.extend(record(d) for d in range(len(stats),16))
        pars=fit(stats)
        data=dict(mask=keep)
        for i,(c,w) in enumerate(pars):data.update({f'c{i}':c,f'w{i}':w})
        np.savez(OUT/'model.npz',**data)
        baseline=json.loads((best.OUT/'reload_verification.json').read_text())['pooled']['score']
        for name in ('raw_replay','reload_verification'):
            res=best.online.best.roll.gls.best.walk.replay(load())
            if name=='reload_verification':
                old=json.loads((OUT/'raw_replay.json').read_text())
                np.testing.assert_allclose(res['pooled']['candidate'],old['pooled']['candidate'],rtol=0,atol=1e-10)
            (OUT/f'{name}.json').write_text(json.dumps(res,indent=2))
            print(name,res['pooled']['score'],flush=True)
            if res['pooled']['score']<=baseline:break


if __name__=='__main__':main()
