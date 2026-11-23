"""Estimate conditional errors of chronological out-of-fold forecasts by recent book transition."""
import json
import numpy as np
import experiment_book_state_run_context_v1 as runs
from experiment_causal_cross_horizon_ridge_v1 import ART,measure

walk=runs.walk
OUT=ART/'state_conditional_errors_v1'
NCOARSE=36
NFINE=NCOARSE*9


def inputs(raw,p):
    states=np.stack([runs.contexts(raw,k,True)//9 for k in runs.PAIRS])
    local=p-p.mean(1,keepdims=True)
    sign=np.tanh(.75*(raw[:,:,41].astype(float)-raw[:,:,42].astype(float)))
    moving=(raw[:,:,0].astype(float)-.00136).cumsum(0)
    t=np.arange(len(p))
    lag=np.tanh((moving-moving[(t-8).clip(0)])/3.)
    x=np.concatenate((np.ones_like(p[:,:,:1]),np.tanh(2.5*local),sign[:,:,None],lag[:,:,None]),axis=2)
    return states,x.astype('float32')


def statistics(p,y,states,x):
    p,y=p.astype(float),y.astype(float)
    pc=p-p.mean((0,1));yc=y-y.mean((0,1))
    slope=((pc*yc).mean((0,1))/(pc*pc).mean((0,1))).clip(.4,2.5)
    err=y/slope-p
    err-=err.mean(1,keepdims=True)
    x=x.reshape(-1,x.shape[2]).astype(float)
    err=err.reshape(-1,3)
    gs=[];hs=[]
    for st in states:
        idx=st.ravel()
        def collect(v):
            return np.bincount(idx,weights=v,minlength=NFINE)
        gram=np.stack([[collect(a*b) for b in x.T] for a in x.T],axis=1).transpose(2,0,1)
        rhs=np.stack([[collect(a*b) for b in err.T] for a in x.T]).transpose(2,0,1)
        gs.append(gram);hs.append(rhs)
    return np.stack(gs),np.stack(hs)


def fit(history):
    g,h=(np.stack(v) for v in zip(*history))
    n,d=g.shape[:2]
    def solve(gram,rhs,prior,strength):
        penalty=strength*np.array([1.,.30,.30,.30,.7,.7])
        return np.linalg.solve(gram+np.diag(penalty),rhs+penalty[:,None]*prior)
    def estimate(gram,rhs):
        gg=gram.reshape(d,NCOARSE,9,6,6).sum(2)
        hh=rhs.reshape(d,NCOARSE,9,6,3).sum(2)
        coarse=solve(gg,hh,0.,450.)
        fine=solve(gram,rhs,coarse.repeat(9,axis=1),180.)
        return np.stack((coarse.repeat(9,axis=1),fine-coarse.repeat(9,axis=1)))
    per_day=np.stack([estimate(a,b) for a,b in zip(g,h)])
    pooled=estimate(g.mean(0),h.mean(0))
    weights=.5*(pooled+np.median(per_day,axis=0))*abs(np.sign(per_day).mean(0))
    return (.45*weights.clip(-.7,.7)).astype('float32')


def predict(p,states,x,w,use_context,mask=1.):
    w=w[0]+(w[1] if use_context else 0)
    delta=np.zeros_like(p)
    for i,st in enumerate(states):
        for start in range(0,len(p),240):
            s=slice(start,start+240)
            delta[s]+=np.einsum('tsf,tsfh->tsh',x[s],w[i,st[s]],optimize=False)
    delta/=len(states)
    return (p+mask*(delta-delta.mean(1,keepdims=True))).astype('float32')


def main():
    OUT.mkdir(exist_ok=True)
    geometry=walk.cohort.geometry()
    history,prior,earlier,folds,daily=[],[],[],[],[]
    for end in (8,10,12,14):
        c=walk.blend.fixed_weight(prior)
        w=np.stack(walk.fit(earlier)) if earlier else None
        if w is not None:w[:,:,:2]=0
        weights=fit(history) if history else None
        groups=[[],[]]
        for d in (end,end+1):
            raw=walk.book.rawday(d)
            p,y,fs,s=walk.blend.training_day(d,prior)
            prior.append(s)
            p=walk.blend.parent.fixed_forecast(p,fs,c)
            mr=walk.cohort.features(raw,p,geometry,False)
            earlier.append(walk.moments(p[::4],y[::4],tuple(v[::4] for v in mr)))
            if w is not None:p=walk.apply(p,mr,w)
            st,x=inputs(raw,p)
            history.append(statistics(p[::4],y[::4],st[:,::4],x[::4]))
            if weights is None:continue
            for j in range(2):
                q=predict(p,st,x,weights,j)
                if d==end:
                    hs,hx=inputs(raw[:201],p[:201])
                    np.testing.assert_allclose(q[:201],predict(p[:201],hs,hx,weights,j),atol=2e-6,rtol=0)
                row=tuple(v.reshape(-1,3) for v in (p,q,y))
                groups[j].append(row)
                if j:
                    sc=measure([row])
                    daily.append(np.subtract(sc['candidate'],sc['base']))
        if groups[0]:
            f={k:measure(v) for k,v in zip(('current','transition'),groups)}
            folds.append(f)
            print(end+1,[v['gain'] for v in f.values()],flush=True)
    gain=np.stack([np.subtract(f['transition']['candidate'],f['transition']['base']) for f in folds])
    adv=np.stack([np.subtract(f['transition']['candidate'],f['current']['candidate']) for f in folds])
    gate=(gain.min(0)>0)&(adv.mean(0)>0)&((np.stack(daily)>0).sum(0)>=4)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=np.stack(daily).tolist(),retained=gate.tolist()),indent=2))
    if gate.any():
        weights=fit(history)
        np.savez(OUT/'model.npz',weights=weights,mask=gate)
        s=walk.replay(load())
        incumbent=json.loads((walk.specialist.OUT/'raw_replay.json').read_text())['pooled']['score']
        s['gain_over_specialist']=s['pooled']['score']-incumbent
        (OUT/'raw_replay.json').write_text(json.dumps(s,indent=2))
        print(s['pooled'],s['gain_over_specialist'],flush=True)

def load():
    with np.load(OUT/'model.npz') as state:
        weights=state['weights'].copy()
        mask=state['mask'].copy()
    parent=walk.specialist.load()
    def advance(raw,p):
        q=parent(raw,p)
        return predict(q,*inputs(raw,q),weights,True,mask)
    return advance


def verify():
    expected=json.loads((OUT/'raw_replay.json').read_text())
    actual=walk.replay(load())
    for field in ('candidate','score','rows'):
        np.testing.assert_allclose(actual['pooled'][field],expected['pooled'][field],rtol=0,atol=1e-10)
    actual['prior_best']=json.loads((walk.specialist.OUT/'raw_replay.json').read_text())['pooled']['score']
    actual['gain_over_prior_best']=actual['pooled']['score']-actual['prior_best']
    (OUT/'reload_verification.json').write_text(json.dumps(actual,indent=2))
    print(actual['pooled'],actual['gain_over_prior_best'],flush=True)


if __name__=='__main__':
    import sys
    if '--verify' in sys.argv:verify()
    else:main()
