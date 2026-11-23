"""Update stock-specific dynamics from already observed earlier-to-later forecast transitions."""
import json
import numpy as np
import experiment_rebased_expert_mixer_v1 as best
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'online_forecast_transitions_v1'
LAGS=(5,20,40)
BLOCK=16


def state(p):
    rel=p.astype(float)-p.mean(1,keepdims=True)
    n=len(p);t=np.arange(n)
    old=rel[(t-5).clip(0)]
    x=np.concatenate((rel,rel-old,np.tanh(3*rel)*.30),axis=2).clip(-3.,3.)
    return x,rel[:,:,:2].clip(-3,3)


def blocksum(a):
    n=len(a);nb=(n+BLOCK-1)//BLOCK
    pad=np.zeros((nb*BLOCK,*a.shape[1:]),dtype=float);pad[:n]=a
    return pad.reshape(nb,BLOCK,*a.shape[1:]).sum(1)


def observed(x,out,h):
    t=np.arange(len(x))
    prev=x[(t-h).clip(0)].copy();prev[:h]=0
    n=len(x);nb=(n+BLOCK-1)//BLOCK
    # At index u the last available supervised pair is forecast(u-h), forecast(u).
    pad=np.zeros((nb*BLOCK,50,x.shape[-1]));pad[:n]=prev
    a=pad.reshape(nb,BLOCK,50,-1)
    gram=np.einsum('btsi,btsj->bsij',a,a,optimize=True)
    rhs=blocksum(prev[:,:,:,None]*out[:,:,None,:])
    return gram,rhs


def history_record(p):
    x,b=state(p)
    rows=[]
    for h in LAGS:
        g,r=observed(x,b,h)
        rows.append((g.sum(0)/(len(p)-h),r.sum(0)/(len(p)-h)))
    return rows


def prior_fit(history):
    priors=[]
    for days in zip(*history):
        g,r=[np.mean(a,0) for a in zip(*days)]
        g=.7*g.mean(0)+.3*g;r=.7*r.mean(0)+.3*r
        penalty=.34*np.eye(g.shape[-1])[None]*(g.diagonal(axis1=1,axis2=2).clip(.0002))[:,None,:]
        priors.append((np.linalg.solve(g+penalty,r),g))
    return priors


def past(a,width):
    cs=np.concatenate((np.zeros_like(a[:1]),a.cumsum(0)))
    t=np.arange(len(a));return cs[t]-cs[(t-width).clip(0)]


def features(p,priors):
    x,b=state(p);idx=np.arange(len(p))//BLOCK
    local=[];common=[]
    for h,(prior,previous) in zip(LAGS,priors):
        g,r=observed(x,b,h)
        err=r-g@prior
        gr=past(g,48);rr=past(err,48)
        reg=280*(previous+np.eye(x.shape[2])*.006)
        # A stock's own unsupervised updates vs the market's pooled unsupervised updates.
        shared=np.linalg.solve(gr.mean(1)+reg.mean(0),rr.mean(1))
        common_coef=np.broadcast_to(shared[:,None],(len(shared),50,*shared.shape[1:]))
        local_coef=common_coef+np.linalg.solve(gr+reg,rr-gr@common_coef)
        for dst,coef in ((common,common_coef),(local,local_coef)):
            dx=np.einsum('tsf,tsfk->tsk',x,coef[idx],optimize=True)
            dst.append((dx-dx.mean(1,keepdims=True)).astype('float32'))
    return np.concatenate(common,2),np.concatenate(local,2)


def stat(p,y,x):
    a=p.astype(float)-p.mean((0,1));b=y.astype(float)-y.mean((0,1))
    slope=((a*b).mean((0,1))/(a*a).mean((0,1)).clip(.0001)).clip(.35,3)
    e=b/slope-a;e-=e.mean(1,keepdims=True)
    u=x.reshape(-1,x.shape[2]).astype(float)
    return u.T@u/len(u),u.T@e.reshape(-1,3)/len(u)


def fit(history):
    gs,bs=map(np.stack,zip(*history));g=gs.mean(0)
    reg=1.35*np.diag(g.diagonal().clip(.00002))
    daily=np.linalg.solve(gs+reg,bs)
    w=.5*(.65*np.linalg.solve(g+reg,bs.mean(0))+.35*np.median(daily,0))
    return (w*abs(np.sign(daily).mean(0))).clip(-.35,.35).astype('float32')


def load():
    parent=best.load()
    with np.load(OUT/'model.npz') as st:
        priors=list(zip(st['means'],st['grams']));weights=st['weights'];arm=st['arm']
    def predict(raw,p):
        q=parent(raw,p);x=features(q,priors)
        return (q+sum(a@(w*(arm==i)) for i,(a,w) in enumerate(zip(x,weights)))).astype('float32')
    return predict


def main():
    OUT.mkdir(exist_ok=True)
    histories=[];readouts=[[],[]];days=[];folds=[]
    for end in (8,10,12,14):
        prior=prior_fit(histories) if histories else None
        ws=[fit(r) for r in readouts] if readouts[0] else []
        groups=[[[],[]] for _ in range(2)]
        for d in (end,end+1):
            p=np.load(best.refit.OUT/f'reference_{d}.npy');y=labels(d).reshape(p.shape)
            histories.append(history_record(p))
            if prior is None:continue
            for j,ref in enumerate((p,np.load(best.refit.core.OUT/f'pred_{d}.npy'))):
                xx=features(ref,prior);t=np.arange(d%4,len(p),4)
                if not j and d==end:
                    prefix=features(ref[:171],prior)
                    for z,small in zip(xx,prefix):np.testing.assert_allclose(z[:171],small,atol=2e-6,rtol=0)
                day=[]
                for i,x in enumerate(xx):
                    if j==0:
                        tr=t[t<len(ref)-60];readouts[i].append(stat(ref[tr],y[tr],x[tr]))
                    if not ws:continue
                    q=(ref[t]+x[t]@ws[i]).astype('float32')
                    row=tuple(v.reshape(-1,3) for v in (ref[t],q,y[t]));groups[j][i].append(row)
                    s=measure([row]);day.append(np.subtract(s['candidate'],s['base']))
                if ws and j==0:days.append(day)
        if ws:
            folds.append([[measure(a) for a in pair] for pair in groups])
            print(end+1,[[np.subtract(s['candidate'],s['base']).tolist() for s in pair] for pair in folds[-1]],flush=True)
    g=np.array([[[np.subtract(s['candidate'],s['base']) for s in pair] for pair in f] for f in folds])
    good=(g[:,0].min(0)>1e-6)&(g[:,1].mean(0)>0)&((np.array(days)>1e-6).sum(0)>=3)
    personal=good[1]&(~good[0]|((g[:,0,1]-g[:,0,0]).mean(0)>0))
    arm=np.where(personal,1,np.where(good[0],0,-1))
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,arm=arm.tolist(),daily=np.array(days).tolist(),fold_starts=[13,15]),indent=2))
    print('arm per target',arm,flush=True)
    if (arm>=0).any():
        prior=prior_fit(histories)
        np.savez(OUT/'model.npz',means=np.stack([v[0] for v in prior]),grams=np.stack([v[1] for v in prior]),
                 weights=np.stack([fit(r) for r in readouts]),arm=arm)
        replay=best.roll.gls.best.walk.replay
        res=replay(load())
        (OUT/'raw_replay.json').write_text(json.dumps(res,indent=2))
        prev=json.loads((best.OUT/'reload_verification.json').read_text())['pooled']
        print('official',res['pooled']['score'],res['pooled']['score']-prev['score'],flush=True)
        if res['pooled']['score']>prev['score']:
            check=replay(load())
            np.testing.assert_allclose(check['pooled']['candidate'],res['pooled']['candidate'],rtol=0,atol=1e-10)
            (OUT/'reload_verification.json').write_text(json.dumps(check,indent=2))


if __name__=='__main__':main()
