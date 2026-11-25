"""Estimate a reliable amplitude for the separated-time-scale response from rolling training dates."""
import json
import sys

import numpy as np
import torch

import experiment_y3_order_response_filter_v1 as source
import y3_pressure_validation as trial

OUT=trial.clock.ART/'y3_order_filter_strength_v1'
SCALE=np.array([1.,2.,4.,8.],dtype='float32')


def select_amplitude(folds,daily):
    f=np.array(folds);d=np.array(daily)
    # Compare with the retained factor of one on both reference models and metrics.
    diff=f[:,:,:,1:]-f[:,:,:,:1]
    day=d[:,:,1:]-d[:,:,:1]
    eligible=((diff>0).sum(0).min((0,1))>=3)&(diff.mean(0).min((0,1))>0)&(diff[-1].min((0,1))>0)&((day>0).sum(0).min(0)>=5)
    ids=np.flatnonzero(eligible)
    if not len(ids):return None
    scores=diff[:,0,0].mean(0)
    best=ids[np.argmax(scores[ids])]
    # Prefer the smaller amplitude when the paired fold differences are smaller than one
    # standard error of their mean.
    near=[]
    for i in ids:
        margin=diff[:,0,0,best]-diff[:,0,0,i]
        if margin.mean()<=margin.std(ddof=1)/np.sqrt(len(margin))+1e-12:near.append(i)
    return int(min(near)+1)


def analytic_scale(p,y,s):
    v=np.column_stack([a.ravel() for a in (p,y,s)]).astype(float)
    g=np.cov(v,rowvar=False,bias=True)
    denom=g[0,1]*g[2,2]-g[1,2]*g[0,2]
    if denom<=1e-20:return None
    return float((g[1,2]*g[0,0]-g[0,1]*g[0,2])/denom)


def run():
    hist=[];prior=[];own=[];folds=[];daily=[];optima=[]
    for d in range(16):
        raw=trial.clock.book.rawday(d);y=trial.clock.labels(d).reshape(-1,50,3)
        pre=trial.incumbent.signature.observations(raw);z=source.state(raw,pre)
        if d>=8:
            if d%2==0:
                geo=trial.parent.geometry(hist);core=trial.incumbent.fit(prior)
                w,_=source.fit(own);groups=[[],[]]
            e=source.changes(z,w)[1]
            if d%2==0:
                short=source.state(raw[:357],trial.incumbent.signature.observations(raw[:357]))
                np.testing.assert_allclose(e[:357],source.changes(short,w)[1],rtol=0,atol=5e-6)
            terms=trial.parent.terms(trial.parent.incumbent.state(raw)[1],*geo)
            old=trial.incumbent.signature.predict(pre,core)
            for j,ref in enumerate(trial.curve.REFS):
                p=ref(d).reshape(y.shape).copy();p[:,:,2]+=terms[0]+trial.parent.candidates(terms)[2]
                p[:,:,2]+=trial.incumbent.candidates(old,p)[4]
                moments=[s.sum(0) for s in trial.scoring.statistics(p,y,[a*e for a in SCALE])]
                groups[j].append(moments)
                if not j:
                    daily.append([trial.scoring.audit.pooled(s).tolist() for s in moments])
                    optima.append([analytic_scale(p[:,:,2],y[:,:,2],e),analytic_scale(trial.scoring.late(p),trial.scoring.late(y),e)])
            if d%2:
                folds.append([[trial.scoring.audit.pooled(m).tolist() for m in np.sum(g,axis=0)] for g in groups])
                print('ending',d+1,folds[-1][0],flush=True)
        t=np.arange(d%5,len(raw)-60,5)
        hist.append(trial.parent.incumbent.stats(raw,y,d));prior.append(trial.incumbent.record(pre,t,y));own.append(source.observe(z,y,d))
    pick=select_amplitude(folds,daily)
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,amplitudes=SCALE.tolist(),folds=folds,daily=daily,
            per_day_optima=optima),indent=2))
    if pick is not None:
        model,_=source.fit(own);np.savez(OUT/'model.npz',w=model['w'],amplitude=SCALE[pick])
    return pick is not None


def correction():
    with np.load(OUT/'model.npz') as s:model=dict(w=s['w']);amp=float(s['amplitude'])
    def predict(raw,p):
        z=source.state(raw,trial.incumbent.signature.observations(raw))
        return amp*source.changes(z,model)[1]
    return predict


def layer():return trial.layer(sys.modules[__name__])
def load():return trial.load(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    if run():trial.scoring.outer(sys.modules[__name__],reference=trial.incumbent)
