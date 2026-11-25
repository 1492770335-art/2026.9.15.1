"""Separate shrinkage from new signals and score near-window corrections to a frozen late specialist."""
import json
import sys

import numpy as np
import torch

import experiment_y3_activity_order_signatures_v1 as signature
import y3_tail_specialist_eval as evaluator

parent,clock,curve=evaluator.parent,evaluator.clock,evaluator.curve
OUT=clock.ART/'y3_pressure_horizon_controls_v1'


def record(prefix,t,y):
    ratio=float(curve.settled.historical_geometry()[1][2,1]/curve.settled.historical_geometry()[1][1,1])
    sums=None
    for chunk in np.array_split(t,max(1,(len(t)+127)//128)):
        z,n=signature.signature(prefix,chunk)
        x=z.reshape(-1,z.shape[2])
        late=evaluator.late(y[chunk]).clip(-3.,3.).ravel()
        near=(ratio*clock.center(y[chunk,:,1])).clip(-3.,3.).ravel()
        stats=(x.T@x,x.T@late,x.T@near)
        if sums is None:sums=[a.astype(float) for a in stats]
        else:
            for a,b in zip(sums,stats):a+=b
    gram,far,near=[s/(len(t)*50) for s in sums]
    return (gram,far,n),(gram,near,n)


def fit(hist):
    return np.column_stack([signature.fit(part) for part in zip(*hist)])


def candidates(z,p):
    far,_,order,near,_,near_order=np.moveaxis(z,2,0)
    old=evaluator.late(p)
    sig=far+order
    base=.10*(sig-old)
    scale=curve.settled.historical_geometry()[1]
    baseline_near=clock.center(p[:,:,1])*(scale[2,1]/scale[1,1])
    near_new=near+near_order
    confirm=np.tanh(5*(near_new-baseline_near))*np.tanh(8*(sig-old))
    return [-.10*old,.10*sig,.10*(far-old),base,
            base+.10*near_order,base+.07*(near_new-baseline_near),
            base*(1+.55*confirm),base+.07*order]


def run():
    hist=[];records=[];folds=[];daily=[]
    for d in range(16):
        raw=clock.book.rawday(d);prefix=signature.observations(raw);y=clock.labels(d).reshape(-1,50,3)
        if d>=8:
            if d%2==0:
                w=fit(records);geo=parent.geometry(hist);groups=[[],[]]
            z=signature.predict(prefix,w)
            if d%2==0:
                np.testing.assert_allclose(z[:331],signature.predict(signature.observations(raw[:331]),w),atol=4e-6,rtol=0)
            terms=parent.terms(parent.incumbent.state(raw)[1],*geo)
            for j,ref in enumerate(curve.REFS):
                p=ref(d).reshape(y.shape).copy();p[:,:,2]+=terms[0]+parent.candidates(terms)[2]
                moments=[s.sum(0) for s in evaluator.statistics(p,y,candidates(z,p))]
                groups[j].append(moments)
                if not j:daily.append([evaluator.audit.pooled(a).tolist() for a in moments])
            if d%2:
                folds.append([[evaluator.audit.pooled(a).tolist() for a in np.sum(g,axis=0)] for g in groups])
                print('ending',d+1,folds[-1][0][0],flush=True)
        records.append(record(prefix,np.arange(d%5,len(raw)-60,5),y))
        hist.append(parent.incumbent.stats(raw,y,d))
    f=np.array(folds);d=np.array(daily)
    pick=evaluator.select(f[:,:,:,4:]-f[:,:,:,3:4],d[:,:,4:]-d[:,:,3:4])
    if pick is not None:pick+=4
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,folds=folds,daily=daily,arms=[
        'shrink only','add signal only','unordered replacement','original replacement',
        'near ordering','near model','near agreement','extra far ordering']),indent=2))
    if pick is not None:np.savez(OUT/'model.npz',w=fit(records),pick=pick)
    return pick is not None


def correction():
    with np.load(OUT/'model.npz') as s:w,pick=s['w'],int(s['pick'])
    return lambda raw,p:candidates(signature.predict(signature.observations(raw),w),p)[pick]


def layer():
    prior,update=parent.layer(),correction()
    def apply(raw,p):
        q=prior(raw,p);q[:,:,2]+=update(raw,q)
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as ancestor
    prior,update=ancestor.load(),layer()
    return lambda raw,p:update(raw,prior(raw,p))


if __name__=='__main__':
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    if run():evaluator.outer(sys.modules[__name__])
