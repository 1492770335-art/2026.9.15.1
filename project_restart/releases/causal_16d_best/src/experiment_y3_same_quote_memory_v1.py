"""Compare current side-specific book fields to their last observed value at the identical price."""
import sys

import numpy as np
import torch

import audit_y3_quote_side_fields as fields
import y3_same_deadline_references as history

quote,trial = fields.quote,fields.trial
study,prev = history.study,history.incumbent
OUT = trial.clock.ART / 'y3_same_quote_memory_v1'
ARMS = ('all book changes', 'same unchanged price', 'previous visit to this price',
        'both types of same-price update', 'most recent displaced observation',
        'other side field assignment', 'newly exposed price', 'most recent plus previous visit',
        'past price within two ticks')
CANDIDATES=(1,2,3,6,7)
CONTROLS=(0,4,5,8)


def previous_visits(level,valid):
    """For each stock-price key, link consecutive observed times in ascending order."""
    times,stocks=np.nonzero(valid)
    key=level[times,stocks]*50+stocks
    order=np.lexsort((times,key))
    same=key[order[1:]]==key[order[:-1]]
    old=np.full(level.shape,-1,dtype=int)
    curr,prior=order[1:][same],order[:-1][same]
    old[times[curr],stocks[curr]]=times[prior]
    return old


def parameters():
    """Only the first four training days identify the fields and calibrate the price grid."""
    profile=fields.training_profile()
    ids=np.r_[19:40,41:83]
    order=ids[np.argsort(profile[ids])]
    return dict(sides=np.stack((order[-4:][::-1],order[:4])), k=quote.grid.calibration(),
                coefficients=np.asarray(quote.calibration()))


def design_with(raw,p,t,params):
    sides=params["sides"]
    mid,half,_,ok=quote.infer(raw, float(params["k"]), params["coefficients"])
    now=np.arange(len(raw))[:,None]
    stock=np.arange(50)[None,:]
    groups=[[] for _ in range(8)]
    for j,level in enumerate((mid-half,mid+half)):
        old=previous_visits(level,ok)
        adjacent=previous_visits(np.floor_divide(level,4),ok)
        def records(features,source):
            age=now-source
            good=(source>=0)&(age<=240)&ok
            return (features-features[source.clip(0),stock])*good[:,:,None],good,age
        x=np.arcsinh(raw[:,:,sides[j]].astype('float32'))
        dx=np.diff(x,axis=0,prepend=x[:1])
        value,good,age=records(x,old)
        # Consecutive observations belong to a single visit. A revisit starts after a gap.
        revisit=good&(age>1)
        stable=good&(age==1)
        wrong,_,_=records(np.arcsinh(raw[:,:,sides[1-j]].astype('float32')),old)
        offset=(old-9).clip(0)
        prior=(x-x[offset,stock])*revisit[:,:,None]
        nearby,on,span=records(x,adjacent)
        states=(dx,value*stable[:,:,None],value*revisit[:,:,None],prior,
                wrong*revisit[:,:,None],dx*((~good)&ok)[:,:,None],dx*revisit[:,:,None],
                nearby*(on&(span>1))[:,:,None])
        for g,v in zip(groups,states):
            cs=np.concatenate((np.zeros_like(v[:1],dtype=float),v.cumsum(0,dtype=float)))
            end=t+1
            g.extend([(cs[end]-cs[(end-w).clip(0)])/np.sqrt(w) for w in (8,40)])
    z=np.concatenate([np.concatenate(g,axis=2) for g in groups],axis=2)
    return trial.clock.center(np.concatenate((p[t],z),axis=2)).astype('float32')


def design(raw,p,t):
    return design_with(raw,p,t,parameters())


def fit(rows):
    total,stable,revisit,late,swap,new,temporal,near=[np.arange(3+16*i,19+16*i) for i in range(8)]
    bank=[(total,3),(stable,19),(revisit,19),(np.r_[stable,revisit],19),(late,19),(swap,19),
          (new,19),(np.r_[temporal,revisit],19),(near,19)]
    return dict(**prev.gram(rows,[(g,n,.35) for g,n in bank]), **parameters())


changes=prev.changes


def correction():
    with np.load(OUT/"model.npz") as f:
        state = dict(f)
    def update(raw, p):
        x = design_with(raw, p, np.arange(len(raw)), state)
        return changes(x, state)[int(state["pick"])]
    return update


def layer():
    return study.layer(sys.modules[__name__],prev)


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
