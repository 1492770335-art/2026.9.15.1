"""Read only the agreement and disagreement between the pretrained and magnitude-tuned path forecasts."""
import sys

import numpy as np
import torch

import y3_scale_conditioned_references as history

parent=history.incumbent
trial=parent.trial
OUT=trial.clock.ART/'y3_path_magnitude_consensus_v1'
ARMS=['frozen signal','tuned signal','two-view mean','shared direction','disputed component',
      'shuffled agreement','consensus beyond linear','soft agreement','half disputed']


def design(z):
    a,b=z[:,:,:2],z[:,:,4:6]
    mean=.5*(a+b)
    shared=np.where(a*b>0,np.sign(mean)*np.minimum(abs(a),abs(b)),0)
    # Remove direction-dependent confidence attributable to another stock by a fixed permutation control.
    wrong=np.where(a*np.roll(b,17,axis=1)>0,np.sign(a)*np.minimum(abs(a),abs(np.roll(b,17,axis=1))),0)
    reliability=2*abs(a*b)/(a*a+b*b+.0001)
    soft=mean*reliability
    return trial.clock.center(np.concatenate((a,b,mean,shared,mean-shared,wrong,soft),2))


def historical_state(raw,d):
    return design(parent.historical_state(raw,d))


record_with_state=parent.record_with_state
changes_with_reference=parent.changes_with_reference


def warm_history():
    rows=[]
    for d in range(4,8):
        raw=trial.clock.book.rawday(d)
        y=trial.clock.labels(d).reshape(-1,50,3)
        rows.append(record_with_state(historical_state(raw,d),parent.path.early.load(d,raw),y,d))
    return rows


def fit(records):
    gs,bs=[np.stack(v).astype(float) for v in zip(*records)]
    g=gs.mean(0)
    def read(cols,controls):
        basis=np.eye(len(g))[:,cols]
        basis[:controls]=-trial.curve.solve(g[:controls,:controls],g[:controls,cols],.05)
        daily=bs@basis
        mu=daily.mean(0)
        mu*=abs(mu)/(abs(mu)+daily.std(0)/np.sqrt(len(records))+1e-7)
        return basis@trial.curve.solve(basis.T@g@basis,mu,1.)
    groups=[list(range(3+2*j,5+2*j)) for j in range(7)]
    a,b,mean,shared,disputed,wrong,soft=groups
    unique=read(disputed,7)
    bank=[read(a,3),read(b,3),read(mean,3),read(shared,3),unique,read(wrong,3),read(shared,7),
          read(soft,3),.5*unique]
    m=dict(w=(.38*np.column_stack(bank)).astype('float32'))
    if len(records)==12:
        with np.load(parent.OUT/'model.npz') as state:
            m.update({k:state[k] for k in state.files if '/' in k})
    return m


def select(f,d):
    good=[]
    for j in (2,3,4,6,7,8):
        v=f[:,:,0,j]
        if ((v>0).sum(0).min()>=3 and v.mean(0).min()>0 and v[-1].min()>0 and
                (d[:,0,j]>0).sum()>=5 and f[:,:,1,j].mean(0).min()>0 and
                (v-f[:,:,0,5]).mean(0).min()>0):
            good.append(j)
    return max(good,key=lambda j:f[:,0,0,j].mean()) if good else None


def correction():
    with np.load(OUT/'model.npz') as f:
        m=dict(f)
    return lambda raw,p:changes_with_reference(design(parent.encode(raw,m)),m,p)[int(m['pick'])]


def layer():
    predict,update=parent.layer(),correction()
    def apply(raw,p):
        result=predict(raw,p)
        result[:,:,2]+=update(raw,result)
        return result
    return apply


if __name__=='__main__':
    torch.set_num_threads(2)
    history.run(sys.modules[__name__])
