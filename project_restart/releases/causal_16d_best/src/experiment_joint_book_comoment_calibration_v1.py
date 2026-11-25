"""Reprice the new mean-product and residual covariance forecasts at each label horizon."""
import sys
import numpy as np
import torch

import experiment_joint_factor_order_calibration_v1 as calibration
import y3_book_covariance_residual_references as history

parent,trial,stack=history.incumbent,history.trial,history.stack
OUT=trial.clock.ART/'joint_book_comoment_calibration_v1'
ARMS=('unchanged','half Y3 adjustment','one Y3 adjustment','two Y3 adjustments','date-median slope',
      'separate covariance components','joint slope and covariance components','date-confirmed components')
basis,fitted=calibration.basis,calibration.fitted


def signals(raw,st):
    out=[]
    for ticks in np.array_split(np.arange(len(raw)),max(1,(len(raw)+799)//800)):
        out.append(parent.advice(raw,ticks,st)[:,:,[1,2,3]])
    return np.concatenate(out)


def record(d):
    raw=trial.clock.book.rawday(d)
    old=history.previous.references(d,raw)
    current=history.references(d,raw)
    y=trial.clock.labels(d).reshape(-1,50,3)
    x=signals(raw,parent.params(d-d%2))
    rows=[]
    for a,p in zip(old,current):
        v=np.concatenate((np.ones_like(p[:,:,:1]),p,basis(a,p,x),y),axis=2).reshape(-1,11).astype(float)
        rows.append(v.T@v)
    return rows


def adjustment():
    with np.load(OUT/'model.npz') as f: w=f['w']
    if not np.any(w[1:]):
        def apply_increment(raw,old,p):
            return p+((p[:,:,2]-old[:,:,2])[:,:,None]*w[0]).astype('float32')
        return apply_increment
    with np.load(parent.OUT/'model.npz') as f: st=dict(f)
    return lambda raw,old,p:p+(basis(old,p,signals(raw,st))@w).astype('float32')


def layer():
    before=history.previous.incumbent.layer()
    change=parent.correction()
    adapt=adjustment()
    def apply(raw,base):
        old=before(raw,base)
        p=old.copy()
        p[:,:,2]+=change(raw,old)
        return adapt(raw,old,p)
    return apply


def replay_with():
    adapt=adjustment()
    def read(d,raw):
        p,h,n=history.read(d,raw)
        old,part,_=history.previous.read(d,raw)
        return adapt(raw,old,p),adapt(raw[:n],part,h),n
    return stack.verifier,read


if __name__=='__main__':
    torch.set_num_threads(2)
    mod=sys.modules[__name__]
    if sys.argv[1:]==['replay'] or calibration.train(mod):
        calibration.replay(mod)
