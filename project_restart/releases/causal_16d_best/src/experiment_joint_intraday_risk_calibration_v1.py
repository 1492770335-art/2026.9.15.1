"""Price the replicated intraday factor response for each return horizon independently."""
import sys
import numpy as np
import torch

import experiment_joint_factor_order_calibration_v1 as calibration
import y3_live_risk_order_references as history

parent,trial,stack=history.incumbent,history.trial,history.stack
OUT=trial.clock.ART/'joint_intraday_risk_calibration_v1'
ARMS=calibration.prior.ARMS
fitted,gain,basis=calibration.fitted,calibration.gain,calibration.basis


def signals(raw,st):
    # The fourth branch is agreement between odd and even previously settled intraday intervals.
    return parent.features(raw,np.arange(len(raw)),st)[:,:,9:12]


def record(d):
    raw=trial.clock.book.rawday(d)
    old=history.previous.references(d,raw)
    forecasts=history.references(d,raw)
    x=signals(raw,parent.state(d-d%2))
    y=trial.clock.labels(d).reshape(-1,50,3)
    data=[]
    for p,q in zip(old,forecasts):
        a=np.concatenate((np.ones_like(p[:,:,:1]),q,basis(p,q,x),y),2).reshape(-1,11).astype(float)
        data.append(a.T@a)
    return data


def adjustment():
    with np.load(OUT/'model.npz') as f:
        w=f['w']
    with np.load(parent.OUT/'model.npz') as f:
        st=dict(f)
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
        old,short,_=history.previous.read(d,raw)
        p,h,n=history.read(d,raw)
        return adapt(raw,old,p),adapt(raw[:n],short,h),n
    return stack.verifier,read


if __name__=='__main__':
    torch.set_num_threads(2)
    mod=sys.modules[__name__]
    if sys.argv[1:]==['replay'] or calibration.train(mod):
        calibration.replay(mod)
