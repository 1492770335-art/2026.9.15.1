"""Extract the common and conflicting parts of near- and long-horizon stock order with monotone cross-sectional fits."""
import sys
import numpy as np
import torch
from scipy.optimize import isotonic_regression
import y3_terminal_panel_trial as screen

trial=screen.trial
OUT=trial.clock.ART/'y3_monotone_horizon_projection_v1'
ARMS=('linear exposure','full monotone projection','out-of-basket projection','nonlinear ordering residual',
      'decreasing projection','misassigned anchor','opposing monotone parts','very short anchor','half projection residual')
CANDIDATES=(1,2,3,4,6,7,8)
CONTROLS=(0,5)


def monotone(key,value,cross):
    result=np.empty_like(value)
    for group in ((0,1) if cross else (0,)):
        ref=np.arange(1-group,50,2) if cross else np.arange(50)
        dest=np.arange(group,50,2) if cross else ref
        k=key[:,ref];order=np.argsort(k,axis=1,kind='stable')
        k=np.take_along_axis(k,order,1)
        v=np.take_along_axis(value[:,ref],order,1)
        for t in range(len(k)):
            anchors=k[t];out=v[t];weights=None
            if np.any(anchors[1:]==anchors[:-1]):
                anchors,index,count=np.unique(anchors,return_index=True,return_counts=True)
                out=np.add.reduceat(out,index)/count
                weights=count
            response=isotonic_regression(out,weights=weights).x
            result[t,dest]=np.interp(key[t,dest],anchors,response)
    return trial.clock.center(result)


def features(raw,p,cross_only=False):
    coeff=trial.curve.settled.historical_geometry()[1]
    short,near,balanced=np.moveaxis(trial.clock.center(p),2,0)
    tail=balanced-coeff[2,1]/coeff[1,1]*near
    slope=(near*tail).mean(1,keepdims=True)/(.0005+(near*near).mean(1,keepdims=True))
    linear=slope*near
    cross=monotone(near,tail,True)
    if cross_only:
        output=np.zeros((*tail.shape,7),dtype='float32')
        output[:,:,0]=linear
        output[:,:,2]=cross
        return output
    fit=monotone(near,tail,False)
    inverse=monotone(-near,tail,True)
    wrong=monotone(np.roll(near,17,axis=1),tail,True)
    short_fit=monotone(short,tail,True)
    return np.stack((linear,fit,cross,cross-linear,inverse,wrong,short_fit),2).astype('float32')


def fit(rows):
    return screen.projected_readouts(rows,[([3],3,.4),([4],4,.4),([5],4,.4),([6],4,.4),([7],4,.4),
            ([8],4,.4),([5,7],4,.4),([9],4,.4),([6],4,.2)])


def correction():
    with np.load(OUT/'model.npz') as f:
        w,pick=f['w'],int(f['pick'])
    cross_only=not np.any(w[[4,6,7,8,9],pick])
    return lambda raw,p:screen.changes(features(raw,p,cross_only),p,w)[pick]


def layer():
    predict,update=screen.history.incumbent.layer(),correction()
    def apply(raw,p):
        q=predict(raw,p)
        q[:,:,2]+=update(raw,q)
        return q
    return apply


if __name__=='__main__':
    torch.set_num_threads(2)
    screen.run(sys.modules[__name__])
