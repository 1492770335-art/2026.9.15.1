"""Recover level and co-movement information hidden when paired anonymous columns are differenced."""
import sys

import numpy as np
import torch

import experiment_y3_order_response_filter_v1 as source
import y3_filter_validation as trial

OUT=trial.clock.ART/'y3_paired_gross_flow_v1'
ARMS=['paired level','paired co-movement','both','co-movement minus delayed activity','level minus delayed activity']
LEFT=[41,50,54,77]
RIGHT=[42,51,55,78]


def components(raw):
    p=trial.prior.incumbent.signature.impact.arrival.signals(raw)
    a=np.arcsinh(raw[:,:,LEFT].astype('float32'));b=np.arcsinh(raw[:,:,RIGHT].astype('float32'))
    da=np.diff(a,axis=0,prepend=a[:1]);db=np.diff(b,axis=0,prepend=b[:1])
    gross=np.tanh((a+b)/2)
    # This amount vanishes when either side is unchanged or the sides move in opposite directions.
    simultaneous=np.tanh((abs(da)+abs(db)-abs(da-db))*.5)
    return p,gross,simultaneous


def compact_filters(p,a):
    # Store the symmetric controls and separated-time contrasts only.
    z=np.concatenate((p,a),2);z-=source.smooth(z,120)
    p,a=z[:,:,:8],z[:,:,8:];pieces=[]
    for span in (4,16,64):pieces.append(trial.clock.center(source.smooth(z,span)).astype('float16'))
    same=(p[:,:,:,None]*a[:,:,None,:]).reshape(len(z),50,32)
    pieces.append(trial.clock.center(source.smooth(same,24)).astype('float16'))
    filtered=[];contrasts=[]
    for gap,age in ((4,16),(16,64),(64,4)):
        h=source.smooth(z,gap);h=np.concatenate((h[:1]*0,h[:-1]))
        shock=(h[:,:,:8,None]*a[:,:,None,:]-p[:,:,:,None]*h[:,:,None,8:]).reshape(len(z),50,32)
        same=source.smooth(shock,gap)
        filtered.append(trial.clock.center(same).astype('float16'))
        contrasts.append(trial.clock.center(source.smooth(shock,age)-same).astype('float16'))
    return np.concatenate(pieces+filtered+contrasts,2)


def state(raw):
    p,a,b=components(raw)
    t=(np.arange(len(p))-24).clip(0)
    return [compact_filters(p,q) for q in (a,b,a[t],b[t])]


def observe(state,y,d):
    return [source.observe(x,y,d) for x in state]


def fit(hist):
    return dict(w=np.array([source.fit(h)[0]['w'][:,2] for h in zip(*hist)])),dict(dates=len(hist))


def changes(z,model):
    out=[np.concatenate([r.astype('float32')@w for r in np.array_split(a,max(1,(len(a)+127)//128))]) for a,w in zip(z,model['w'])]
    level,move,old_level,old_move=out
    return [.32*level,.32*move,.25*(level+move),.25*(move-old_move),.25*(level-old_level)]


def correction():return trial.correction(sys.modules[__name__])
def layer():return trial.layer(sys.modules[__name__])
def load():return trial.load(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2);trial.run(sys.modules[__name__])
