"""Locate the part of a historical price jump supported by the contemporaneous state transition."""
import sys
import numpy as np
import torch
import experiment_y3_transition_implied_moves_v1 as event

history,study,trial,parent=event.history,event.study,event.trial,event.parent
OUT=trial.clock.ART/'y3_implied_crossing_depth_v1'
ARMS=('existing pressure confirmation','uniform supported fraction','near starting price','near ending price',
      'unexplained outer levels','near start with pressure support','outer levels with pressure support',
      'permuted stock implied move','static state implied move','relative location inside the jump',
      'unexplained confirmed by order flow','both supporting and unexplained outer levels')
CANDIDATES=(2,3,4,5,6,9,10,11)
CONTROLS=(0,1,7,8)


def calibration():
    return dict(**{k:v for k,v in event.calibration().items() if k!='k'},**parent.queues.calibration())


def summarize(r,signals,mark,t,h):
    ids=t[:,None]-np.arange(h-1,-1,-1)
    valid=ids>=np.where(t>=2401,2401,0)[:,None]
    ids=ids.clip(0)
    change=r[ids].transpose(0,2,1)*valid[:,None]
    path=change.cumsum(2)
    sign=np.sign(path[:,:,-1:])
    extent=abs(path[:,:,-1:])
    start=(path-change)*sign
    end=path*sign
    hi=np.minimum.accumulate(end[:,:,::-1],2)[:,:,::-1]
    lo=np.maximum(start,0)
    w=(hi-lo).clip(0)
    marks=mark[ids].transpose(0,2,1,3)
    pressure=sign*marks[:,:,:,0]>0
    flow=sign*marks[:,:,:,1]>0
    def partition(imp):
        length=np.minimum((imp[ids].transpose(0,2,1)*sign).clip(0),(end-start).clip(0))
        left=(np.minimum(hi,start+length)-lo).clip(0)
        right=(hi-np.maximum(lo,end-length)).clip(0)
        return left,right,w*length/(end-start).clip(1e-8)
    left,right,uniform=partition(signals[:,:,0])
    wrong=partition(np.roll(signals[:,:,0],17,axis=1))[0]
    static=partition(signals[:,:,1])[0]
    np.testing.assert_array_less(left,w+1e-6)
    out=w-left
    values=(w*pressure,uniform,left,right,out,left*pressure,out*pressure,wrong,static,left-right,out*flow)
    def read(a):
        a=a/(np.sqrt(h)*(1+np.sqrt(extent/np.sqrt(h))))
        z=np.einsum('bsh,bshf->bsf',a,marks)
        z[:,:,2:]*=sign
        age=(a*(np.arange(h)/h)).sum(2,keepdims=True)
        return np.concatenate((sign*a.sum(2,keepdims=True),sign*age,z),2)
    return [read(v) for v in values]


def features(raw,t,st,nets):
    r,marks=parent.signals(raw,st)
    _,scores=event.tape(raw,st,nets)
    signals=scores[:,:,[0,2]]*st['scale'][None,:,None]
    groups=[]
    for i in range(0,len(t),32):
        spans=[summarize(r,signals,marks,t[i:i+32],h) for h in parent.WINDOWS]
        groups.append(np.concatenate([np.concatenate(z,2) for z in zip(*spans)],2))
    return trial.clock.center(np.concatenate(groups).astype('float32'))


def historical_design(d):
    st=calibration()
    nets=event.restore(st)
    cache={}
    def build(raw,p,t):
        key=len(raw),t.tobytes()
        if key not in cache:
            cache[key]=features(raw,t,st,nets)
        return np.concatenate((trial.clock.center(p[t]),cache[key]),2)
    return build


def fit(rows):
    width=6*len(parent.WINDOWS)
    sets=[np.arange(3+i*width,3+(i+1)*width) for i in range(11)]
    groups=sets+[np.r_[sets[5],sets[6]]]
    return dict(**parent.book.prev.gram(rows,[(s,3,.35) for s in groups]),**calibration())


changes=parent.book.changes


def load_correction(directory):
    with np.load(directory/'model.npz') as f:
        st=dict(f)
    nets=event.restore(st)
    def apply(raw,p):
        x=np.concatenate((trial.clock.center(p),features(raw,np.arange(len(raw)),st,nets)),2)
        return changes(x,st)[int(st['pick'])]
    return apply


def correction():
    return load_correction(OUT)


def layer():
    return study.layer(sys.modules[__name__],parent)


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
