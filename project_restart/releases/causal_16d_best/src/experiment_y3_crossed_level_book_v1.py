"""Read the book at ticks that last carried each surviving level of the recent price displacement."""
import sys
import numpy as np
import torch

import y3_reconstructed_touch_queues as queues
import y3_recipient_exposure_references as history

trial,study,book = history.trial,history.study,history.incumbent.book
OUT = trial.clock.ART/'y3_crossed_level_book_v1'
ARMS = ('ordinary price changes','last crossing book','first crossing book','last versus first book',
        'last beyond all price moves','recent confirmed crossings','crossings contradicted by pressure',
        'wrong tick book at the same levels','displacement and current book','all last and first features')
CANDIDATES = (1,2,3,4,5,6,9)
CONTROLS = (0,7,8)
WINDOWS = (32,96)
WIDTH = 6*len(WINDOWS)


def signals(raw,st):
    _,_,_,_,_,_,buy,sell = queues.tape.prices(raw,st)
    bid,ask = queues.size(raw,st)
    vol = st['volume_scale']
    pressure = (bid-ask)/(bid+ask+.001)
    flow = np.tanh((buy-sell)/vol)
    depth = np.tanh(np.log1p((bid+ask)/vol)-2)
    spread = np.tanh(np.arcsinh(raw[:,:,40].astype('float32')))
    k = float(st['k'])
    r = np.log1p(k*(raw[:,:,0].astype(float)-float(np.float16(.00136))))/k
    return r,np.stack((pressure,flow,depth,spread),2).astype('float32')


def summarize(t,h,r,bookmarks):
    ids = t[:,None]-np.arange(h-1,-1,-1)
    valid = ids>=np.where(t<2401,0,2401)[:,None]
    ids = ids.clip(0)
    inc = r[ids].transpose(0,2,1)*valid[:,None,:]
    level = inc.cumsum(2)
    start = level-inc
    end = level[:,:,-1:]
    sign = np.sign(end)
    p = level*sign
    prior = start*sign
    last = (np.minimum.accumulate(p[:,:,::-1],2)[:,:,::-1]-prior.clip(0)).clip(0)
    first = (np.minimum(p,abs(end))-np.maximum.accumulate(prior.clip(0),2)).clip(0)
    np.testing.assert_allclose(last.sum(2),abs(end[:,:,0]),atol=2.e-7)
    marks = bookmarks[ids].transpose(0,2,1,3)
    age = np.arange(h)/h
    def read(weights,marks):
        # Length in price units, tempered to avoid undue weight on a single volatile name.
        weights = weights/np.sqrt(h)/(1+np.sqrt(abs(end)/np.sqrt(h)))
        signed = weights*sign
        return np.stack((signed.sum(2),(signed*age).sum(2),
                         (weights*marks[:,:,:,0]).sum(2),(weights*marks[:,:,:,1]).sum(2),
                         (signed*marks[:,:,:,2]).sum(2),(signed*marks[:,:,:,3]).sum(2)),2)
    # Normalize absolute-tick weights to sum to the same displacement as the crossing weights.
    gross = abs(inc)
    gross *= abs(end)/gross.sum(2,keepdims=True).clip(1.e-8)
    ordinary = read(gross,marks)
    l,f = read(last,marks),read(first,marks)
    agree = sign*marks[:,:,:,0]>0
    wrong = np.roll(marks,h//2,axis=2)
    return (ordinary,l,f,l-f,l-ordinary,read(last*agree,marks),read(last*(~agree),marks),
            read(last,wrong),read(last,np.broadcast_to(marks[:,:,-1:],marks.shape)),np.concatenate((l,f),2))


def features(raw,t,st):
    r,m = signals(raw,st)
    output = []
    for start in range(0,len(t),40):
        v = t[start:start+40]
        groups = zip(*(summarize(v,h,r,m) for h in WINDOWS))
        output.append(np.concatenate([np.concatenate(a,axis=2) for a in groups],2).astype('float32'))
    return trial.clock.center(np.concatenate(output))


def design(raw,p,t,st):
    return np.concatenate((trial.clock.center(p[t]),features(raw,t,st)),2)


def historical_design(day):
    st = queues.calibration()
    cache = {}
    def build(raw,p,t):
        key = len(raw),t.tobytes()
        if key not in cache:
            cache[key] = features(raw,t,st)
        return np.concatenate((trial.clock.center(p[t]),cache[key]),2)
    return build


def fit(rows):
    specs = []
    pos = 3
    for i in range(len(ARMS)):
        width = WIDTH*(2 if i==9 else 1)
        specs.append((np.arange(pos,pos+width),3,.35))
        pos += width
    return dict(**book.prev.gram(rows,specs),**queues.calibration())


changes = book.changes


def correction():
    with np.load(OUT/'model.npz') as f:
        st = dict(f)
    def apply(raw,p):
        return changes(design(raw,p,np.arange(len(raw)),st),st)[int(st['pick'])]
    return apply


def layer():
    return study.layer(sys.modules[__name__],history.incumbent)


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
