"""Price ordered shocks in reconstructed touch liquidity, executions and quote activity."""
from functools import lru_cache
import sys
import numpy as np
import torch
import experiment_y3_independent_order_clocks_v1 as clocks
import y3_medium_order_clock_references as history
import y3_reconstructed_touch_queues as queues

trial,study,parent=history.trial,history.study,history.incumbent
OUT=trial.clock.ART/'y3_touch_flow_clocks_v1'
ARMS=('simultaneous book','pressure before activity','activity before pressure','order contrast',
      'signed and unsigned order','late contrast','quote price changes ignored','other stock activity',
      'chronological components separately','middle less early response')
CANDIDATES=(1,2,3,4,5,8,9)
CONTROLS=(0,6,7)
SPANS=(4,16,64)


def inputs(raw,st):
    mid,half,_,_,valid,_,buy,sell=queues.tape.prices(raw,st)
    bid,ask=queues.size(raw,st)
    volume=st['volume_scale']
    den=.3*volume+(bid+ask)
    prev=np.maximum(np.arange(len(raw))-1,0)
    if len(raw)>2401:
        prev[2401]=2401
    active=valid&valid[prev]
    old_b,old_a=mid[prev]-half[prev],mid[prev]+half[prev]
    now_b,now_a=mid-half,mid+half
    b=np.where(now_b>=old_b,bid,0)-np.where(now_b<=old_b,bid[prev],0)
    a=np.where(now_a<=old_a,ask,0)-np.where(now_a>=old_a,ask[prev],0)
    free_b=(bid-bid[prev]+sell)*(now_b==old_b)
    free_a=(ask-ask[prev]+buy)*(now_a==old_a)
    def clip(x):
        return np.tanh(x)*active
    pressure=np.stack((clip((b-a)/den[prev]),clip(2*(bid-ask)/den),clip((buy-sell)/(.3*volume+den[prev])),
                       clip((free_b-free_a)/den[prev])),2)
    naive=pressure.copy()
    naive[:,:,0]=clip((bid-bid[prev]-ask+ask[prev])/den[prev])
    naive[:,:,3]=clip((bid-bid[prev]+sell-ask+ask[prev]-buy)/den[prev])
    activity=np.stack((clip((buy+sell)/den[prev]),clip(abs(b)+abs(a)),clip(np.log1p(half.clip(0))/2)),2)
    return np.concatenate((pressure,naive,activity),2).astype('float32')


def bank(raw,t,st,weights=None):
    vals=inputs(raw,st)
    results=[]
    for lo,hi in ((0,min(len(raw),2401)),(2401,len(raw))):
        idx=t[(t>=lo)&(t<hi)]-lo
        if not len(idx):
            continue
        v=vals[lo:hi]-clocks.filters.smooth(vals[lo:hi],128)
        p,n,a=v[:,:,:4],v[:,:,4:8],v[:,:,8:]
        def mul(p,a):
            return (p[:,:,:,None]*a[:,:,None,:]).reshape(*p.shape[:2],12)
        out=np.zeros((len(idx),50,6,108 if weights is None else 3),dtype='float32')
        for k,gap in enumerate(SPANS):
            past=clocks.filters.smooth(v,gap)
            lag=np.concatenate((np.zeros_like(past[:1]),past[:-1]))
            x,y=mul(lag[:,:,:4],a),mul(p,lag[:,:,8:])
            plain=mul(lag[:,:,4:8],a)-mul(n,lag[:,:,8:])
            wrong=mul(lag[:,:,:4],np.roll(a,17,axis=1))-mul(p,np.roll(lag[:,:,8:],17,axis=1))
            z=np.stack((mul(p,a),x,y,x-y,plain,wrong),2)
            for j,age in enumerate(SPANS):
                pos=12*(3*k+j)
                if weights is None:
                    out[:,:,:,pos:pos+12]=clocks.filters.smooth(z,age)[idx]
                else:
                    reduced=np.einsum('tsci,cij->tscj',z,weights[:,pos:pos+12],optimize=True)
                    out+=clocks.filters.smooth(reduced,age)[idx]
        results.append(trial.clock.center(out))
    return np.concatenate(results)


@lru_cache(maxsize=16)
def record(d):
    raw=trial.clock.book.rawday(d)
    t=np.arange(90+d%13,len(raw)-60,13)
    t=t[(t+60<2401)|(t>=2401)]
    x=bank(raw,t,queues.calibration()).reshape(-1,6,108).transpose(1,0,2).copy()
    y=trial.clock.labels(d).reshape(-1,50,3)
    target=trial.clock.center(np.stack([y[t+h,:,1] for h in (0,20,40)],2)).clip(-4,4).reshape(len(t)*50,3)
    return x.transpose(0,2,1)@x/len(target),x.transpose(0,2,1)@target/len(target)


@lru_cache(maxsize=7)
def params(end):
    gs,bs=[np.asarray(v,dtype=float) for v in zip(*[record(d) for d in range(end)])]
    g=gs.mean(0)
    b=bs.mean(0)
    b*=abs(b)/(abs(b)+bs.std(0)/np.sqrt(end)+1e-6)
    ridge=np.eye(108)[None]*g.diagonal(axis1=1,axis2=2).clip(.00001)[:,None,:]
    return np.linalg.solve(g+1.05*ridge,b).astype('float32')


def signals(raw,t,st,w):
    x=bank(raw,t,st,w)
    return np.concatenate((x.reshape(len(t),50,-1),x[:,:,3,1:2]-x[:,:,3,0:1]),2)


def historical_design(d):
    w=params(max(4,d-d%2))
    cache={}
    def build(raw,p,t):
        key=len(raw),t.tobytes()
        if key not in cache:
            cache[key]=signals(raw,t,queues.calibration(),w)
        return np.concatenate((trial.clock.center(p[t]),cache[key]),2)
    return build


def fit(rows):
    s=[np.arange(3+3*i,6+3*i) for i in range(6)]
    groups=[*s[:4],np.r_[s[0],s[3]],s[3][1:],s[4],s[5],np.r_[s[1],s[2]],np.array([21])]
    return dict(**clocks.linear.gram(rows,[(i,3,.35) for i in groups]),heads=params(len(rows)+4),**queues.calibration())


changes=clocks.changes


def correction():
    with np.load(OUT/'model.npz') as archive:
        st=dict(archive)
    def apply(raw,p):
        z=signals(raw,np.arange(len(raw)),st,st['heads'])
        return changes(np.concatenate((trial.clock.center(p),z),2),st)[int(st['pick'])]
    return apply


def layer():
    return study.layer(sys.modules[__name__],parent)


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
