"""Read recent moves against the price movement implied by concurrent discrete-field transitions."""
from functools import lru_cache
import json
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import y3_crossed_level_references as history

parent,study,trial=history.incumbent,history.study,history.trial
OUT=trial.clock.ART/'y3_transition_implied_moves_v1'
ARMS=('observed move','implied move','unexplained move','confirmed move','excess implied move',
      'actual movement opposite implied','high impact confirmed','other stock confirms','current states only',
      'confirmed and unexplained','price and implied directions agree','large price moves alone')
CANDIDATES=(1,2,3,4,5,6,9,10)
CONTROLS=(0,7,8,11)
WIDTH=8
CUTOFF=.75


def encoder():
    return nn.Sequential(nn.Linear(153,80),nn.LayerNorm(80),nn.SiLU(),nn.Linear(80,48),nn.SiLU(),nn.Linear(48,2))


def codes(raw,t,static=False):
    previous=np.maximum(t-1,np.where(t>=2401,2401,0))
    y=raw[t,:,83:].astype('int32')
    old=y if static else raw[previous,:,83:].astype('int32')
    return 3*old+y


def price(raw,k):
    return np.log1p(k*(raw[:,:,0].astype(float)-float(np.float16(.00136))))/k


def encode(c):
    return F.one_hot(c.long(),9).float().flatten(-2)


@lru_cache(maxsize=1)
def calibration():
    k=float(parent.queues.calibration()['k'])
    items=[]
    scales=[]
    for d in range(4):
        raw=trial.clock.book.rawday(d)
        r=price(raw,k)
        scales.append(np.sqrt(np.mean(r*r,axis=0)))
        t=np.arange(1+d%5,len(raw),5)
        items.append((codes(raw,t),r[t]))
    scale=np.mean(scales,axis=0).clip(.08)
    c=torch.from_numpy(np.concatenate([v[0] for v in items]).reshape(-1,17)).cuda()
    r=torch.from_numpy(np.concatenate([v[1]/scale for v in items]).flatten().astype('float32')).cuda()
    state=dict(k=k,scale=scale.astype('float32'))
    for mode in (0,1):
        torch.manual_seed(29171)
        model=encoder().cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.0012,weight_decay=.04)
        values=c if mode==0 else 4*(c%3)
        for ep in range(7):
            for ix in torch.randperm(len(c),device='cuda').split(768):
                out=model(encode(values[ix]))
                y=r[ix]
                loss=F.smooth_l1_loss(out[:,0],y,beta=1.5)+.5*F.binary_cross_entropy_with_logits(out[:,1],(y.abs()>CUTOFF).float())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        state.update({f'n{mode}_{k}':v.detach().cpu().numpy() for k,v in model.state_dict().items()})
    return state


def restore(st):
    nets=[]
    for i in (0,1):
        net=encoder()
        net.load_state_dict({k:torch.from_numpy(st[f'n{i}_{k}']) for k in net.state_dict()})
        nets.append(net.eval())
    return nets


@torch.inference_mode()
def tape(raw,st,nets):
    chunks=[]
    for i in range(0,len(raw),96):
        t=np.arange(i,min(i+96,len(raw)))
        c=torch.from_numpy(codes(raw,t))
        dyn=nets[0](encode(c))
        current=nets[1](encode(4*(c%3)))
        chunks.append(torch.stack((dyn[:,:,0],dyn[:,:,1].sigmoid(),current[:,:,0]),2).numpy())
    return price(raw,float(st['k']))/st['scale'],np.concatenate(chunks)


def features(raw,t,st,nets):
    r,signals=tape(raw,st,nets)
    implied,prob,current=signals.transpose(2,0,1)
    def confirm(a,b):
        return np.sign(a)*np.minimum(abs(a),abs(b))*(a*b>0)
    matched=confirm(r,implied)
    bank=(r,implied,r-implied,matched,implied-matched,r*(r*implied<0),matched*prob,
          confirm(r,np.roll(implied,17,axis=1)),current,r*(r*implied>0),r*(abs(r)>CUTOFF))
    blocks=[]
    marks=np.tanh(raw[:,:,21].astype('float32'))
    session=np.where(t>=2401,2401,0)
    for z in bank:
        cum=np.concatenate((np.zeros((1,50,2)),np.stack((z,abs(z)*marks),2).cumsum(0,dtype=float)))
        b=[]
        for lo,hi in ((0,4),(4,16),(16,48),(48,128)):
            end=np.maximum(t+1-lo,session)
            begin=np.maximum(t+1-hi,session)
            b.append((cum[end]-cum[begin])/np.sqrt(hi-lo))
        blocks.append(np.concatenate(b,2))
    return trial.clock.center(np.concatenate(blocks,2).astype('float32'))


def historical_design(day):
    st=calibration()
    nets=restore(st)
    cache={}
    def build(raw,p,t):
        key=len(raw),t.tobytes()
        if key not in cache:
            cache[key]=features(raw,t,st,nets)
        return np.concatenate((trial.clock.center(p[t]),cache[key]),2)
    return build


def fit(rows):
    g=[np.arange(3+WIDTH*i,3+WIDTH*(i+1)) for i in range(11)]
    specs=[(v,3,.35) for v in (*g[:9],np.r_[g[3],g[2]],g[9],g[10])]
    return dict(**parent.book.prev.gram(rows,specs),**calibration())


changes=parent.book.changes


def correction():
    with np.load(OUT/'model.npz') as f:
        st=dict(f)
    nets=restore(st)
    def apply(raw,p):
        x=np.concatenate((trial.clock.center(p),features(raw,np.arange(len(raw)),st,nets)),2)
        return changes(x,st)[int(st['pick'])]
    return apply


def layer():
    return study.layer(sys.modules[__name__],parent)


def audit():
    st=calibration()
    nets=restore(st)
    daily=[]
    for d in range(8,16):
        r,s=tape(trial.clock.book.rawday(d),st,nets)
        q=s[:,:,1]
        high=q>=np.quantile(q,.9)
        low=q<=np.quantile(q,.1)
        event=abs(r)>CUTOFF
        daily.append([np.corrcoef(r.ravel(),s[:,:,i].ravel())[0,1] for i in (0,2)]
                     +[event.mean(),event[high].mean(),event[low].mean(),(r[high]**2).sum()/(r**2).sum()])
    report=dict(days=list(range(9,17)),metrics=['implied_move_corr','current_only_corr','large_move_rate',
                'top_probability_rate','bottom_probability_rate','top_probability_squared_move_share'],
                daily=np.array(daily).tolist(),mean=np.mean(daily,0).tolist())
    (OUT/'event_diagnostic.json').write_text(json.dumps(report,indent=2))
    print('concurrent event identification',report['mean'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    audit()
    study.score_run(sys.modules[__name__],reference=history)
