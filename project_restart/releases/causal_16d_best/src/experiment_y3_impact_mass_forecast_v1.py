"""Predict the size and directional imbalance of high-impact transitions in future twenty-tick windows."""
from functools import lru_cache
import json
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_y3_transition_implied_moves_v1 as event

parent,study,history,trial=event.parent,event.study,event.history,event.trial
OUT=trial.clock.ART/'y3_impact_mass_forecast_v1'
ARMS=('direct future return','sum of activity heads','confirmed large move component','unconfirmed large moves',
      'ordinary moves','confirmed later windows','conditioned beyond reassigned labels','reassigned event labels',
      'fixed expected size','conditional size contribution','independent component readouts')
CANDIDATES=(1,2,3,4,5,6,9,10)
CONTROLS=(0,7,8)


def context(raw,t,st,nets):
    r,s=event.tape(raw,st,nets)
    fields=(0,7,11,14,17,21,24,26,38,40,41,42,45,50,51,54,55,77,78)
    z=np.tanh(np.arcsinh(raw[:,:,fields].astype('float32')))
    z=np.concatenate((z,s,np.tanh(r[:,:,None]).astype('float32')),2)
    cs=np.concatenate((np.zeros_like(z[:1],dtype=float),z.cumsum(0,dtype=float)))
    first=np.where(t>=2401,2401,0)
    def avg(width):
        lo=np.maximum(first,t+1-width)
        return ((cs[t+1]-cs[lo])/(t+1-lo)[:,None,None]).astype('float32')
    a,b=avg(8),avg(64)
    return np.concatenate((z[t],a-b,b,np.broadcast_to(a.mean(1,keepdims=True),a.shape)),2),r,s


def targets(r,s,t,wrong):
    # The tag on a move is an auxiliary training label. At test time no future tag is observed.
    confirmed=(r*s[:,:,0]>0)&(s[:,:,1]>=.5)
    if wrong:
        confirmed=np.roll(confirmed,17,axis=1)
    big=abs(r)>event.CUTOFF
    parts=np.stack((r*(~big),r*big*confirmed,r*big*(~confirmed)),2)
    np.testing.assert_allclose(parts.sum(2),r,atol=1e-8)
    stat=np.stack((parts,abs(parts)),3)
    cs=np.concatenate((np.zeros_like(stat[:1]),stat.cumsum(0,dtype=float)))
    chunks=np.stack([cs[t+h+21]-cs[t+h+1] for h in (0,20,40)],2)
    return (chunks/np.sqrt(20)).astype('float32')


@lru_cache(maxsize=16)
def record(day):
    raw=trial.clock.book.rawday(day)
    # All three targets mature before the end of the same trading session.
    t=np.r_[np.arange(121+day%7,2401-70,7),np.arange(2522+day%7,len(raw)-70,7)][::2]
    st=event.calibration()
    x,r,s=context(raw,t,st,event.restore(st))
    return x.astype('float16'),targets(r,s,t,False),targets(r,s,t,True)


class Forecast(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.local=nn.Sequential(nn.Linear(width,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,32),nn.SiLU())
        self.group=nn.Linear(32,5)
        self.out=nn.Sequential(nn.Linear(64,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,21))

    def forward(self,x):
        z=self.local(x)
        gate=self.group(z)
        message=gate.softmax(2)@(gate.softmax(1).transpose(1,2)@z)
        out=self.out(torch.cat((z,message),2))
        latent=out[:,:,3:].reshape(*x.shape[:2],3,3,2)
        mass=F.softplus(latent[...,0])
        direction=latent[...,1].tanh()
        return out[:,:,:3],mass,direction


@lru_cache(maxsize=7)
def params(end):
    rec=[record(d) for d in range(end)]
    x,real,wrong=[torch.from_numpy(np.concatenate(v)).cuda() for v in zip(*rec)]
    st=dict(event.calibration(),width=x.shape[-1],typical=real[...,1].mean((0,1)).cpu().numpy())
    for i,labels in enumerate((real,real,wrong)):
        torch.manual_seed(28701)
        model=Forecast(x.shape[-1]).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.04)
        for ep in range(6):
            order=torch.randperm(len(x),generator=torch.Generator().manual_seed(701+ep))
            for idx in order.split(48):
                truth=labels[idx]
                parts,gross=truth[...,0],truth[...,1]
                direct,mass,direction=model(x[idx].float())
                loss=F.smooth_l1_loss(direct,parts.sum(3),beta=1.5)
                if i:
                    loss+=.5*F.smooth_l1_loss(mass.log1p(),gross.log1p(),beta=.5)
                    # Weight sign errors by realized activity, which leaves empty bins free of direction labels.
                    share=parts/gross.clamp_min(.001)
                    loss+=.6*((direction-share).square()*gross.clamp_max(5)).mean()
                    loss+=.6*F.smooth_l1_loss(mass*direction,parts,beta=1.)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
            if ep==3:
                opt.param_groups[0]['lr']*=.5
        st.update({f'f{i}_{k}':v.detach().cpu().numpy() for k,v in model.state_dict().items()})
    print('trained through day',end,flush=True)
    return st


def restore(st):
    models=[]
    for i in range(3):
        m=Forecast(int(st['width']))
        m.load_state_dict({k:torch.from_numpy(st[f'f{i}_{k}']) for k in m.state_dict()})
        models.append(m.eval())
    return models


@torch.inference_mode()
def predictions(x,st,models):
    scores=[]
    for k in range(0,len(x),64):
        z=torch.from_numpy(x[k:k+64]).float()
        direct=models[0](z)[0].sum(2)
        _,mass,sign=models[1](z)
        _,bad,wrong=models[2](z)
        parts=(mass*sign)
        total=parts.sum((2,3))
        misplaced=(bad*wrong).sum((2,3))
        fixed=(sign*torch.from_numpy(st['typical'])).sum((2,3))
        summ=parts.sum(2)
        bank=(direct,total,summ[:,:,1],summ[:,:,2],summ[:,:,0],parts[:,:,1:,1].sum(2),
              total-misplaced,misplaced,fixed,total-fixed,summ[:,:,0],summ[:,:,1],summ[:,:,2])
        scores.append(torch.stack(bank,2).numpy())
    return trial.clock.center(np.concatenate(scores)*st['scale'][None,:,None]).astype('float32')


def signals(raw,t,st,nets):
    x,_,_=context(raw,t,st,event.restore(st))
    return predictions(x,st,nets)


def historical_design(day):
    st=params(max(4,day-day%2))
    nets=restore(st)
    cache={}
    def build(raw,p,t):
        key=len(raw),t.tobytes()
        if key not in cache:
            cache[key]=signals(raw,t,st,nets)
        return np.concatenate((trial.clock.center(p[t]),cache[key]),2)
    return build


def fit(rows):
    specs=[(np.array([3+i]),3,.35) for i in range(10)]+[(np.arange(13,16),3,.35)]
    return dict(**parent.book.prev.gram(rows,specs),**params(len(rows)+4))


changes=parent.book.changes


def correction():
    with np.load(OUT/'model.npz') as f:
        st=dict(f)
    models=restore(st)
    def apply(raw,p):
        x=np.concatenate((trial.clock.center(p),signals(raw,np.arange(len(raw)),st,models)),2)
        return changes(x,st)[int(st['pick'])]
    return apply


def layer():
    return study.layer(sys.modules[__name__],parent)


@torch.inference_mode()
def check_day(day,st):
    x,real,_=record(day)
    net=restore(st)[1]
    pieces=[]
    for a in np.array_split(x,8):
        _,mass,side=net(torch.from_numpy(a).float())
        pieces.append(torch.stack((mass,mass*side),-1).numpy())
    predicted=np.concatenate(pieces)
    def correlation(a,b):
        a=a.reshape(-1,3,3)
        b=b.reshape(-1,3,3)
        a=a-a.mean(0)
        b=b-b.mean(0)
        return (a*b).mean(0)/(a.std(0)*b.std(0)).clip(1e-8)
    return [correlation(predicted[...,0],real[...,1]).tolist(),correlation(predicted[...,1],real[...,0]).tolist()]


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
    diag=[check_day(day,params(day-day%2)) for day in range(8,16)]
    report=dict(days=list(range(9,17)),metrics=['gross movement','signed movement'],regimes=['ordinary','confirmed large','other large'],
                intervals=['1–20','21–40','41–60'],daily=diag,mean=np.mean(diag,0).tolist())
    (OUT/'forecast_skill.json').write_text(json.dumps(report,indent=2))
    print('gross and signed movement correlations',np.round(report['mean'],3),flush=True)
