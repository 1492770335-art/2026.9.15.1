"""Pretrain on future upside and downside realized variance, then test its incremental value for Y3."""
from functools import lru_cache
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_y3_ex_ante_error_routing_v1 as risk
import experiment_y3_rank_spacing_calibration_v1 as screen
import y3_pair_incumbent_validation as trial

OUT=trial.clock.ART/'y3_semivariance_pretraining_v1'
ARMS=['total energy only','future energy imbalance','risk conditioned scores','joint risk and direction',
      'stock-reassigned direction','permuted upside/downside labels','joint beyond energy','middle and last imbalance']


def state(raw):
    base=risk.state(raw)
    flags=raw[:,:,83:].astype('float32')-1
    move=trial.clock.center(raw[:,:,0].astype('float32')).clip(-15,15)
    y=np.stack((np.maximum(move,0)**2,np.minimum(move,0)**2),2)
    v=risk.filters.smooth(y,80)
    x=np.concatenate((base,np.log1p(v),np.log1p(risk.filters.smooth(y,16)),risk.filters.smooth(flags,25)),2)
    return x.astype('float16'),y


class Regime(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.stock=nn.Sequential(nn.Linear(width,40),nn.LayerNorm(40),nn.SiLU(),nn.Linear(40,28),nn.SiLU())
        self.read=nn.Sequential(nn.Linear(56,32),nn.SiLU(),nn.Linear(32,6))

    def forward(self,x):
        h=self.stock(x.float())
        return self.read(torch.cat((h,h.mean(1,keepdim=True).expand_as(h)),2)).reshape(len(x),50,3,2)


@lru_cache(maxsize=1)
def pretrain():
    rows=[]
    for d in range(8):
        x,energy=state(trial.clock.book.rawday(d))
        t=np.arange(64+d%9,len(x)-60,9)
        cs=np.concatenate((np.zeros_like(energy[:1]),energy.cumsum(0,dtype=float)))
        target=np.stack([np.log1p((cs[t+b+1]-cs[t+a+1])/(b-a)) for a,b in ((0,20),(20,40),(40,60))],2)
        rows.append((x[t],target.astype('float32')))
    x,y=[torch.as_tensor(np.concatenate(v),device='cuda') for v in zip(*rows)]
    rng=torch.Generator(device='cuda').manual_seed(12013)
    shuffle=torch.rand((*y.shape[:3],1),device='cuda',generator=rng)<.5
    permuted=torch.where(shuffle,y.flip(3),y)
    bank={}
    for name,target in (('real',y),('perm',permuted)):
        torch.manual_seed(3351)
        net=Regime(x.shape[-1]).cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.025)
        for ep in range(8):
            for ids in torch.randperm(len(x),device='cuda').split(48):
                out=net(x[ids])
                error=out-target[ids]
                loss=error.square().mean()+.7*(error[:,:,:,0]-error[:,:,:,1]).square().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.)
                opt.step()
            if ep in (4,6):opt.param_groups[0]['lr']*=.45
        bank.update({name+'/'+k:v.detach().cpu().numpy() for k,v in net.state_dict().items()})
    return bank


@torch.inference_mode()
def views(z,p,bank):
    x=z[0]
    net=Regime(x.shape[-1]).cuda().eval()
    forecasts=[]
    for name in ('real','perm'):
        net.load_state_dict({k.split('/',1)[1]:torch.as_tensor(v) for k,v in bank.items() if k.startswith(name+'/')})
        forecasts.append(np.concatenate([net(torch.from_numpy(chunk).cuda()).cpu().numpy()
                                        for chunk in np.array_split(x,(len(x)+95)//96)]))
    q=trial.clock.center(p)
    def contrast(source):
        a,b=np.moveaxis(source,3,0)
        return np.tanh(2*(a-b)),np.log1p(np.maximum(0,a+b))
    tilt,energy=contrast(forecasts[0])
    counterfeit,_=contrast(forecasts[1])
    q3=q[:,:,2:3]
    def directions(v):
        return np.concatenate((v,v*np.tanh(2*q3)*np.tanh(2*q[:,:,:1])),2)
    scale=trial.clock.center(energy)
    return trial.clock.center(np.concatenate((q,scale*q3,scale*q[:,:,:1],directions(tilt),directions(np.roll(tilt,17,axis=1)),
                                              directions(counterfeit)),2))


def record_with_state(z,p,y,d):
    t=np.arange(63+d%7,len(p)-60,7)
    x=views(z,p,pretrain())[t]
    q=trial.clock.center(p[t,:,2])
    truth=trial.clock.center(y[t,:,2])
    scale=np.sum(q*truth,dtype=float)/np.sum(q*q,dtype=float).clip(1e-6)
    e=truth/np.clip(scale,.4,2.)-q
    x=x.reshape(e.size,-1)
    return x.T@x/len(x),x.T@e.ravel()/len(x)


def fit(hist):
    grams,covs=[np.stack(a) for a in zip(*hist)]
    g=grams.mean(0)
    def solve(cols):
        idx=np.asarray(cols)
        b=covs[:,idx].mean(0)
        safe=b*abs(b)/(abs(b)+covs[:,idx].std(0)/np.sqrt(len(hist))+1e-6)
        w=np.zeros(len(g))
        w[idx]=trial.curve.solve(g[np.ix_(idx,idx)],safe,.7)
        return w
    base=solve(np.arange(9))
    tilt=solve(np.arange(9,15))
    joint=solve(np.arange(15))
    bank=np.column_stack((base,tilt,solve(np.arange(3,9)),joint,solve(np.r_[0:9,15:21]),
                         solve(np.r_[0:9,21:27]),joint-base,solve([10,11,13,14])))
    return {**pretrain(),'readout':(.40*bank).astype('float32')}


def changes_with_reference(z,m,p):
    return list(np.moveaxis(views(z,p,m)@m['readout'],2,0))


def select(f,d):
    choices=[]
    for i in (1,3,6,7):
        v=f[:,:,0,i]
        if ((v>0).all() and (d[:,0,i]>0).sum()>=4 and
            all((v-f[:,:,0,k]).mean(0).min()>0 for k in (0,4,5))):
            choices.append(i)
    return int(max(choices,key=lambda i:f[:,0,0,i].mean())) if choices else None


def correction():
    return trial.previous.previous.prior.correction(sys.modules[__name__])


def layer():
    return trial.layer(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2)
    screen.run(sys.modules[__name__])
