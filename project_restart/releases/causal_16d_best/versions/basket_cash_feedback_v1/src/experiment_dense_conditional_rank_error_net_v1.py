"""Predict and price ordering mistakes among stocks with similar incumbent Y2 and Y3 forecasts."""
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dense_multisize_portfolio_net_v1 as features

base,REFERENCE,rel,onset=features.base,features.REFERENCE,features.rel,features.onset
OUT=REFERENCE.parent/"dense_conditional_rank_error_net_v1"
ARMS=("matched_rank_error","matched_return_error","consistent_rank_and_return_error")
WIDTH=83+34+features.WIDTH+3


def inputs(raw,grid,p,t):
    stock=p[t].astype(np.float32)
    stock-=stock.mean(1,keepdims=True)
    book=np.tanh(.55*np.arcsinh(raw[t,:,:83].astype(np.float32)))
    categories=raw[t,:,83:]
    return np.concatenate((book,categories==0,categories==2,features.state(raw,grid)[t],stock),axis=-1)


def weights(p):
    # Symmetric weights make gains to one name and losses to its peers sum to zero.
    f=p[:,:,-2:]
    f=f/(.08+f.std(1,keepdim=True))
    dist=(f[:,:,None]-f[:,None,:]).square().sum(-1)
    w=torch.exp(-2.5*dist)
    w=w*(1-torch.eye(f.shape[1],device=f.device))
    total=w.sum(-1).clamp_min(1)
    return w/torch.maximum(total[:,:,None],total[:,None,:])


def targets(p,y):
    w=weights(p)
    diff=y[:,:,None,:]-y[:,None,:,:]
    wins=(w[:,:,:,None]*torch.tanh(diff/.6)).sum(2)
    payoff=(w[:,:,:,None]*diff.clamp(-3,3)).sum(2)
    return torch.cat((wins,payoff),dim=-1)


def collect():
    OUT.mkdir(exist_ok=True)
    ratio=base.panel.ratio()
    for d in range(4,16):
        p=base.read_p(d)
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        ticks=np.concatenate([np.arange(a+d%5,b-60,5) for a,b in rel.fields.parent.sessions(grid)])
        x=inputs(raw,grid,p,ticks)
        truth=y[ticks].astype(np.float32)
        target=np.stack((truth[...,2],truth[...,2]-ratio*truth[...,1]),-1)
        obs=targets(torch.from_numpy(x[...,-3:]),torch.from_numpy(target)).numpy()
        np.savez(OUT/f"day_{d:02d}.npz",x=x.astype(np.float16),target=obs.astype(np.float16))


class RankNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(WIDTH,48),nn.LayerNorm(48),nn.SiLU(),
                                  nn.Linear(48,24),nn.SiLU(),nn.Linear(24,4))

    def forward(self,x):
        return self.layers(x)


def train(end):
    records=[]
    for day in range(4,end):
        with np.load(OUT/f"day_{day:02d}.npz") as f:
            records.append((f["x"],f["target"]))
    x,y=[torch.from_numpy(np.concatenate(c)).to(features.DEVICE).float() for c in zip(*records)]
    scale=x.square().mean((0,1)).sqrt().clamp_min(.08)
    x=(x/scale).clamp(-5,5)
    torch.manual_seed(24073)
    model=RankNet().to(features.DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0011,weight_decay=.07)
    for step in range(340):
        ix=torch.randint(len(x),(48,),device=x.device)
        pred=model(x[ix])
        loss=F.smooth_l1_loss(pred,y[ix],beta=.5)+.25*pred.square().mean()
        if step==210:
            optimizer.param_groups[0]["lr"]*=.4
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    return np.concatenate([scale.cpu().numpy(),*[v.detach().cpu().numpy().ravel() for v in model.parameters()]])


def agree(a,b):
    return torch.where(a*b>0,a.sign()*torch.minimum(a.abs(),b.abs()),0.)


@torch.inference_mode()
def signals(raw,grid,p,bank):
    unit=torch.as_tensor(bank[:WIDTH]).double()
    net=RankNet().double().eval()
    offset=WIDTH
    for v in net.parameters():
        v.copy_(torch.as_tensor(bank[offset:offset+v.numel()].reshape(v.shape)))
        offset+=v.numel()
    out=[]
    # Keep the temporary 100-field tensors bounded while computing the history only once.
    history=features.state(raw,grid)
    for a in range(0,len(p),64):
        sl=slice(a,a+64)
        book=np.tanh(.55*np.arcsinh(raw[sl,:,:83].astype(np.float32)))
        cats=raw[sl,:,83:]
        f=p[sl].astype(np.float32)
        f-=f.mean(1,keepdims=True)
        x=torch.as_tensor(np.concatenate((book,cats==0,cats==2,history[sl],f),axis=-1)).double()
        z=net((x/unit).clamp(-5,5))
        rank=agree(z[...,0],z[...,1])
        pay=agree(z[...,2],z[...,3])
        s=torch.stack((rank,pay,agree(rank,pay)),dim=-1)
        w=weights(x[...,-3:])
        out.append((w.sum(2,keepdim=True)*s-w@s).numpy())
    return np.concatenate(out)


def predict(raw,grid,p,params):
    q=p.copy()
    z=signals(raw,grid,p,params["bank"])
    q[...,2]+=(params["strength"]*z[...,int(params["choice"])]).astype(q.dtype)
    return q


if __name__=="__main__":
    torch.set_num_threads(1)
    collect()
    base.rolling(sys.modules[__name__])
