"""Marginalize future price-move counts to predict cash with a stock exchange encoder."""
from functools import lru_cache
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dense_antisymmetric_asset_exchange_v1 as pre

peer,direct,strip,event,onset,rel = pre.peer,pre.direct,pre.strip,pre.event,pre.onset,pre.rel
REFERENCE,read_p = pre.REFERENCE,pre.read_p
OUT = REFERENCE.parent/'dense_exchange_count_mixture_v1'
ARMS = ('direct_band_cash','count_conditional_cash','timing_premium')
C,D = 6,15
L = pre.H+2*pre.K
BANDS = ((0,5),(5,20),(20,60))
CUTS = ((1,2,3),(2,4,7),(6,11,18))
DEVICE = pre.DEVICE


@torch.inference_mode()
def tokens(x,model,bank):
    rows = []
    for a in range(0,len(x),64):
        z = torch.tensor(((x[a:a+64]-bank[:pre.N])/bank[pre.N:2*pre.N]).clip(-6,6),device=DEVICE,dtype=torch.float32)
        n = len(z)
        if n<64:
            z = F.pad(z,(0,0,0,0,0,64-n))
        # Keep the uncentered stock token so liquidity levels are visible to the timing head.
        _,b,c = model.token_views(z)
        rows.append(torch.cat((model.encode(z),b,c),dim=-1)[:n].cpu().numpy())
    return np.concatenate(rows)


@lru_cache(maxsize=16)
def examples(day):
    raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
    data = []
    bank = pre.encoder()
    model = pre.restore(bank)
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(50+day%5,len(m)-60,5)
        gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        keep = gaps[t+61]==gaps[t]
        price = m.cumsum(0)
        count = (abs(m)>.001).cumsum(0)
        cash = np.stack([price[t+hi]-price[t+lo] for lo,hi in BANDS],axis=-1)
        cls = np.stack([np.digitize(count[t+hi]-count[t+lo],cut) for (lo,hi),cut in zip(BANDS,CUTS)],axis=-1)
        x = tokens(pre.inputs(raw[a:b])[t],model,bank)
        data.append((x,cash.astype(np.float32),cls,keep))
    return tuple(map(np.concatenate,zip(*data)))


def network(width):
    return nn.Sequential(nn.Linear(L,40),nn.LayerNorm(40),nn.SiLU(),nn.Linear(40,width))


@lru_cache(maxsize=5)
def fit_heads(end):
    x,y,cls,live = map(np.concatenate,zip(*(examples(d) for d in range(end))))
    scale = y[live].std(0).clip(.02)
    prior = np.stack([(cls==k).astype(float) for k in range(4)],axis=-1)
    prior = ((prior*live[...,None,None]).sum(0)+.5)/(live.sum(0)[:,None,None]+2.)
    mean,sd = x[live].mean(0),x[live].std(0).clip(.008)
    x = torch.tensor(((x-mean)/sd).clip(-6,6),device=DEVICE)
    y = torch.tensor((y/scale).clip(-6,6),device=DEVICE)
    cls = torch.tensor(cls,device=DEVICE)
    live = torch.tensor(live,device=DEVICE)
    models = []
    for mixture,width in ((False,3),(True,24)):
        torch.manual_seed(79203)
        model = network(width).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.04)
        for _ in range(460):
            ix = torch.randint(len(x),(36,),device=DEVICE)
            pred = model(x[ix])
            target = y[ix]
            good = live[ix]
            yy = target[good]
            if mixture:
                logits,marks = pred[good].reshape(-1,3,2,4).unbind(-2)
                key = cls[ix][good]
                expected = (logits.softmax(-1)*marks).sum(-1)
                realized = marks.gather(-1,key[...,None]).squeeze(-1)
                loss = F.cross_entropy(logits.reshape(-1,4),key.reshape(-1))
                loss += F.mse_loss(realized,yy)+.5*F.mse_loss(expected,yy)+.008*marks.square().mean()
            else:
                loss = F.mse_loss(pred[good],yy)+.008*pred.square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        models.append(torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy())
    return np.concatenate((mean,sd,scale,prior.ravel(),*models))


def load_heads(bank):
    offset = 2*L+3+50*3*4
    models = []
    for width in (3,24):
        model = network(width).to(DEVICE).eval()
        n = sum(p.numel() for p in model.parameters())
        torch.nn.utils.vector_to_parameters(torch.tensor(bank[offset:offset+n],device=DEVICE,dtype=torch.float32),model.parameters())
        offset += n
        models.append(model)
    assert offset==len(bank)
    return models


@torch.inference_mode()
def estimates(x,head,models,diagnostics=False):
    prior = torch.tensor(head[2*L+3:2*L+603].reshape(50,3,4),device=DEVICE,dtype=torch.float32)
    scale = torch.tensor(head[2*L:2*L+3],device=DEVICE,dtype=torch.float32)
    rows = []
    for a in range(0,len(x),64):
        z = torch.tensor(((x[a:a+64]-head[:L])/head[L:2*L]).clip(-6,6),device=DEVICE,dtype=torch.float32)
        n = len(z)
        if n<64:
            z = F.pad(z,(0,0,0,0,0,64-n))
        direct = models[0](z)
        logits,marks = models[1](z).reshape(64,50,3,2,4).unbind(-2)
        prob = logits.softmax(-1)
        mixture = (prob*marks).sum(-1)
        timing = ((prob-prior)*marks).sum(-1)
        v = torch.cat((direct*scale,mixture*scale,timing*scale),dim=-1)
        if diagnostics:
            v = torch.cat((v,prob.flatten(-2)),dim=-1)
        rows.append(v[:n].cpu().numpy())
    return np.concatenate(rows)


def features(x,p,head,models):
    predicted = strip.center(estimates(x,head,models))
    # Own relative forecasts and the stock's broad price sensitivity are observable controls.
    return np.concatenate((strip.center(p.astype(float)),strip.center(x[:,:,:3].astype(float)),predicted),axis=2)


@lru_cache(maxsize=12)
def record(day):
    x,_,_,keep = examples(day)
    grid,y = [onset.settled.load(day,k) for k in ('time','y')]
    t = np.concatenate([a+np.arange(50+day%5,b-a-60,5) for a,b in rel.fields.parent.sessions(grid)])
    cutoff = max(v for v in (4,6,8,12) if v<=day)
    head = fit_heads(cutoff)
    f = features(x,read_p(day)[t],head,load_heads(head))
    z = np.concatenate((f,y[t,:,1:]),axis=2)
    z -= ((z*keep[...,None]).sum(1)/keep.sum(1)[:,None].clip(1))[:,None]
    flat = z[keep]
    return flat.T@flat/len(flat)


def train(end):
    co = direct.edge.cycle.ref.fit(np.stack([record(d) for d in range(4,end)]),C,D)
    frozen = pre.encoder()
    return np.r_[len(frozen),frozen,fit_heads(end),co.ravel()]


def signals(raw,grid,p,params):
    cut = 1+int(params[0])
    frozen,head = params[1:cut],params[cut:-D*6]
    model,models = pre.restore(frozen),load_heads(head)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        x = tokens(pre.inputs(raw[a:b]),model,frozen)
        f = features(x,p[a:b],head,models)
        out[a:b] = (f@params[-D*6:].reshape(D,6)).reshape(b-a,50,3,2)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
