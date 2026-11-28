"""Predict which price boundary is crossed first and the terminal return conditional on that event."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from functools import lru_cache
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as tf

from experiment_dense_market_conditional_graph_risk_v1 import event,onset,rel,strip
import experiment_dense_book_state_pretrain_v1 as src

base,ref,REFERENCE,read_p = src.base,src.ref,src.REFERENCE,src.read_p
OUT = REFERENCE.parent/'dense_first_exit_state_net_v1'
TARGETS = (0,1,2)
ARMS = ('temporal_exit_payoff','peer_exit_payoff','joint_exit_payoff')
H = np.array([5,20,60])
R = np.sqrt(H/5)
C = 6


class ExitNet(src.Encoder):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(src.H,18) for _ in range(3)])


@lru_cache(maxsize=12)
def examples(day):
    raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
    unit = base.risk.walk.widths()
    rows = []
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(40+day%12,b-a-60,12)
        tape,_ = src.stream(raw[a:b],m,ok,src.axes(),unit)
        path = m.cumsum(0)/unit
        gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        increments = path[t[:,None]+np.arange(1,61)]-path[t,None]
        sides,amounts,valid = [],[],[]
        for h,r in zip(H,R):
            change = increments[:,:h]/r
            hits = abs(change)>=1.
            loc = hits.argmax(1)
            which = np.take_along_axis(change,loc[:,None,:],axis=1)[:,0]
            sides.append(np.where(hits.any(1),1+np.sign(which),1))
            amounts.append(change[:,-1].clip(-6,6))
            valid.append(gaps[t+h+1]==gaps[t-40])
        rows.append((src.tokens(tape,t),np.stack(sides,2).astype(int),np.stack(amounts,2).astype(np.float32),
                     np.stack(valid,2)))
    return tuple(torch.from_numpy(np.concatenate(v)) for v in zip(*rows))


def losses(raw,side,amount,valid):
    v = raw.reshape(*side.shape,6)
    prob,means = v[...,:3],v[...,3:]
    err = -prob.log_softmax(-1).gather(-1,side[...,None]).squeeze(-1)
    payoff = means.gather(-1,side[...,None]).squeeze(-1)
    expected = (prob.softmax(-1)*means).sum(-1)
    err += .40*tf.smooth_l1_loss(payoff,amount,reduction='none',beta=1.)
    err += .15*(expected-amount).square()
    return (err*valid).sum()/valid.sum().clamp_min(1)


@lru_cache(maxsize=4)
def pretrained(end):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(17391)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ExitNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(),lr=.0025,weight_decay=.015)
    batch = [torch.cat(v) for v in zip(*(examples(d) for d in range(end)))]
    rng = np.random.default_rng(17391)
    for step in range(600):
        ix = rng.integers(0,len(batch[0]),20)
        x,side,amount,valid = [z[ix].to(dev) for z in batch]
        level,temporal,peer = model(x).unbind(2)
        loss = losses(level,side,amount,valid)+losses(level.detach()+temporal,side,amount,valid)
        loss += losses((level+temporal).detach()+peer,side,amount,valid)
        loss += .025*(temporal.square().mean()+peer.square().mean())
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),2.)
        opt.step()
    print(f'first-exit network trained on days 1-{end}, device={dev}',flush=True)
    return np.concatenate([w.detach().cpu().numpy().ravel() for w in model.parameters()])


def restore(flat):
    enc = ExitNet().double()
    i = 0
    with torch.no_grad():
        for w in enc.parameters():
            j = i+w.numel()
            w.copy_(torch.from_numpy(flat[i:j].reshape(w.shape)))
            i = j
    return enc.eval()


def channels(parts):
    v = parts.cumsum(2).reshape(*parts.shape[:3],3,6)
    logits = v[...,:3]
    exp = np.exp(logits-logits.max(axis=-1,keepdims=True))
    prob = exp/exp.sum(-1,keepdims=True)
    means = v[...,3:]
    pay = (prob*means).sum(-1)
    # The event-probability and conditional-size changes sum to the total expected-return revision.
    outputs = []
    for i,j in ((0,1),(1,2),(0,2)):
        event = ((prob[:,:,j]-prob[:,:,i])*means[:,:,i]).sum(-1)
        size = (prob[:,:,j]*(means[:,:,j]-means[:,:,i])).sum(-1)
        outputs.append(np.concatenate((event,size),axis=2))
    return pay[:,:,0],outputs


def features(p,parts,valid,unit):
    current,extra = channels(parts)
    ctrl = np.concatenate((p,current*unit[:,None]*R),axis=2)
    return [base.links.center(np.concatenate((ctrl,e*unit[:,None]*np.tile(R,2)),axis=2),valid) for e in extra]


def record(day,enc):
    raw,grid,y = [onset.settled.load(day,k) for k in ('x','time','y')]
    p = read_p(day)
    beta = strip.daily_scale(p,y).clip(.2,4.)
    c = strip.physical()[1]
    err = y/beta-p
    far = err.copy()
    far[...,1:] -= err[...,:-1]*(c[1:]/c[:-1]*beta[:-1]/beta[1:])
    gram = np.zeros((3,18,18))
    n = 0
    unit = base.risk.walk.widths()
    for a,b,m,ok in event.sessions(raw,grid):
        tape,_ = src.stream(raw[a:b],m,ok,src.axes(),unit)
        for ix,parts in src.views(tape,np.arange(day%8,b-a-60,8),enc):
            mask = ok[ix]
            labels = base.links.center(np.concatenate((err[a+ix],far[a+ix]),axis=2),mask)
            for j,z in enumerate(features(p[a+ix],parts,mask,unit)):
                v = np.concatenate((z,labels),axis=2)[mask]
                gram[j] += v.T@v
            n += mask.sum()
    return gram/n


def train(end):
    before = max(4,end-4)
    flat = pretrained(before)
    enc = restore(flat)
    dates = np.stack([record(d,enc) for d in range(before,end)])
    co = []
    for j in range(3):
        g = dates[:,j].mean(0)
        w = base.links.base.whitening(g[:C,:C],.04)
        v = np.eye(18)[:,C:12].copy()
        v[:C] = -w@(w.T@g[:C]@v)
        v = v@base.links.base.whitening(v.T@g@v,.65)
        scores = v.T@dates[:,j,:,12:]
        mu = scores.mean(0)
        trust = mu**2/(mu**2+2*scores.var(0)/len(dates)+1e-8)
        co.append(v[:12]@(mu[:,:3]*trust[:,:3]*(mu[:,:3]*mu[:,3:]>0)))
    return np.concatenate((base.risk.walk.widths(),src.axes().ravel(),flat,np.stack(co).ravel()))


def signals(raw,grid,p,bank):
    out = np.zeros((*p.shape[:2],3,3))
    i,k = 50+100*src.PC,3*12*3
    enc = restore(bank[i:-k])
    weights = bank[-k:].reshape(3,12,3)
    for a,b,m,ok in event.sessions(raw,grid):
        tape,_ = src.stream(raw[a:b],m,ok,bank[50:i].reshape(100,src.PC),bank[:50])
        for ix,parts in src.views(tape,np.arange(b-a),enc):
            for j,z in enumerate(features(p[a+ix],parts,ok[ix],bank[:50])):
                out[a+ix,:,j] = z@weights[j]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
