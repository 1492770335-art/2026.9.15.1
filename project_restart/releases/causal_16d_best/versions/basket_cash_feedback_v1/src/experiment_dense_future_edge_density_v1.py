"""Forecast the distribution of future five-tick book transitions before mapping those transitions to cash."""
from functools import lru_cache
import json
import sys

import numpy as np
import torch
from torch import nn

import experiment_dense_after_book_transition_v1 as edge

base,strip,event,onset,rel = edge.base,edge.strip,edge.event,edge.onset,edge.rel
REFERENCE,read_p = edge.REFERENCE,edge.read_p
OUT = REFERENCE.parent/'dense_future_edge_density_v1'
ARMS = ('forecast_endpoints','forecast_transition_interaction','forecast_following_cash')
BANDS = ((0,), (5,10,15), tuple(range(20,60,5)))
C,D = 8,17
W = 32
N = 28


def fields(raw,m):
    depth = base.depth(raw)
    q = np.tanh(depth@base.axis())
    ids = np.digitize(q,[-.65,-.22,.22,.65])
    t = np.arange(len(m))
    n = (t+1)[:,None,None]
    d = np.tanh(depth)
    fast = edge.cycle.ref.window(d,5)/np.minimum(n,5)
    slow = edge.cycle.ref.window(d,40)/np.minimum(n,40)
    v = np.sqrt(edge.cycle.ref.window(m*m,80)/np.minimum(n[:,:,0],80)+.0004)
    r = np.tanh(edge.cycle.ref.window(m,5)/(v*np.sqrt(5)))
    liquidity = np.tanh(np.arcsinh(raw[...,base.BID].astype(float))+np.arcsinh(raw[...,base.ASK].astype(float)))
    f = np.concatenate((d,d-fast,fast-slow,liquidity[:,:,:3],
                        np.stack((q,r,np.log(v),q.mean(1,keepdims=True)+np.zeros_like(q)),axis=2)),axis=2)
    return f.astype(np.float32),ids,v


def targets(code,t):
    out = []
    for shifts in BANDS:
        s = t[:,None]+np.asarray(shifts)
        out.append((5*code[s]+code[s+5]).transpose(0,2,1))
    return out


@lru_cache(maxsize=1)
def encoder():
    data = []
    counts = np.ones((5,3,25))
    for d in range(4):
        raw,grid = [onset.settled.load(d,k) for k in ('x','time')]
        for a,b,m,ok in event.sessions(raw,grid):
            t = np.arange(40+d%4,b-a-60,4)
            x,ids,_ = fields(raw[a:b],m)
            gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
            valid = gap[t+61]==gap[t]
            idx = ids[t][valid]
            y = []
            for j,ar in enumerate(targets(ids,t)):
                z = ar[valid]
                for k in range(z.shape[1]):
                    np.add.at(counts[:,j],(idx,z[:,k]),1/z.shape[1])
                y.append(z)
            data.append((x[t][valid],np.concatenate(y,axis=1)))
    x,target = [np.concatenate(v) for v in zip(*data)]
    mu,sd = x.mean(0),x.std(0).clip(.02)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.as_tensor(((x-mu)/sd).clip(-6,6),device=device)
    y = torch.as_tensor(target,device=device)
    torch.manual_seed(48713)
    net = nn.Sequential(nn.Linear(N,W),nn.SiLU(),nn.Linear(W,W),nn.SiLU(),nn.Linear(W,75)).to(device)
    opt = torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.025)
    for step in range(650):
        ix = torch.randint(len(x),(1536,),device=device)
        logp = net(x[ix]).reshape(-1,3,25).log_softmax(2)
        loss = -sum(logp[:,j].gather(1,z).mean() for j,z in enumerate(y[ix].split((1,3,8),dim=1)))/3
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return np.concatenate((mu,sd,(counts/counts.sum(2,keepdims=True)).ravel(),
                 *(p.detach().cpu().numpy().ravel() for p in net.parameters())))


def density(x,code,bank):
    mu,sd = bank[:N],bank[N:2*N]
    n = 2*N+375
    prior = bank[2*N:n].reshape(5,3,25)
    h = ((x-mu)/sd).clip(-6,6)
    for i,(ins,outs) in enumerate(((N,W),(W,W),(W,75))):
        w = bank[n:n+ins*outs].reshape(outs,ins)
        n += ins*outs
        h = h@w.T+bank[n:n+outs]
        n += outs
        if i<2:
            h = h/(1+np.exp(-h.clip(-50,50)))
    logits = h.reshape(*x.shape[:-1],3,25)
    val = np.exp(logits-logits.max(-1,keepdims=True))
    return val/val.sum(-1,keepdims=True),prior[code]


@lru_cache(maxsize=16)
def reward_moments(d):
    raw,grid = [onset.settled.load(d,k) for k in ('x','time')]
    totals = np.zeros((3,25))
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(40+d%3,b-a-10,3)
        _,ids,vol = fields(raw[a:b],m)
        gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid = gap[t+11]==gap[t]
        code = (5*ids[t]+ids[t+5])[valid]
        path = m.cumsum(0)
        totals[0] += np.bincount(code,minlength=25)
        for j in range(2):
            y = path[t+5*(j+1)]-path[t+5*j]
            y -= (y*valid).sum(1,keepdims=True)/valid.sum(1,keepdims=True).clip(1)
            totals[j+1] += np.bincount(code,weights=(y/vol[t])[valid],minlength=25)
    return totals


def rewards(end):
    totals = np.sum([reward_moments(d) for d in range(end)],axis=0)
    c = totals[0]+totals[0,::-1]
    r = (totals[1:]-totals[1:,::-1])/(c+1.)
    a,b = np.indices((5,5)).reshape(2,25)
    design = np.column_stack((np.eye(5)[a],np.eye(5)[b]))
    avg = design@base.solve(design.T@(c[:,None]*design),design.T@(c*r[0]),.004)
    return np.stack((avg,r[0]-avg,r[1]))


def features(raw,m,p,bank,table):
    x,code,vol = fields(raw,m)
    out = np.zeros((*m.shape,D))
    out[:,:,:3] = p
    out[:,:,3] = x[...,-4]
    out[:,:,4] = x[...,-3]
    for start in range(0,len(m),64):
        sl = slice(start,min(start+64,len(m)))
        prob,prior = density(x[sl],code[sl],bank)
        size = vol[sl,...,None]*np.array([1,3,8])
        out[sl,:,5:8] = (prior@(table[0]+table[1]))*size
        delta = prob-prior
        for j in range(3):
            out[sl,:,C+3*j:C+3*j+3] = (delta@table[j])*size
    return strip.center(out)


@lru_cache(maxsize=12)
def record(d):
    raw,grid,y = [onset.settled.load(d,k) for k in ('x','time','y')]
    p = read_p(d)
    table = rewards(d)
    g,n = 0.,0
    loss = np.zeros((2,3))
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(40+d%5,b-a-60,5)
        gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid = gap[t+61]==gap[t]
        f = features(raw[a:b],m,p[a:b],encoder(),table)[t]
        x,code,_ = fields(raw[a:b],m)
        prob,prior = density(x[t][valid],code[t][valid],encoder())
        for j,z in enumerate(targets(code,t)):
            for k,dist in enumerate((prob,prior)):
                loss[k,j] -= np.log(np.take_along_axis(dist[:,j],z[valid],1).clip(1e-12)).mean(1).sum()
        path = m.cumsum(0)
        cash = np.stack([path[t+h]-path[t+l] for l,h in ((0,5),(5,20),(20,60))],axis=2)
        z = np.concatenate((f,cash,y[a+t,:,1:]),axis=2)
        z -= ((z*valid[...,None]).sum(1)/valid.sum(1)[:,None].clip(1))[:,None]
        rows = z[valid]
        g += rows.T@rows
        n += len(rows)
    return g/n,loss/n


def train(end):
    return np.r_[encoder(),rewards(end).ravel(),
                 edge.cycle.ref.fit(np.stack([record(d)[0] for d in range(4,end)]),C,D).ravel()]


def signals(raw,grid,p,bank):
    out = np.zeros((*p.shape[:2],3,2))
    table = bank[-75-6*D:-6*D].reshape(3,25)
    for a,b,m,ok in event.sessions(raw,grid):
        f = features(raw[a:b],m,p[a:b],bank[:-75-6*D],table)
        out[a:b] = (f@bank[-6*D:].reshape(D,6)).reshape(b-a,50,3,2)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
    out = {}
    for end in (8,12):
        days = [record(d) for d in range(end,end+4)]
        tr = np.mean([record(d)[0] for d in range(4,end)],0)
        test,loss = [np.mean(v,0) for v in zip(*days)]
        pr = np.eye(len(tr))-np.eye(len(tr))[:,:C]@base.solve(tr[:C,:C],tr[:C],.015)
        g = pr.T@test@pr
        r = g/np.sqrt(np.outer(g.diagonal(),g.diagonal()).clip(1e-16))
        out[end+1] = dict(log_loss=loss[0].tolist(),state_only_loss=loss[1].tolist(),
                         conditional_cash=[r[C+h,D+h] for h in range(3)])
    (OUT/'forecastability.json').write_text(json.dumps(out,indent=2))
