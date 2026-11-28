"""Vary the placement of GRU history observations while keeping the beginning and end of the window fixed."""
from functools import lru_cache
import sys

import numpy as np
import torch
from torch.nn import functional as F

import experiment_dense_price_event_exchange_gru_v1 as recurrent

pre,peer,direct,strip,event,onset,rel = recurrent.pre,recurrent.peer,recurrent.direct,recurrent.strip,recurrent.event,recurrent.onset,recurrent.rel
REFERENCE,read_p = recurrent.REFERENCE,recurrent.read_p
OUT = REFERENCE.parent/'dense_equal_span_event_gru_v1'
ARMS = ('all_price_changes','retained_price_changes','reversed_price_changes')
C,D = 9,18


def grids(m,ok,t):
    ticks = np.arange(len(m))[:,None]
    lower = np.maximum((t-60)[:,None],np.maximum.accumulate(np.where(~ok,ticks+1,0),axis=0)[t]).clip(0)
    lower = np.minimum(lower,t[:,None])
    fractions = np.array([0,.5,5/6,.95,1.])
    calendar = np.rint(lower[...,None]+(t[:,None]-lower)[...,None]*fractions).astype(int)
    arrived = (abs(m)>.001)&ok
    # Credit a persistent or reversed move only at its five-tick confirmation.
    path = m.cumsum(0)
    lag = (np.arange(len(m))-5).clip(0)
    before = (lag-1).clip(0)
    alive = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    confirmed = arrived[lag]&(ticks>=6)&(alive[1:]==alive[before])
    held = (path-path[before])*np.sign(m[lag])>=.5*abs(m[lag])
    result = [calendar]
    for mass in (arrived,confirmed&held,confirmed&~held):
        count = direct.edge.cycle.ref.window(mass.astype(float),160)
        rate = (count+1)/(np.minimum(ticks+1,160)+4)
        clock = np.cumsum(1+mass*(3/rate.clip(.04)).clip(0,45),axis=0)
        out = np.empty_like(calendar)
        for s in range(50):
            start = clock[lower[:,s],s]
            target = start[:,None]+(clock[t,s]-start)[:,None]*fractions
            out[:,s] = np.searchsorted(clock[:,s],target).clip(max=len(m)-1)
        out[:,:,0],out[:,:,-1] = lower,t[:,None]
        result.append(np.minimum(t[:,None,None],np.maximum(lower[...,None],out)))
    return result


@torch.inference_mode()
def features(raw,m,ok,p,t,base,frozen,bank,model):
    tok = recurrent.count.tokens(pre.inputs(raw),frozen,base)
    choices = grids(m,ok,t)
    out = np.empty((len(t),50,D))
    out[:,:,:3] = p[t]
    mean,sd = [torch.tensor(v,device=pre.DEVICE,dtype=torch.float32)
               for v in (bank[:recurrent.L],bank[recurrent.L:2*recurrent.L])]
    unit = bank[2*recurrent.L:2*recurrent.L+3]
    for a in range(0,len(t),64):
        sl = slice(a,a+64)
        n = len(t[sl])
        for j,indices in enumerate(choices):
            z = torch.tensor(recurrent.sequence(tok,m,t[sl],indices[sl]),device=pre.DEVICE,dtype=torch.float32)
            z[...,:recurrent.L] = ((z[...,:recurrent.L]-mean)/sd).clamp(-6,6)
            if n<64:
                z = F.pad(z,(0,0,0,0,0,0,0,64-n))
            value = model.correction(z)[:n].cpu().numpy()*unit
            if j==0:
                out[sl,:,3:6] = model.local(z[:,:,-1,:recurrent.L])[:n].cpu().numpy()*unit
                out[sl,:,6:9] = value
            else:
                out[sl,:,C+3*(j-1):C+3*j] = value-out[sl,:,6:9]
    return strip.center(out)


@lru_cache(maxsize=12)
def record(day):
    raw,grid,y = [onset.settled.load(day,k) for k in ('x','time','y')]
    p = read_p(day)
    base,bank = pre.encoder(),recurrent.encoder()
    frozen,model = pre.restore(base),recurrent.restore(bank)
    g,n = 0.,0
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(60+day%5,len(m)-60,5)
        gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        keep = gaps[t+61]==gaps[t]
        f = features(raw[a:b],m,ok,p[a:b],t,base,frozen,bank,model)
        z = np.concatenate((f,y[a+t,:,1:]),axis=2)
        z -= ((z*keep[...,None]).sum(1)/keep.sum(1)[:,None].clip(1))[:,None]
        v = z[keep]
        g += v.T@v
        n += len(v)
    return g/n


def train(end):
    co = direct.edge.cycle.ref.fit(np.stack([record(d) for d in range(4,end)]),C,D)
    base = pre.encoder()
    return np.r_[len(base),base,recurrent.encoder(),co.ravel()]


def signals(raw,grid,p,params):
    cut = 1+int(params[0])
    base,bank = params[1:cut],params[cut:-D*6]
    frozen,model = pre.restore(base),recurrent.restore(bank)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        f = features(raw[a:b],m,ok,p[a:b],np.arange(len(m)),base,frozen,bank,model)
        out[a:b] = (f@params[-D*6:].reshape(D,6)).reshape(b-a,50,3,2)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
