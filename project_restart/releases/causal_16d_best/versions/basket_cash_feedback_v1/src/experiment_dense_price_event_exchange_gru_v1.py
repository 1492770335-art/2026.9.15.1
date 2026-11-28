"""Compare equal-time, price-event and reversal-event histories with a shared causal recurrent encoder."""
from functools import lru_cache
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dense_exchange_count_mixture_v1 as count

pre,peer,direct,strip,event,onset,rel = count.pre,count.peer,count.direct,count.strip,count.event,count.onset,count.rel
REFERENCE,read_p = count.REFERENCE,count.read_p
OUT = REFERENCE.parent/'dense_price_event_exchange_gru_v1'
ARMS = ('calendar_history','price_event_history','reversal_history')
C,D,L,DEVICE = 6,15,count.L,pre.DEVICE
Q = L+2


def clocks(m,ok,t):
    times = np.arange(len(m))[:,None]
    hit = (abs(m)>.001)&ok
    last = np.maximum.accumulate(np.where(hit,times,0),axis=0)
    sign = np.sign(m[last,np.arange(50)])
    prev = np.vstack((np.zeros(50),sign[:-1]))
    switch = hit&(np.sign(m)!=prev)
    reset = np.maximum.accumulate(np.where(~ok,times+1,0),axis=0)[t].clip(max=len(m)-1)
    wall = np.broadcast_to(t[:,None,None]-np.array([60,30,10,3,0]),(len(t),50,5))
    indices = [wall]
    for event_mask in (hit,switch):
        group = np.empty((len(t),50,5),dtype=int)
        group[:,:,-1] = t[:,None]
        for s in range(50):
            tick = np.flatnonzero(event_mask[:,s])
            order = np.searchsorted(tick,t)[:,None]-np.array([6,4,2,1])
            previous = np.r_[0,tick]
            group[:,s,:4] = previous[(order+1).clip(0)]
        indices.append(group)
    return [np.minimum(t[:,None,None],np.maximum(z,reset[:,:,None])) for z in indices]


def sequence(tokens,m,t,idx):
    stock = np.arange(50)[None,:,None]
    age = np.log1p(t[:,None,None]-idx)/np.log(61)
    movement = np.tanh(m[idx,stock]/.12)
    return np.concatenate((tokens[idx,stock],age[...,None],movement[...,None]),axis=-1).astype(np.float16)


class Readout(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Sequential(nn.Linear(L,32),nn.LayerNorm(32),nn.SiLU(),nn.Linear(32,3))
        self.rnn = nn.GRU(Q,24,batch_first=True)
        self.price = nn.Linear(24,3,bias=False)

    def correction(self,x):
        z = x.reshape(-1,5,Q)
        _,h = self.rnn(z)
        _,current = self.rnn(z[:,-1:])
        return self.price(h[0]-current[0]).reshape(*x.shape[:2],3)


@lru_cache(maxsize=1)
def encoder():
    base = pre.encoder()
    frozen = pre.restore(base)
    bank,ys,valid = [],[],[]
    for day in range(4):
        raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
        for a,b,m,ok in event.sessions(raw,grid):
            t = np.arange(60+day%5,len(m)-60,5)
            tok = count.tokens(pre.inputs(raw[a:b]),frozen,base)
            bank.append(np.stack([sequence(tok,m,t,idx) for idx in clocks(m,ok,t)],axis=1))
            gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
            live = gap[t+61]==gap[t]
            path = m.cumsum(0)
            y = np.stack([path[t+hi]-path[t+lo] for lo,hi in count.BANDS],axis=2)
            y -= ((y*live[...,None]).sum(1)/live.sum(1)[:,None].clip(1))[:,None]
            ys.append(y.astype(np.float32))
            valid.append(live)
    panels = np.concatenate(bank)
    x = panels[:,0,:,-1,:L].astype(np.float32)
    mu,sd = x.mean((0,1)),x.std((0,1)).clip(.008)
    cash,ok = np.concatenate(ys),np.concatenate(valid)
    scale = cash[ok].std(0).clip(.02)
    obs = torch.as_tensor(panels,device=DEVICE)
    loc = torch.tensor(((x-mu)/sd).clip(-6,6),device=DEVICE)
    target = torch.tensor((cash/scale).clip(-5,5),device=DEVICE)
    keep = torch.tensor(ok[...,None],device=DEVICE)
    mean,den = [torch.tensor(v,device=DEVICE) for v in (mu,sd)]
    torch.manual_seed(35127)
    net = Readout().to(DEVICE)
    for stage,params,steps in ((0,list(net.local.parameters()),300),(1,[*net.rnn.parameters(),*net.price.parameters()],500)):
        opt = torch.optim.AdamW(params,lr=.0015,weight_decay=.04)
        for i in range(steps):
            ix = torch.randint(len(loc),(32,),device=DEVICE)
            baseline = net.local(loc[ix])
            if stage:
                z = obs[ix,i%3].float()
                z[...,:L] = ((z[...,:L]-mean)/den).clamp(-6,6)
                pred = baseline.detach()+net.correction(z)
            else:
                pred = baseline
            pred = pred-pred.mean(1,keepdim=True)
            loss = (F.smooth_l1_loss(pred,target[ix],beta=1.5,reduction='none')*keep[ix]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params,2.)
            opt.step()
    weights = torch.nn.utils.parameters_to_vector(net.parameters()).detach().cpu().numpy()
    return np.concatenate((mu,sd,scale,weights))


def restore(params):
    model = Readout().to(DEVICE).eval()
    torch.nn.utils.vector_to_parameters(torch.tensor(params[2*L+3:],dtype=torch.float32,device=DEVICE),model.parameters())
    model.rnn.flatten_parameters()
    return model


@torch.inference_mode()
def features(raw,m,ok,p,t,base,frozen,bank,model):
    tok = count.tokens(pre.inputs(raw),frozen,base)
    choices = clocks(m,ok,t)
    out = np.empty((len(t),50,D))
    mean,sd = [torch.tensor(v,device=DEVICE,dtype=torch.float32) for v in (bank[:L],bank[L:2*L])]
    units = bank[2*L:2*L+3]
    out[:,:,:3] = p[t]
    for a in range(0,len(t),64):
        sl = slice(a,a+64)
        batch = t[sl]
        n = len(batch)
        for j,ix in enumerate(choices):
            z = torch.tensor(sequence(tok,m,batch,ix[sl]),device=DEVICE,dtype=torch.float32)
            z[...,:L] = ((z[...,:L]-mean)/sd).clamp(-6,6)
            z = F.pad(z,(0,0,0,0,0,0,0,64-n)) if n<64 else z
            if j==0:
                out[sl,:,3:6] = model.local(z[:,:,-1,:L])[:n].cpu().numpy()*units
            out[sl,:,C+3*j:C+3*j+3] = model.correction(z)[:n].cpu().numpy()*units
    return strip.center(out)


@lru_cache(maxsize=12)
def record(day):
    raw,grid,y = [onset.settled.load(day,k) for k in ('x','time','y')]
    p = read_p(day)
    base,bank = pre.encoder(),encoder()
    frozen,model = pre.restore(base),restore(bank)
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
    return np.r_[len(base),base,encoder(),co.ravel()]


def signals(raw,grid,p,params):
    cut = 1+int(params[0])
    base,bank = params[1:cut],params[cut:-D*6]
    frozen,model = pre.restore(base),restore(bank)
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
