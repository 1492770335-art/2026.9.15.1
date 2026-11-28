"""Marginalize over a near-term path state to estimate the remaining long-horizon forecast error."""
from functools import lru_cache
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dense_pressure_survival_bridge_v1 as ref

onset,rel,strip,event,base = ref.onset,ref.rel,ref.strip,ref.event,ref.base
REFERENCE,read_p = ref.REFERENCE,ref.read_p
OUT = REFERENCE.parent/'dense_excursion_state_mixture_v1'
ARMS = ('single_expectation','conditional_payoff','state_probability_shift')
K = 4


def inputs(raw,m,ok,p):
    q = strip.center(p.astype(float))
    side = np.where(q[:,:,1]>=0,1.,-1.)
    v = np.sqrt(ref.ref.smooth(m**2,100)+.0004)
    t = np.arange(len(m))
    path = m.cumsum(0)
    ret = [(path-path[(t-h).clip(0)])/(np.sqrt(h)*v) for h in (5,20,60)]
    depth = np.tanh(base.depth(raw))
    fast,slow = ref.ref.smooth(depth,5),ref.ref.smooth(depth,40)
    # Align signed fields with the current twenty-tick forecast before estimating state probabilities.
    a = np.concatenate((q,p-q,depth,fast-slow,np.stack(ret,axis=2)),axis=2)*side[...,None]
    r = np.stack((np.log(v),ref.ref.smooth((abs(m)>.001).astype(float),20),
                  (abs(q[:,:,1])-abs(q[:,:,0])), np.broadcast_to(t[:,None]/2400,m.shape)),axis=2)
    return np.concatenate((np.arcsinh(a),r),axis=2).astype(np.float32),side,v


def near_state(m,t,side,v):
    path = m.cumsum(axis=0)
    marked = np.stack([(path[t+h]-path[t])*side for h in range(1,21)],axis=2)
    boundary = np.sqrt(20)*v*.65
    hit = marked.max(2)>=boundary
    # Categories depend only on the first twenty future observations.
    state = np.where(hit, np.where(marked[:,:,-1]>=.5*marked.max(2),0,1),
                     np.where(marked.min(2)<=-boundary,2,3))
    return state,marked[:,:,-1]


@lru_cache(maxsize=12)
def sample(d):
    raw,grid,y = [onset.settled.load(d,k) for k in ('x','time','y')]
    p = read_p(d)
    rows = []
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(30+d%6,b-a-60,6)
        f,s,v = inputs(raw[a:b],m,ok,p[a:b])
        z,near = near_state(m,t,s[t],v[t])
        gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        good = gap[t+61]==gap[t]
        path = m.cumsum(0)
        far = (path[t+60]-path[t+20])*s[t]
        def center(arr):
            return arr-(arr*good).sum(1,keepdims=True)/good.sum(1,keepdims=True).clip(1)
        q = center(p[a+t,:,2])
        label = center(y[a+t,:,2])
        rows.append((f[t][good], z[good], (near/(v[t]*np.sqrt(20)))[good], (far/(v[t]*np.sqrt(40)))[good],
                     q[good],label[good],s[t][good]))
    f,z,near,far,q,y,side = [np.concatenate(v) for v in zip(*rows)]
    beta = np.clip(q@y/(q@q),.2,4.)
    err = side*(y/beta-q)
    return f,z,np.stack((near,far,err),axis=1).astype(np.float32)


class Mixture(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(n,36),nn.SiLU(),nn.Linear(36,24),nn.SiLU())
        self.logits = nn.Linear(24,K)
        self.experts = nn.Linear(24,K*3)
        self.direct = nn.Linear(24,3)

    def forward(self,x):
        h = self.hidden(x)
        return self.logits(h),self.experts(h).reshape(-1,K,3),self.direct(h)


def fit(days,seed):
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    f,z,r = [np.concatenate(v) for v in zip(*(sample(i) for i in days))]
    mean,sd = f.mean(0),f.std(0).clip(.02)
    x = torch.tensor(((f-mean)/sd).clip(-5,5),device=device)
    state = torch.tensor(z,device=device)
    targets = torch.tensor(r.clip(-8,8),device=device)
    # The auxiliary losses distinguish realizing an early gain from giving it back later.
    prior = np.bincount(z,minlength=K)/len(z)
    net = Mixture(x.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(),lr=.0018,weight_decay=.035)
    for epoch in range(5):
        for idx in torch.randperm(len(x),device=device).split(2048):
            logits,pay,direct = net(x[idx])
            selected = pay[torch.arange(len(idx),device=device),state[idx]]
            w = torch.tensor([.30,.30,1.],device=device)
            loss = F.cross_entropy(logits,state[idx]) + ((direct-targets[idx]).square()*w).mean()
            loss += ((selected-targets[idx]).square()*w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return np.concatenate((mean,sd,prior,*(p.detach().cpu().numpy().ravel() for p in net.parameters())))


def unpack(bank):
    n = 27
    mean,sd,prior = bank[:n],bank[n:2*n],bank[2*n:2*n+K]
    offset = 2*n+K
    weights = []
    for shape in ((36,n),(36,),(24,36),(24,),(K,24),(K,),(K*3,24),(K*3,),(3,24),(3,)):
        end = offset+int(np.prod(shape))
        weights.append(bank[offset:end].reshape(shape))
        offset = end
    assert offset==len(bank)
    return mean,sd,prior,weights


def estimates(f,bank):
    mean,sd,prior,(w1,b1,w2,b2,w3,b3,w4,b4,w5,b5) = unpack(bank)
    x = ((f-mean)/sd).clip(-5,5)
    def silu(a):
        return a/(1+np.exp(-a.clip(-50,50)))
    h = silu(silu(x@w1.T+b1)@w2.T+b2)
    logits = h@w3.T+b3
    prob = np.exp(logits-logits.max(-1,keepdims=True))
    prob /= prob.sum(-1,keepdims=True)
    pay = (h@w4.T+b4).reshape(*f.shape[:-1],K,3)[...,-1]
    direct = (h@w5.T+b5)[...,-1]
    return np.stack((direct,(pay*prob).sum(-1),(pay*(prob-prior)).sum(-1)),axis=-1)


def train(end):
    return np.stack([fit(range(4+i,end,2),27361+i) for i in range(2)])


def signals(raw,grid,p,bank):
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        f,s,_ = inputs(raw[a:b],m,ok,p[a:b])
        v = sum(estimates(f,model) for model in bank)*.5
        out[a:b,:,:,1] = strip.center(v*s[...,None])
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__ == '__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
