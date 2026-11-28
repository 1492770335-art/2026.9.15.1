"""Pretrain stock tokens on return legs with antisymmetric, market-neutral exchanges between all 50 names."""
from functools import lru_cache
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import prepare_dense_market_edge_transmission_reference_v1 as best

peer,direct,strip,event,onset,rel = best.model.peer,best.model.direct,best.model.strip,best.model.event,best.model.onset,best.model.rel
REFERENCE,read_p = best.OUT,best.read_p
OUT = REFERENCE.parent/'dense_antisymmetric_asset_exchange_v1'
ARMS = ('stock_token','uniform_exchange','affinity_exchange')
COL = [0,11,26,34,36,38,39,41,42,50,51,54,55,64,74,79,80]
N,H,K = 100+2*len(COL),24,12
C,D = 3,3+24+12+12
GROUPS = (np.arange(C,C+H),np.arange(C+H,C+H+K),np.arange(C+H+K,D))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def inputs(raw):
    z = np.tanh(.65*np.arcsinh(raw.astype(np.float32)))
    h = z[:,:,COL]
    def avg(k):
        return direct.edge.cycle.ref.window(h,k)/np.minimum(np.arange(1,len(h)+1),k)[:,None,None]
    fast,slow = avg(6),avg(48)
    return np.concatenate((z,h-fast,fast-slow),axis=2).astype(np.float32)


class Exchange(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(N,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,H),nn.Tanh())
        self.pair = nn.Linear(H,2*K)
        self.query = nn.Linear(H,8)
        self.local_price = nn.Linear(H,3,bias=False)
        self.uniform_price = nn.Linear(K,3,bias=False)
        self.adaptive_price = nn.Linear(K,3,bias=False)

    def token_views(self,x):
        z = self.encode(x)
        a,b = self.pair(z).tanh().chunk(2,dim=-1)
        # For a symmetric adjacency, each ordered contribution has an equal
        # and opposite contribution at the receiving node.
        uniform = a*b.mean(1,keepdim=True)-b*a.mean(1,keepdim=True)
        q = F.normalize(self.query(z),dim=-1)
        strength = (4*(q@q.transpose(1,2))-4).exp()
        eye = torch.eye(z.shape[1],device=z.device,dtype=z.dtype)
        strength = strength*(1-eye)
        degree = strength.sum(2).clamp_min(1.e-5).rsqrt()
        w = strength*degree[:,:,None]*degree[:,None,:]
        dynamic = a*(w@b)-b*(w@a)-uniform
        return z-z.mean(1,keepdim=True),uniform,dynamic

    def forward(self,x):
        a,b,c = self.token_views(x)
        return self.local_price(a),self.uniform_price(b),self.adaptive_price(c)


@lru_cache(maxsize=1)
def encoder():
    xs,ys,valid = [],[],[]
    for day in range(4):
        raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
        for a,b,m,ok in event.sessions(raw,grid):
            t = np.arange(50+day%5,len(m)-60,5)
            gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
            live = gaps[t+61]==gaps[t]
            price = m.cumsum(0)
            r = np.stack([price[t+hi]-price[t+lo] for lo,hi in ((0,5),(5,20),(20,60))],axis=2)
            r -= ((r*live[...,None]).sum(1)/live.sum(1)[:,None].clip(1))[:,None]
            xs.append(inputs(raw[a:b])[t])
            ys.append(r.astype(np.float32))
            valid.append(live)
    x,y,good = map(np.concatenate,(xs,ys,valid))
    mu,sd = x.mean((0,1)),x.std((0,1)).clip(.04)
    x = torch.as_tensor(((x-mu)/sd).clip(-6,6),device=DEVICE)
    target = torch.as_tensor((y/y[good].std(0).clip(.02)).clip(-5,5),device=DEVICE)
    ok = torch.as_tensor(good[...,None],device=DEVICE)
    torch.manual_seed(54089)
    model = Exchange().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.05)
    for _ in range(680):
        ix = torch.randint(len(x),(40,),device=DEVICE)
        local,fixed,change = model(x[ix])
        losses = [F.smooth_l1_loss(p,target[ix],reduction='none',beta=1.5)
                  for p in (local,local.detach()+fixed,(local+fixed).detach()+change)]
        loss = (sum(losses)*ok[ix]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        opt.step()
    weights = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy()
    return np.r_[mu,sd,weights]


def restore(params):
    state = Exchange().to(DEVICE).eval()
    v = torch.tensor(params[2*N:],dtype=torch.float32,device=DEVICE)
    torch.nn.utils.vector_to_parameters(v,state.parameters())
    return state


@torch.inference_mode()
def views(raw,p,params,model):
    x = ((inputs(raw)-params[:N])/params[N:2*N]).clip(-6,6)
    out = []
    for start in range(0,len(x),64):
        z = torch.as_tensor(x[start:start+64],device=DEVICE,dtype=torch.float32)
        size = len(z)
        if size<64:
            z = F.pad(z,(0,0,0,0,0,64-size))
        out.append(torch.cat(model.token_views(z),dim=-1)[:size].cpu().numpy())
    return np.concatenate((strip.center(p.astype(float)),np.concatenate(out)),axis=-1)


@lru_cache(maxsize=12)
def record(day):
    raw,grid,y = [onset.settled.load(day,k) for k in ('x','time','y')]
    p = read_p(day)
    g,n = 0.,0
    params = encoder()
    model = restore(params)
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(50+day%5,len(m)-60,5)
        gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        live = gaps[t+61]==gaps[t]
        x = views(raw[a:b],p[a:b],params,model)[t]
        z = np.concatenate((x,y[a+t,:,1:]),axis=2)
        z -= ((z*live[...,None]).sum(1)/live.sum(1)[:,None].clip(1))[:,None]
        flat = z[live]
        g += flat.T@flat
        n += len(flat)
    return g/n


def train(end):
    gs = np.stack([record(d) for d in range(4,end)])
    g = gs.mean(0)
    slopes = (gs[:,[1,2],[-2,-1]]/gs[:,[1,2],[1,2]]).clip(.2,4)
    residual = gs[:,:D,-2:]/slopes[:,None]-gs[:,:D,1:3]
    co = []
    for ix in GROUPS:
        v = np.eye(D)[:,ix]
        v[:C] = -peer.base.solve(g[:C,:C],g[:C,ix],.025)
        rhs = v.T@residual
        mean = rhs.mean(0)
        shrink = mean**2/(mean**2+3*rhs.var(0)/len(rhs)+1e-14)
        co.append(v@peer.base.solve(v.T@g[:D,:D]@v,mean*shrink,.85))
    return np.r_[encoder(),np.stack(co,axis=1).ravel()]


def signals(raw,grid,p,params):
    netbank = params[:-D*6]
    model = restore(netbank)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        x = views(raw[a:b],p[a:b],netbank,model)
        out[a:b] = (x@params[-D*6:].reshape(D,6)).reshape(b-a,50,3,2)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
