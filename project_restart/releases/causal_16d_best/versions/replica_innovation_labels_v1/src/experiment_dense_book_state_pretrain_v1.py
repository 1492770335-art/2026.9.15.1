"""Pretrain a causal recurrent cross-stock encoder and calibrate its innovations on later training dates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from functools import lru_cache
import sys

import numpy as np
import torch
from torch import nn

from experiment_dense_market_conditional_graph_risk_v1 import event,onset,rel,strip
import experiment_dense_peer_crossing_payoff_v1 as parent

base,ref,REFERENCE,read_p = parent.base,parent.ref,parent.REFERENCE,parent.read_p
OUT = REFERENCE.parent/'dense_book_state_pretrain_v1'
ARMS = ('local_recurrent_innovation','cross_stock_innovation','joint_innovation')
TARGETS = (2,)
AGES = np.array([40,20,10,4,1,0])
PC,H,D,T = 12,20,19,12
C = 3+T


@lru_cache(maxsize=1)
def axes():
    gram = np.zeros((100,100))
    for d in range(4):
        raw,grid = [onset.settled.load(d,k) for k in ('x','time')]
        for a,b,m,ok in event.sessions(raw,grid):
            t = np.arange(d,b-a,27)
            x = np.tanh(.65*np.arcsinh(raw[a+t].astype(float)))[ok[t]]
            gram += x.T@x/len(x)
    sd = np.sqrt(gram.diagonal()/8).clip(.04)
    val,v = np.linalg.eigh(gram/8/np.outer(sd,sd))
    return v[:,-PC:]/sd[:,None]/np.sqrt(val[-PC:]+.6)


def stream(raw,m,ok,basis,unit):
    v = np.tanh(.65*np.arcsinh(raw.astype(np.float32)))
    paired = v[:,:,base.risk.walk.ref.model.FIELDS]
    book = paired[...,::2]-paired[...,1::2]
    x = np.concatenate((np.tanh(v@basis.astype(np.float32)),book,
                        np.tanh(m/unit)[...,None],ok[...,None]),axis=2)*ok[...,None]
    return x.astype(np.float32),book


def tokens(stream,t):
    idx = t[:,None]-AGES
    seq = stream[idx.clip(0)]*(idx>=0)[:,:,None,None]
    return np.ascontiguousarray(seq.transpose(0,2,1,3))


class Encoder(nn.Module):
    def __init__(self,input_dim=D):
        super().__init__()
        self.input_dim = input_dim
        self.level = nn.Linear(input_dim,H)
        self.memory = nn.GRU(input_dim,H,batch_first=True)
        self.query = nn.Linear(H,8,bias=False)
        self.key = nn.Linear(H,8,bias=False)
        self.identity = nn.Parameter(torch.zeros(50,8))
        self.peer = nn.Linear(2*H,H)
        self.heads = nn.ModuleList([nn.Linear(H,T) for _ in range(3)])

    def states(self,x):
        n = len(x)
        now = self.level(x[:,:,-1]).tanh()
        differences = torch.cat((x[:,:,:1],x[:,:,1:]-x[:,:,:-1]),dim=2)
        _,hidden = self.memory(differences.reshape(-1,len(AGES),self.input_dim))
        hist = hidden.reshape(n,50,H)
        q = self.query(hist)+self.identity
        k = self.key(now)+self.identity
        score = q@k.transpose(1,2)/np.sqrt(8.)
        allowed = (x[:,:,-1,-1]>0)[:,None,:] & ~torch.eye(50,dtype=torch.bool,device=x.device)[None]
        attention = score.masked_fill(~allowed,-1000.).softmax(2)*allowed
        message = attention@hist
        peer = self.peer(torch.cat((message,message*now),dim=2)).tanh()
        return now,hist,peer

    def forward(self,x):
        return torch.stack([head(z) for head,z in zip(self.heads,self.states(x))],dim=2)


@lru_cache(maxsize=12)
def examples(day):
    raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
    rows = []
    unit = base.risk.walk.widths()
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(40+day%13,b-a-60,13)
        x,book = stream(raw[a:b],m,ok,axes(),unit)
        path = m.cumsum(0)
        gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid = gaps[t+61]==gaps[t-40]
        y = np.concatenate(((book[t+20]-book[t])*2,(book[t+40]-book[t])*2,
                            np.stack([(path[t+h+20]-path[t+h])/unit for h in (20,40)],axis=2)/3),axis=2)
        rows.append((tokens(x,t),y.clip(-5,5).astype(np.float32),valid))
    return tuple(torch.from_numpy(np.concatenate(v)) for v in zip(*rows))


@lru_cache(maxsize=4)
def pretrained(end):
    torch.manual_seed(198731)
    torch.use_deterministic_algorithms(True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Encoder().to(dev)
    opt = torch.optim.AdamW(model.parameters(),lr=.0025,weight_decay=.015)
    x,y,valid = [torch.cat(v) for v in zip(*(examples(d) for d in range(end)))]
    rng = np.random.default_rng(198731)
    for step in range(480):
        ix = rng.integers(0,len(x),20)
        xx,yy,mask = x[ix].to(dev),y[ix].to(dev),valid[ix].to(dev)
        parts = model(xx)
        level,hist,peer = parts.unbind(2)
        errors = torch.stack((level-yy,level.detach()+hist-yy,(level+hist).detach()+peer-yy),axis=2)
        split = errors.square()*mask[:,:,None,None]
        loss = split[...,:10].mean()+.45*split[...,10:].mean()+.025*parts[:,:,1:].square().mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),2.)
        opt.step()
    flat = np.concatenate([w.detach().cpu().numpy().ravel() for w in model.parameters()])
    print(f'encoder trained on days 1-{end}, device={dev}, steps=480',flush=True)
    return flat


def restore(flat):
    encoder = Encoder().double()
    with torch.no_grad():
        i = 0
        for w in encoder.parameters():
            j = i+w.numel()
            w.copy_(torch.from_numpy(flat[i:j].reshape(w.shape)))
            i = j
    return encoder.eval()


def views(x,t,encoder,hidden=False):
    with torch.no_grad():
        for lo in range(0,len(t),48):
            ix = t[lo:lo+48]
            s = torch.from_numpy(tokens(x,ix)).double()
            v = torch.stack(encoder.states(s),dim=2) if hidden else encoder(s)
            yield ix,v.numpy()


def control(parts,p,mask):
    ctrl = np.concatenate((p,parts[:,:,0]),axis=2)
    return [base.links.center(np.concatenate((ctrl,z),axis=2),mask)
            for z in (parts[:,:,1],parts[:,:,2],parts[:,:,1]+parts[:,:,2])]


def calibration(day,encoder):
    raw,grid,y = [onset.settled.load(day,k) for k in ('x','time','y')]
    p = read_p(day)
    beta = strip.daily_scale(p,y).clip(.2,4.)
    c = strip.physical()[1]
    err = y/beta-p
    far = err[...,2]-err[...,1]*c[2]/c[1]*beta[1]/beta[2]
    gram = np.zeros((3,C+T+2,C+T+2))
    count = 0
    for a,b,m,ok in event.sessions(raw,grid):
        x,_ = stream(raw[a:b],m,ok,axes(),base.risk.walk.widths())
        t = np.arange(day%8,b-a-60,8)
        for ix,parts in views(x,t,encoder):
            mask = ok[ix]
            labels = base.links.center(np.stack((err[a+ix,:,2],far[a+ix]),axis=2),mask)
            for j,z in enumerate(control(parts,p[a+ix],mask)):
                values = np.concatenate((z,labels),axis=2)[mask]
                gram[j] += values.T@values
            count += mask.sum()
    return gram/count


def train(end):
    before = max(4,end-4)
    flat = pretrained(before)
    enc = restore(flat)
    dates = np.stack([calibration(d,enc) for d in range(before,end)])
    weights = []
    white = base.links.base.whitening
    for j in range(3):
        g = dates[:,j].mean(0)
        w = white(g[:C,:C],.05)
        v = np.eye(C+T+2)[:,C:C+T].copy()
        v[:C] = -w@(w.T@g[:C]@v)
        v = v@white(v.T@g@v,.85)
        pay = v.T@dates[:,j,:,-2:]
        mu = pay.mean(0)
        stable = mu**2/(mu**2+2*pay.var(0)/len(dates)+1e-8)
        weights.append(v[:-2]@(mu[:,0]*stable[:,0]*(mu.prod(1)>0)))
    return np.concatenate((base.risk.walk.widths(),axes().ravel(),flat,np.stack(weights).ravel()))


def signals(raw,grid,p,bank):
    out = np.zeros((*p.shape[:2],3,1))
    i,k = 50+100*PC,3*(C+T)
    encoder = restore(bank[i:-k])
    weights = bank[-k:].reshape(3,C+T)
    for a,b,m,ok in event.sessions(raw,grid):
        x,_ = stream(raw[a:b],m,ok,bank[50:i].reshape(100,PC),bank[:50])
        for ix,parts in views(x,np.arange(b-a),encoder):
            for j,z in enumerate(control(parts,p[a+ix],ok[ix])):
                out[a+ix,:,j,0] = z@weights[j]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
