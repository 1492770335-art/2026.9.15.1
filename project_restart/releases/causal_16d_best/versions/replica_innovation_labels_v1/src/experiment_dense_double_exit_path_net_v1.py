"""Learn first passage, opposite-boundary passage and terminal displacement jointly across three maturities."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from functools import lru_cache
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as tf

from experiment_dense_market_conditional_graph_risk_v1 import event,onset,rel,strip
import experiment_dense_first_exit_state_net_v1 as first

src,base,ref,REFERENCE,read_p,TARGETS = first.src,first.base,first.ref,first.REFERENCE,first.read_p,first.TARGETS
OUT = REFERENCE.parent/'dense_double_exit_path_net_v1'
ARMS = ('first_side','return_crossing','terminal_payoff')
H,R = first.H,first.R
C,D = 6,9


class PathNet(src.Encoder):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(src.H,30) for _ in range(3)])


@lru_cache(maxsize=12)
def examples(day):
    raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
    unit = base.risk.walk.widths()
    rows = []
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(40+day%12,b-a-60,12)
        tape,_ = src.stream(raw[a:b],m,ok,src.axes(),unit)
        price = m.cumsum(0)/unit
        gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        changes = price[t[:,None]+np.arange(1,61)]-price[t,None]
        sides,crosses,values,masks = [],[],[],[]
        for h,r in zip(H,R):
            x = changes[:,:h]
            up,down = x>=np.sqrt(r),x<=-np.sqrt(r)
            u = np.where(up.any(1),up.argmax(1),h)
            d = np.where(down.any(1),down.argmax(1),h)
            sides.append((u<d).astype(int)-(d<u)+1)
            crosses.append((u<h)&(d<h))
            values.append((x[:,-1]/r).clip(-6,6))
            masks.append(gap[t+h+1]==gap[t-40])
        rows.append((src.tokens(tape,t),np.stack(sides,2),np.stack(crosses,2),np.stack(values,2).astype('f'),
                     np.stack(masks,2)))
    return tuple(torch.from_numpy(np.concatenate(col)) for col in zip(*rows))


def distribution(v):
    side = v[...,:3].softmax(-1)
    rev = v[...,3:5].sigmoid()
    return torch.stack((side[...,0]*(1-rev[...,0]),side[...,0]*rev[...,0],side[...,1],
                        side[...,2]*(1-rev[...,1]),side[...,2]*rev[...,1]),axis=-1)


def loss(output,s,both,y,valid):
    v = output.reshape(*s.shape,10)
    l = -v[...,:3].log_softmax(-1).gather(-1,s[...,None]).squeeze(-1)
    pos = (s==2).long()
    second = v[...,3:5].gather(-1,pos[...,None]).squeeze(-1)
    l += tf.binary_cross_entropy_with_logits(second,both.float(),reduction='none')*(s!=1)
    category = torch.where(s==1,2,torch.where(s==2,3,0)+both.long())
    means = v[...,5:]
    amount = means.gather(-1,category[...,None]).squeeze(-1)
    expectation = (distribution(v)*means).sum(-1)
    l += .4*tf.smooth_l1_loss(amount,y,reduction='none')+.15*(expectation-y).square()
    return (l*valid).sum()/valid.sum().clamp_min(1)


@lru_cache(maxsize=4)
def pretrained(end):
    torch.manual_seed(91574)
    torch.use_deterministic_algorithms(True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = PathNet().to(device)
    opt = torch.optim.AdamW(net.parameters(),lr=.0025,weight_decay=.015)
    batch = [torch.cat(z) for z in zip(*(examples(d) for d in range(end)))]
    rng = np.random.default_rng(91574)
    for _ in range(640):
        ix = rng.integers(0,len(batch[0]),20)
        x,s,b,y,valid = [z[ix].to(device) for z in batch]
        level,time,peer = net(x).unbind(2)
        cost = sum(loss(v,s,b,y,valid) for v in (level,level.detach()+time,(level+time).detach()+peer))
        cost += .025*(time.square().mean()+peer.square().mean())
        opt.zero_grad()
        cost.backward()
        nn.utils.clip_grad_norm_(net.parameters(),2.)
        opt.step()
    print(f'five path classes fitted on days 1-{end} ({device})',flush=True)
    return np.concatenate([v.detach().cpu().numpy().ravel() for v in net.parameters()])


def restore(params):
    enc = PathNet().double()
    i = 0
    with torch.no_grad():
        for w in enc.parameters():
            j = i+w.numel()
            w.copy_(torch.from_numpy(params[i:j].reshape(w.shape)))
            i = j
    return enc.eval()


def odds(side,rev):
    return np.stack((side[...,0]*(1-rev[...,0]),side[...,0]*rev[...,0],side[...,1],
                     side[...,2]*(1-rev[...,1]),side[...,2]*rev[...,1]),axis=-1)


def channels(parts):
    v = parts.cumsum(2).reshape(*parts.shape[:3],3,10)
    logit = v[...,:3]
    exp = np.exp(logit-logit.max(-1,keepdims=True))
    side = exp/exp.sum(-1,keepdims=True)
    rev = 1/(1+np.exp(-v[...,3:5]))
    means = v[...,5:]
    pi0 = odds(side[:,:,0],rev[:,:,0])
    pi1 = odds(side[:,:,2],rev[:,:,0])
    pi2 = odds(side[:,:,2],rev[:,:,2])
    level = (pi0*means[:,:,0]).sum(-1)
    delta = [((pi1-pi0)*means[:,:,0]).sum(-1),((pi2-pi1)*means[:,:,0]).sum(-1),
             (pi2*(means[:,:,2]-means[:,:,0])).sum(-1)]
    return level,delta


def views(parts,p,ok,unit):
    level,delta = channels(parts)
    scale = R*unit[:,None]
    base_x = np.concatenate((p,level*scale),axis=2)
    return [base.links.center(np.concatenate((base_x,z*scale),axis=2),ok) for z in delta]


def record(day,net):
    raw,grid,y = [onset.settled.load(day,k) for k in ('x','time','y')]
    p = read_p(day)
    beta = strip.daily_scale(p,y).clip(.2,4.)
    err = y/beta-p
    c = strip.physical()[1]
    far = err.copy()
    far[...,1:] -= err[...,:-1]*(c[1:]/c[:-1]*beta[:-1]/beta[1:])
    gram = np.zeros((3,D+6,D+6))
    count = 0
    for a,b,m,ok in event.sessions(raw,grid):
        unit = base.risk.walk.widths()
        x,_ = src.stream(raw[a:b],m,ok,src.axes(),unit)
        for ix,parts in src.views(x,np.arange(day%8,b-a-60,8),net):
            mask = ok[ix]
            target = base.links.center(np.concatenate((err[a+ix],far[a+ix]),axis=2),mask)
            for j,z in enumerate(views(parts,p[a+ix],mask,unit)):
                v = np.concatenate((z,target),axis=2)[mask]
                gram[j] += v.T@v
            count += mask.sum()
    return gram/count


def train(end):
    before = max(4,end-4)
    params = pretrained(before)
    enc = restore(params)
    rows = np.stack([record(d,enc) for d in range(before,end)])
    co = []
    for j in range(3):
        g = rows[:,j].mean(0)
        w = base.links.base.whitening(g[:C,:C],.04)
        v = np.eye(D+6)[:,C:D].copy()
        v[:C] = -w@(w.T@g[:C]@v)
        v = v@base.links.base.whitening(v.T@g@v,.65)
        pay = v.T@rows[:,j,:,D:]
        mu = pay.mean(0)
        trust = mu**2/(mu**2+2*pay.var(0)/len(rows)+1e-8)
        co.append(v[:D]@(mu[:,:3]*trust[:,:3]*(mu[:,:3]*mu[:,3:]>0)))
    return np.concatenate((base.risk.walk.widths(),src.axes().ravel(),params,np.stack(co).ravel()))


def signals(raw,grid,p,bank):
    i,k = 50+100*src.PC,3*D*3
    enc = restore(bank[i:-k])
    weights = bank[-k:].reshape(3,D,3)
    out = np.zeros((*p.shape[:2],3,3))
    for a,b,m,ok in event.sessions(raw,grid):
        x,_ = src.stream(raw[a:b],m,ok,bank[50:i].reshape(100,src.PC),bank[:50])
        for ix,parts in src.views(x,np.arange(b-a),enc):
            for j,z in enumerate(views(parts,p[a+ix],ok[ix],bank[:50])):
                out[a+ix,:,j] = z@weights[j]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
