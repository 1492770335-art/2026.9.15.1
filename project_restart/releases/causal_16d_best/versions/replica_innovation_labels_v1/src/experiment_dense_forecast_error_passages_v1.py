"""Learn boundary crossings of the error path around the already-issued three-horizon forecast."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from functools import lru_cache
import sys

import numpy as np
import torch
from torch import nn

from experiment_dense_market_conditional_graph_risk_v1 import event,onset,rel,strip
import experiment_dense_double_exit_path_net_v1 as passage

src,base,ref,REFERENCE,read_p,TARGETS = passage.src,passage.base,passage.ref,passage.REFERENCE,passage.read_p,passage.TARGETS
OUT = REFERENCE.parent/'dense_forecast_error_passages_v1'
ARMS = ('snapshot_error','error_odds_revision','temporal_error_payoff')
H,R = passage.H,passage.R
KNOTS = np.r_[0,H]
INTERP = np.stack([np.interp(np.arange(1,61),KNOTS,row) for row in np.eye(4)[1:]])


class ResidualNet(src.Encoder):
    def __init__(self):
        super().__init__(src.D+6)
        self.heads = nn.ModuleList([nn.Linear(src.H,30) for _ in range(3)])


def codec():
    with np.load(onset.settled.GEO) as geo:
        return np.r_[float(geo['k']),geo['coeff'][:,0],geo['coeff'][:,1]]


def stream(raw,m,ok,p,basis,unit,code):
    tape,_ = src.stream(raw,m,ok,basis,unit)
    k,intercept,scale = code[0],code[1:4],code[4:]
    issued = scale[2]*np.log1p(((p.astype(float)-intercept)*k/scale).clip(-.99))/k
    path = m.cumsum(0)
    gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    t = np.arange(len(raw))
    completed = []
    for j,h in enumerate(H):
        old = (t-h).clip(0)
        valid = (t>=h)[:,None]&(gaps[t+1]==gaps[old])
        completed.append(valid*(path[t]-path[old]-issued[old,:,j]))
    values = np.tanh(np.concatenate((issued,np.stack(completed,axis=2)),axis=2)/(unit[:,None]*np.tile(R,2)))
    x = np.concatenate((tape[:,:,:-1],values*ok[...,None],tape[:,:,-1:]),axis=2)
    return x.astype(np.float32),issued,path,gaps


@lru_cache(maxsize=12)
def samples(day):
    raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
    p = read_p(day)
    unit = base.risk.walk.widths()
    rows = []
    for a,b,m,ok in event.sessions(raw,grid):
        x,issued,path,gap = stream(raw[a:b],m,ok,p[a:b],src.axes(),unit,codec())
        t = np.arange(40+day%11,b-a-60,11)
        changes = path[t[:,None]+np.arange(1,61)]-path[t,None]
        error = (changes-(issued[t]@INTERP).transpose(0,2,1))/unit
        sides,both,y,valid = [],[],[],[]
        for h,r in zip(H,R):
            v = error[:,:h]
            up,down = v>=np.sqrt(r),v<=-np.sqrt(r)
            hi = np.where(up.any(1),up.argmax(1),h)
            lo = np.where(down.any(1),down.argmax(1),h)
            sides.append(1+(hi<lo).astype(int)-(lo<hi))
            both.append((hi<h)&(lo<h))
            y.append((v[:,-1]/r).clip(-6,6))
            valid.append(gap[t+h+1]==gap[t-40])
        rows.append((src.tokens(x,t),np.stack(sides,2),np.stack(both,2),np.stack(y,2).astype('f'),np.stack(valid,2)))
    return tuple(torch.from_numpy(np.concatenate(col)) for col in zip(*rows))


def train(end):
    torch.manual_seed(37181)
    torch.use_deterministic_algorithms(True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResidualNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.025)
    training = [torch.cat(v) for v in zip(*(samples(d) for d in range(4,end)))]
    gen = np.random.default_rng(37181)
    for _ in range(420):
        ix = gen.integers(0,len(training[0]),20)
        x,s,b,y,valid = [z[ix].to(dev) for z in training]
        cur,history,peer = model(x).unbind(2)
        total = sum(passage.loss(v,s,b,y,valid) for v in (cur,cur.detach()+history,(cur+history).detach()+peer))
        total += .04*(history.square().mean()+peer.square().mean())
        opt.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(),2.)
        opt.step()
    flat = np.concatenate([v.detach().cpu().numpy().ravel() for v in model.parameters()])
    print(f'forecast error paths fitted on days 5-{end} ({dev})',flush=True)
    return np.concatenate((base.risk.walk.widths(),src.axes().ravel(),codec(),flat))


def restore(flat):
    model = ResidualNet().double()
    i = 0
    with torch.no_grad():
        for w in model.parameters():
            j = i+w.numel()
            w.copy_(torch.from_numpy(flat[i:j].reshape(w.shape)))
            i = j
    return model.eval()


def signals(raw,grid,p,bank):
    i = 50+100*src.PC
    code = bank[i:i+7]
    model = restore(bank[i+7:])
    out = np.zeros((*p.shape[:2],3,3))
    for a,b,m,ok in event.sessions(raw,grid):
        x,issued,_,_ = stream(raw[a:b],m,ok,p[a:b],bank[50:i].reshape(100,src.PC),bank[:50],code)
        conv = R*bank[:50,None]*(code[4:]/code[-1])
        for t,parts in src.views(x,np.arange(b-a),model):
            cur,delta = passage.channels(parts)
            prior,odds,total = cur,delta[0]+delta[1],sum(delta)
            z = np.stack((prior,odds,total),axis=2)*conv[:,None]
            # Use only relative signals at issue time; absorb a common calibration offset.
            out[a+t] = base.links.center(z.reshape(len(t),50,9),ok[t]).reshape(len(t),50,3,3)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
