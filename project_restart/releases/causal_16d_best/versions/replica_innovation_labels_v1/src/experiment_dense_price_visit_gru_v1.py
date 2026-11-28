"""Read the completed visits to one stock-price cell with a gated recurrent network."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from functools import lru_cache
import sys

import numpy as np
import torch
from torch import nn

from experiment_dense_market_conditional_graph_risk_v1 import event, onset, rel, strip
import experiment_dense_returning_pressure_spells_v1 as visit


ref,read_p,REFERENCE = visit.ref,visit.read_p,visit.REFERENCE
OUT = REFERENCE.parent/'dense_price_visit_gru_v1'
ARMS = ('current_visit','previous_visits_increment','full_recurrent_signal')
TAPS,INPUT,HIDDEN = 5,16,16


def encode(raw,m,ok,p,t,unit):
    path = m.cumsum(0)
    cell = np.rint(path/unit).astype(int)
    entry,last = visit.history(cell,ok)
    b = np.tanh(.7*np.arcsinh(raw[...,visit.base.risk.walk.ref.model.FIELDS].astype(float)))
    book = b[...,::2]-b[...,1::2]
    gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    stop = np.broadcast_to(t[:,None],(len(t),50))
    tokens = np.empty((len(t),50,TAPS,INPUT),np.float32)
    for k in reversed(range(TAPS)):
        end = stop.clip(0)
        start = entry[end,visit.STOCK]
        previous = last[start,visit.STOCK]
        good = (stop>=0)&(gaps[t+1]==gaps[start,visit.STOCK])&(t[:,None]-end<=1200)
        age = np.log1p(end-start)/4.
        away = np.where(previous>=0,np.log1p((start-previous).clip(0))/5.,0.)
        approach = np.sign(cell[start,visit.STOCK]-cell[(start-1).clip(0),visit.STOCK])
        residual = (path[end,visit.STOCK]-path[start,visit.STOCK])/unit
        memory_age = np.log1p(t[:,None]-end)/6.
        q = book[end,visit.STOCK]
        token = np.concatenate((q,q-book[start,visit.STOCK],
                                np.stack((age,away,approach,residual,memory_age,good),axis=2)),axis=2)
        tokens[:,:,k] = token*good[...,None]
        stop = np.where(good,previous,-1)
    return tokens,np.tanh(p[t].astype(np.float32))


class PriceVisitGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = nn.GRUCell(INPUT,HIDDEN)
        self.current = nn.Sequential(nn.Linear(INPUT+3,HIDDEN),nn.Tanh())
        self.context = nn.Linear(HIDDEN,HIDDEN)
        self.local = nn.Linear(HIDDEN,4)
        self.increment = nn.Linear(HIDDEN,4,bias=False)
        nn.init.zeros_(self.local.weight)
        nn.init.zeros_(self.local.bias)
        nn.init.zeros_(self.increment.weight)

    def forward(self,tokens,p):
        shape = p.shape[:-1]
        series = tokens.reshape(-1,TAPS,INPUT)
        h = torch.zeros_like(series[:,0,:HIDDEN])
        for k in range(TAPS-1):
            x = series[:,k]
            h = torch.where(x[:,-1:]>0,self.recurrent(x,h),h)
        current = self.current(torch.cat((series[:,-1],p.reshape(-1,3)),dim=-1))
        fixed = self.local(current).reshape(*shape,4)
        update = self.increment(h*torch.sigmoid(self.context(current))).reshape(*shape,4)
        return fixed-fixed.mean(1,keepdim=True),update-update.mean(1,keepdim=True)


@lru_cache(maxsize=12)
def dataset(day):
    raw,grid,y = [onset.settled.load(day,k) for k in ('x','time','y')]
    p = read_p(day)
    scale = strip.daily_scale(p,y).clip(.2,4.)
    c = strip.physical()[1]
    ratio = c[1:]/c[:-1]*scale[:-1]/scale[1:]
    rows = []
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(24+day%12,b-a-60,12)
        x,q = encode(raw[a:b],m,ok,p[a:b],t,visit.base.risk.walk.widths())
        error = y[a+t].astype(float)/scale-p[a+t]
        z = np.concatenate((error[...,1:],error[...,1:]-error[...,:-1]*ratio),axis=2)
        rows.append((x,q,strip.center(z).clip(-4,4).astype(np.float32),ok[t]))
    return tuple(np.concatenate(v) for v in zip(*rows))


def train(end):
    torch.manual_seed(7193)
    torch.use_deterministic_algorithms(True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PriceVisitGRU().to(device)
    x,p,y,valid = [torch.as_tensor(np.concatenate(v),device=device) for v in zip(*(dataset(d) for d in range(4,end)))]
    optimizer = torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.035)
    for epoch in range(6):
        for idx in torch.randperm(len(p),device=device).split(48):
            a,b = model(x[idx],p[idx])
            def loss(value):
                error = torch.nn.functional.smooth_l1_loss(value,y[idx],reduction='none',beta=1.)
                return (error*valid[idx][...,None]).mean()
            objective = loss(a+b)+.5*loss(a)+.005*(a.square().mean()+b.square().mean())
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),.6)
            optimizer.step()
    weights = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy()
    print(f'fit days 5–{end} on {device}, {len(y)} timestamps',flush=True)
    return np.r_[visit.base.risk.walk.widths(),weights]


def signals(raw,grid,p,bank):
    model = PriceVisitGRU().eval()
    torch.nn.utils.vector_to_parameters(torch.as_tensor(bank[50:].astype(np.float32)),model.parameters())
    out = np.zeros((*p.shape[:2],3,2))
    with torch.no_grad():
        for a,b,m,ok in event.sessions(raw,grid):
            x,q = encode(raw[a:b],m,ok,p[a:b],np.arange(b-a),bank[:50])
            for start in range(0,len(x),64):
                sl = slice(start,start+64)
                fixed,delta = model(torch.from_numpy(x[sl]),torch.from_numpy(q[sl]))
                out[a+start:a+start+len(delta)] = torch.stack((fixed,delta,fixed+delta),dim=2).numpy()[...,:2]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__ == '__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
