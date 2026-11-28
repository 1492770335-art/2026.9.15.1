"""Predict the market's next twelve five-tick returns with a smooth horizon-conditioned readout."""
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_dense_market_band_supervision_v1 as study

strip,rel,onset,REFERENCE,read_p=study.strip,study.rel,study.onset,study.REFERENCE,study.read_p
OUT=REFERENCE.parent/'dense_market_forward_cash_v1'
ARMS=('interval_supervision','shared_horizon_curve','distant_curve_innovation')
N=12
IDX=(0,3,11)


def basis():
    t=np.arange(N,dtype=float)
    knots=np.array([0,1,3,6,11])
    return np.stack([np.interp(t,knots,c) for c in np.eye(len(knots))])


class Curve(study.Market):
    def __init__(self,smooth):
        super().__init__()
        self.head[-1]=nn.Linear(48,5 if smooth else N)
        self.register_buffer('basis',torch.tensor(basis() if smooth else np.eye(N),dtype=torch.float32))

    def forward(self,x):
        return super().forward(x)@self.basis


def records(d):
    grid,y=[onset.settled.load(d,k) for k in ('time','y')]
    y=y.mean(1,dtype=float)
    _,coef,_=strip.physical()
    units=coef[2]/coef
    with np.load(study.clock.OUT/f'day_{d:02d}.npz') as f:
        full=study.design(f['c'],f['h'])
    offsets=[];ticks=[];base=0
    for a,b in rel.fields.parent.sessions(grid):
        t=np.arange(a+d%5,b,5)
        valid=np.flatnonzero(t+60<b)
        offsets.extend(base+valid);ticks.extend(t[valid])
        base+=len(t)
    assert base==len(full)
    t=np.array(ticks)
    legs=y[t[:,None]+np.arange(0,60,5),0]*units[0]
    return full[offsets],legs,y[t]*units,t


def train(end):
    data=[records(d) for d in range(end)]
    x,legs,y=[torch.from_numpy(np.concatenate(r)).to(study.DEVICE).float() for r in tuple(zip(*data))[:3]]
    unit=x[::3].square().mean((0,1)).sqrt().clamp_min(.06)
    price=legs.std().clamp_min(.025)
    sy=y.std(0).clamp_min(.04)
    x=(x/unit).clamp(-5,5)
    packed=[unit.cpu().numpy(),np.array([price.item()],np.float32)]
    for mode in range(3):
        torch.manual_seed(71490)
        m=Curve(mode==2).to(study.DEVICE)
        opt=torch.optim.AdamW(m.parameters(),lr=.00085,weight_decay=.06)
        for step in range(460):
            ix=torch.randint(len(y),(256,),device=study.DEVICE)
            bucket=m(x[ix])
            aggregate=bucket.cumsum(-1)[:,IDX]*price
            loss=F.smooth_l1_loss(aggregate/sy,y[ix]/sy,beta=.8)
            if mode:
                loss=.5*loss+.5*F.smooth_l1_loss(bucket,legs[ix]/price,beta=.8)
            if step==305:
                opt.param_groups[0]['lr']*=.38
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(),1.)
            opt.step()
        packed.extend(v.detach().cpu().numpy().ravel() for v in m.parameters())
    return np.concatenate(packed)


@torch.inference_mode()
def unpack(bank):
    unit=torch.from_numpy(bank[:study.W]).double()
    price=bank[study.W]
    pos=study.W+1
    nets=[]
    for mode in range(3):
        m=Curve(mode==2).double().eval()
        for w in m.parameters():
            w.copy_(torch.from_numpy(bank[pos:pos+w.numel()].reshape(w.shape)))
            pos+=w.numel()
        nets.append(m)
    assert pos==len(bank)
    return unit,price,nets


@torch.inference_mode()
def forecasts(x,params):
    unit,price,nets=params
    z=(torch.from_numpy(x).double()/unit).clamp(-5,5)
    return [m(z).numpy()*price for m in nets]


def signals(raw,grid,p,bank):
    params=unpack(bank)
    _,coef,_=strip.physical()
    out=np.empty((len(p),3,2))
    for a,b in rel.fields.parent.sessions(grid):
        market=study.clock.summary(raw[a:b])
        for i in range(0,len(market),64):
            t=np.arange(i,min(i+64,len(market)))
            x=study.design(market[t].astype(np.float16),study.clock.histories(market,t))
            control,free,smooth=forecasts(x,params)
            for j,(left,right) in enumerate(((1,4),(4,12))):
                sums=[v[:,:right].sum(-1) for v in (control,free,smooth)]
                far=(smooth[:,left:right]-control[:,left:right]).sum(-1)
                # The endpoint-only control has identifiable sums at 5,20,60, including both strip boundaries.
                out[a+t,:,j]=np.column_stack((sums[1]-sums[0],sums[2]-sums[0],far))*coef[j+1]/coef[2]
    return np.broadcast_to(out[:,None],(*p.shape[:2],3,2))


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
