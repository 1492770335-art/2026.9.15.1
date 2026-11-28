"""Transport marginal signal skill and subtract the component already priced by today's forecast."""
from functools import lru_cache
import sys

import numpy as np
import torch
from torch.nn import functional as F

import experiment_dense_cash_aligned_response_v1 as parent

pre,strip,event,onset,rel = parent.pre,parent.strip,parent.event,parent.onset,parent.rel
REFERENCE,read_p = parent.REFERENCE,parent.read_p
OUT = REFERENCE.parent/'dense_forecast_overlap_transport_v1'
ARMS = ('current_overlap','current_incumbent_skill','both_skills_updated')
D, P = 12, 3


@torch.inference_mode()
def features(raw,p,ok,bank,model):
    z = pre.inputs(raw)
    rows = []
    for a in range(0,len(z),64):
        x = torch.tensor(((z[a:a+64]-bank[:pre.N])/bank[pre.N:2*pre.N]).clip(-6,6),
                         device=pre.DEVICE,dtype=torch.float32)
        n = len(x)
        x = F.pad(x,(0,0,0,0,0,64-n))
        legs = [h[:n].cpu().numpy().astype(float) for h in model(x)]
        rows.append(np.concatenate((p[a:a+n],*legs),axis=2))
    x = np.concatenate(rows)
    return (x-(x*ok[...,None]).sum(1,keepdims=True)/ok.sum(1)[:,None,None].clip(1))*ok[...,None]


def moment(x,y,valid):
    z = np.concatenate((x,y[...,None]),axis=2)
    z = (z-(z*valid[...,None]).sum(1,keepdims=True)/valid.sum(1)[:,None,None].clip(1))*valid[...,None]
    v = z.reshape(-1,D+1)
    return v.T@v/max(valid.sum(),1)


@lru_cache(maxsize=12)
def records(day):
    raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
    p = read_p(day)
    bank = pre.encoder()
    net = pre.restore(bank)
    g = np.zeros((2,D+1,D+1))
    n = 0
    for a,b,m,ok in event.sessions(raw,grid):
        x = features(raw[a:b],p[a:b],ok,bank,net)
        path = m.cumsum(0)
        gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        t = np.arange(48+day%4,len(m)-60,4)
        for j,h in enumerate((20,60)):
            g[j] += moment(x[t],path[t+h]-path[t],gaps[t+h+1]==gaps[t])*len(t)
        n += len(t)
    return g/n


def train(end):
    hist = np.stack([records(d) for d in range(4,end)])
    return np.r_[hist.mean(0).ravel(),pre.encoder()]


def weights(train_g, current_g, paid_g, reliability, j):
    source_sd = np.sqrt(np.diag(train_g).clip(1e-9))
    sd = np.sqrt(np.diag(current_g).clip(1e-9))
    price_sd = np.sqrt(max(paid_g[-1,-1],1e-9))
    paid_sd = np.sqrt(np.diag(paid_g).clip(1e-9))
    current = current_g/np.outer(sd,sd)
    hist = train_g[:D,-1]/(source_sd[:D]*source_sd[-1])
    recent = paid_g[:D,-1]/(paid_sd[:D]*price_sd)
    # Find signal variations orthogonal to the three present incumbent forecasts.
    residual = np.eye(D)[:,P:]
    residual[:P] = -np.linalg.solve(current[:P,:P]+.015*np.eye(P),current[:P,P:])
    g = residual.T@current@residual+.75*np.eye(D-P)
    all_skills = np.stack((hist.copy(),hist.copy(),hist.copy()),axis=1)
    all_skills[:P,1] += reliability*(recent[:P]-hist[:P])
    all_skills[:,2] += reliability*(recent-hist)
    # Work in raw cash units, then restore the original forecast scale using its marginal slope.
    slope = (all_skills[j+1]*price_sd/sd[j+1]).clip(.08,5)
    return (residual@np.linalg.solve(g,residual.T@all_skills))/sd[:,None]*(price_sd/slope)


def session(x,m,ok,h,hist,j):
    path = m.cumsum(0)
    gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    observed = np.zeros((D,D))
    paid = np.zeros((D+1,D+1))
    mass_x,mass_y = 0.,0.
    cursor = 0
    out = np.zeros((*m.shape,3))
    for i in range(0,len(m),64):
        if i:
            u = x[i-64:i:4].reshape(-1,D)
            observed = .925*observed+u.T@u/len(u)
            mass_x = .925*mass_x+1
        t = np.arange(cursor,i-h+1,4)
        if len(t):
            cursor = t[-1]+4
            good = gap[t+h+1]==gap[t]
            if good.sum()>=100:
                paid = .925*paid+moment(x[t],path[t+h]-path[t],good)
                mass_y = .925*mass_y+1
        current = (observed+2*hist[:D,:D])/(mass_x+2)
        mature = (paid+2*hist)/(mass_y+2)
        rel_score = mass_y/(mass_y+4)
        w = weights(hist,current,mature,rel_score,j)
        out[i:i+64] = np.clip(x[i:i+64]@w,-.5,.5)
    return out


def signals(raw,grid,p,params):
    cut = 2*(D+1)**2
    hist = params[:cut].reshape(2,D+1,D+1)
    bank = params[cut:]
    net = pre.restore(bank)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        x = features(raw[a:b],p[a:b],ok,bank,net)
        for j,h in enumerate((20,60)):
            out[a:b,:,:,j] = session(x,m,ok,h,hist[j],j)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
