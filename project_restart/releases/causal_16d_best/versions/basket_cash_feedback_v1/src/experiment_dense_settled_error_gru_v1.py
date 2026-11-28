"""Condition residual-rank direction on successive realized forecast errors for a common twenty-tick interval."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dense_rank_direction_value_v1 as head
from experiment_dense_peer_activity_bridge_v1 import load
from experiment_dense_session_slope_v1 import GEO
from prepare_dense_direction_rank_reference_v1 import OUT as REFERENCE

rel = head.rel
OUT = REFERENCE.parent / "dense_settled_error_gru_v1"
ARMS = ("settled_memory", "magnitude_dependent_memory", "forecast_revision_memory")
USES_FORECAST = True
STEPS = 8


def observations(raw, grid, p):
    with np.load(GEO) as g:
        k, coef, missing = float(g["k"]), g["coeff"][:, 1], g["row"]
    stream = np.zeros((*raw.shape[:2], 8), dtype=np.float32)
    starts = np.zeros(len(raw), dtype=int)
    for a, b in rel.fields.parent.sessions(grid):
        starts[a:b] = a
        t = np.arange(20, b-a)
        if not len(t):
            continue
        path = np.log1p(k*(raw[a:b, :, 0].astype(float)-.00136)).cumsum(0)*coef[2]/k
        gaps = np.vstack((np.zeros(50), np.all(raw[a:b]==missing, axis=2).cumsum(0)))
        valid = gaps[t+1]==gaps[t-20]
        def center(x):
            mean = (x*valid).sum(1,keepdims=True)/valid.sum(1,keepdims=True).clip(1)
            return valid*(x-mean)
        # Convert predictions to a common return scale. All four shorter forecasts have expired.
        short = np.stack([center(p[a+t-5*j,:,0]*coef[2]/coef[0]) for j in (4,3,2,1)])
        small_moves = np.stack([center(path[t-5*j+5]-path[t-5*j]) for j in (4,3,2,1)])
        miss = small_moves-short
        revision = short.sum(0)-center(p[a+t-20,:,1]*coef[2]/coef[1])
        channels = (miss[-1],miss.sum(0),miss[-1]-miss[0],np.abs(miss).sum(0),
                    revision,revision*np.sign(miss.sum(0)),np.abs(revision),valid.astype(float))
        stream[a+t] = np.stack([np.tanh(c) for c in channels],axis=-1)
    return stream, starts


def history(stream, starts, t):
    times = t[:,None]-5*np.arange(STEPS-1,-1,-1)[None]
    ok = times>=starts[t,None]
    return (stream[times.clip(0)]*ok[:,:,None,None]).transpose(0,2,1,3)


def collect():
    OUT.mkdir(exist_ok=True)
    code = rel.fields.parent.codec()
    for d in range(4,16):
        p = np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else head.ranks.band.pairs.net.future.forecast(d)
        raw,grid = load(d,"x"),load(d,"time")
        x = head.ranks.band.pairs.net.features(raw,grid,code,p)
        stream, starts = observations(raw,grid,p)
        y = load(d,"y")[:,:,2].astype(float)
        y -= y.mean(1,keepdims=True)
        q = x[:,:,-1].astype(float)
        scale = ((q*y).sum()/(q*q).sum()).clip(.2,4)
        t = np.concatenate([np.arange(a+40+d%5,b-61,6) for a,b in rel.fields.parent.sessions(grid)])
        np.savez(OUT/f"day_{d:02d}.npz",x=x[t].astype(np.float16),
                 seq=history(stream,starts,t).astype(np.float16),y=(y[t]/scale-q[t]).astype(np.float32))


class SettledDirection(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.book = nn.Sequential(nn.Linear(width,32),nn.LayerNorm(32),nn.SiLU(),nn.Linear(32,20),nn.SiLU())
        self.gru = nn.GRU(8,12,batch_first=True)
        self.base = nn.Linear(20,2*head.WIDTH+1)
        self.effect = nn.Sequential(nn.Linear(32,20),nn.SiLU(),nn.Linear(20,head.WIDTH+1))

    def forward(self,x,s):
        book = self.book(x)
        _,h = self.gru(s.reshape(-1,STEPS,8))
        h = h[0].reshape(*x.shape[:2],-1)
        common,local,radius = self.base(book).split((1,head.WIDTH,head.WIDTH),-1)
        correction = self.effect(torch.cat((book,h),-1))-self.effect(torch.cat((book,torch.zeros_like(h)),-1))
        return radius.softmax(-1),common+local,correction[:,:,:1],correction[:,:,1:]


def train(end):
    torch.manual_seed(73513)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for d in range(4,end):
        with np.load(OUT/f"day_{d:02d}.npz") as f:
            rows.append(tuple(f[k] for k in ("x","seq","y")))
    x,s,y = [torch.as_tensor(np.concatenate(col),device=device,dtype=torch.float32) for col in zip(*rows)]
    unit = x.square().mean((0,1)).sqrt().clamp_min(.04)
    x = (x/unit).clamp(-5,5)
    y = y/y.square().mean().sqrt().clamp_min(.02)
    ranks = head.ranks.rank_targets(y)
    levels = (ranks*y[...,None]).sum((0,1))/ranks.sum((0,1))
    span = .5*(levels[5:].flip(0)-levels[:5])
    pos,neg = ranks[:,:,5:].flip(-1), ranks[:,:,:5]
    net = SettledDirection(len(unit)).to(device)
    opt = torch.optim.AdamW(net.parameters(),lr=.0015,weight_decay=.04)
    for _ in range(6):
        for ids in torch.randperm(len(x),device=device).split(32):
            mass,base,shift,shape = net(x[ids],s[ids])
            def signed(v):
                return ((pos[ids]*F.softplus(-v)+neg[ids]*F.softplus(v))*span).sum(-1).mean()
            loss = signed(base+shift+shape)+.35*signed(base)+.25*signed(base+shift)
            loss -= .45*((pos[ids]+neg[ids])*mass.log()).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
    torch.save(dict(state=net.cpu().state_dict(),unit=unit.cpu(),span=span.cpu()),OUT/f"net_{end}.pt")
    return np.array([end])


@torch.inference_mode()
def views(raw,grid,code,bank,p):
    pack = torch.load(OUT/f"net_{int(bank[0])}.pt",map_location="cpu",weights_only=True)
    net = SettledDirection(len(pack["unit"])).double().eval()
    net.load_state_dict(pack["state"])
    x = head.ranks.band.pairs.net.features(raw,grid,code,p)
    stream,starts = observations(raw,grid,p)
    output = []
    for t in np.array_split(np.arange(len(x)),(len(x)+31)//32):
        u = (torch.from_numpy(x[t]).double()/pack["unit"]).clamp(-5,5)
        s = torch.from_numpy(history(stream,starts,t)).double()
        m,b,c,d = net(u,s)
        partial = s.clone()
        partial[:,:,:,4:7] = 0
        _,_,old,old_shape = net(u,partial)
        def read(v):
            return (m*pack["span"]*(v.sigmoid()-.5)).sum(-1)
        full = read(b+c+d)
        output.append(torch.stack((full-read(b),full-read(b+c),full-read(b+old+old_shape)),-1).numpy())
    z = np.concatenate(output)
    return z-z.mean(1,keepdims=True)


if __name__=="__main__":
    torch.set_num_threads(1)
    collect()
    rel.main(OUT,ARMS,train,views,with_prediction=True,rolling_reference=REFERENCE)
