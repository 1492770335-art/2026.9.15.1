"""Predict the shared market return from nested histories sampled on two interleaved clocks."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dense_market_basket_error_teacher_v1 as market
import experiment_dense_stock_time_memory_network_v1 as memory

onset,rel,REFERENCE = memory.onset,memory.rel,memory.REFERENCE
OUT = REFERENCE.parent / "dense_market_clock_views_v1"
ARMS = ("complete_clock","thinned_clock_consensus","clock_sensitive_component")
BIN,STEPS = 8,24
DIM = 3*len(memory.FIELDS)


def summary(data):
    parts = []
    for i in range(0,len(data),160):
        v = np.arcsinh(data[i:i+160][:,:,memory.FIELDS].astype(np.float32))
        center = v.mean(1)
        d = v-center[:,None]
        spread = np.sqrt((d*d).mean(1)+1e-4)
        skew = np.tanh(d/spread[:,None]).mean(1)
        parts.append(np.concatenate((center,spread,skew),axis=1))
    return np.concatenate(parts)


def histories(series,t):
    # The bins end before the current frame and neither clock uses a later observation.
    positions = t[:,None,None]-1-(STEPS-1-np.arange(STEPS))[None,:,None]*BIN-np.arange(BIN)[None,None]
    frames = series[positions.clip(0)]
    a,b = frames[:,:,::2].mean(2),frames[:,:,1::2].mean(2)
    return np.stack((a,b),axis=1).astype(np.float16)


class ClockNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(DIM,48),nn.LayerNorm(48),nn.SiLU())
        self.recur = nn.GRU(48,40,batch_first=True)
        self.head = nn.Sequential(nn.Linear(48+40,40),nn.LayerNorm(40),nn.SiLU(),nn.Dropout(.18),nn.Linear(40,3))

    def forward(self,current,history):
        state = self.proj(history.float())
        _,h = self.recur(state)
        return self.head(torch.cat((self.proj(current.float()),h[0]),axis=-1))


def collect():
    OUT.mkdir(exist_ok=True)
    for d in range(16):
        raw,grid,y = [onset.settled.load(d,k) for k in ("x","time","y")]
        group = []
        for a,b in rel.fields.parent.sessions(grid):
            x = summary(raw[a:b])
            t = np.arange(d%5,b-a,5)
            group.append((x[t],histories(x,t),y[a+t].mean(1)))
        c,h,y = [np.concatenate(g) for g in zip(*group)]
        np.savez(OUT/f"day_{d:02d}.npz",c=c.astype(np.float16),h=h,y=y)


def train(end):
    rows = []
    for d in range(end):
        with np.load(OUT/f"day_{d:02d}.npz") as f:
            rows.append(tuple(f[k] for k in ("c","h","y")))
    c,h,y = [torch.from_numpy(np.concatenate(g)).cuda() for g in zip(*rows)]
    del rows
    torch.manual_seed(20915)
    net = ClockNet().cuda()
    ema = ClockNet().cuda()
    ema.load_state_dict(net.state_dict())
    opt = torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.05)
    scale = y.std(0,unbiased=False).clamp_min(.1)
    for ep in range(7):
        for i in torch.randperm(len(c),device="cuda").split(128):
            ha,hb = h[i].unbind(1)
            orig = (ha.float()+hb.float())*.5
            # Randomly remove one of the two sub-clocks in every historical bin.
            mask = torch.rand((*ha.shape[:2],1),device="cuda")>.5
            sparse = torch.where(mask,ha,hb)
            full,thin = net(c[i],orig),net(c[i],sparse)
            label = y[i]/scale
            pred = .5*(full+thin)
            fit = F.smooth_l1_loss(full,label,beta=.8)+F.smooth_l1_loss(thin,label,beta=.8)
            corr = F.cosine_similarity(pred-pred.mean(0),label-label.mean(0),dim=0,eps=1e-5).mean()
            loss = fit+.15*(full-thin).square().mean()-.08*corr
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),.7)
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.035)
        if ep in (3,5):
            for group in opt.param_groups:
                group["lr"] *= .55
    return nn.utils.parameters_to_vector(ema.parameters()).detach().cpu().numpy()


@torch.inference_mode()
def design(raw,grid,code,p,bank):
    net = ClockNet().cuda().eval()
    nn.utils.vector_to_parameters(torch.from_numpy(bank).cuda(),net.parameters())
    z = np.empty((len(p),len(ARMS)))
    for a,b in rel.fields.parent.sessions(grid):
        x = summary(raw[a:b])
        for start in range(0,b-a,128):
            t = np.arange(start,min(start+128,b-a))
            ha,hb = torch.from_numpy(histories(x,t)).cuda().unbind(1)
            curr = torch.from_numpy(x[t].astype(np.float16)).cuda()
            full = net(curr,.5*(ha.float()+hb.float()))[:,2]
            avg = .5*(net(curr,ha)[:,2]+net(curr,hb)[:,2])
            z[a+t] = torch.stack((full,avg,full-avg),axis=1).cpu().numpy()
    return z


def predict(raw,grid,p,params):
    q = p.copy()
    q[...,2] += (design(raw,grid,params["code"],p,params["bank"])@params["w"])[:,None].astype(p.dtype)
    return q


if __name__=="__main__":
    collect()
    market.rolling(OUT,ARMS,train,design,REFERENCE)
