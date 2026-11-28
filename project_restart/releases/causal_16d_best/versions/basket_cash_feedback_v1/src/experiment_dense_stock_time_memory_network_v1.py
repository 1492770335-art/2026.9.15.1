"""Train a joint stock/time network with learned memory slots from scratch on each date prefix."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dense_directed_book_transfer_v1 as graph

book,onset,rel,REFERENCE = graph.book,graph.onset,graph.rel,graph.REFERENCE
OUT = REFERENCE.parent / "dense_stock_time_memory_network_v1"
ARMS = ("local_temporal_memory","joint_stock_time_memory","cross_stock_contribution")
LAGS = (0,4,12,32,80,192)
FIELDS = sorted(set(book.phase.OBS+[0,2,4,5,7,9,13,18,20,21,24,30,33,45,52,58,62,70,75,80,82]))
WIDTH = len(FIELDS)+4
SLOTS = 8


def stream(data):
    raw = np.arcsinh(data[:,:,FIELDS].astype(np.float32))
    ret = data[:,:,0].astype(float)-.00136
    total = np.vstack((np.zeros((1,50)),ret.cumsum(0)))
    end = np.arange(1,len(data)+1)
    tail = [np.arcsinh((total[end]-total[(end-h).clip(0)])/np.sqrt(h)) for h in (5,20,60)]
    volsum = np.vstack((np.zeros((1,50)),(ret**2).cumsum(0)))
    vol = np.sqrt((8+volsum[end]-volsum[(end-96).clip(0)])/(8+np.minimum(end,96)[:,None])).clip(.5,3)
    return np.concatenate((raw,np.stack([*tail,np.log(vol)],axis=2)),axis=2).astype(np.float16),vol.astype(np.float32)


class MemoryNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(WIDTH,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,32))
        self.identity = nn.Parameter(torch.zeros(50,32))
        self.key = nn.Linear(32,SLOTS)
        self.query = nn.Linear(32,SLOTS)
        self.value = nn.Linear(32,24)
        self.age = nn.Parameter(torch.zeros(len(LAGS),SLOTS))
        self.temporal = nn.Linear(32,1)
        self.fuse = nn.Sequential(nn.Linear(32*2+24,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.15))
        self.head = nn.Linear(64,3)
        self.aux = nn.Linear(64,12)

    def forward(self,x):
        h = self.embed(x.float())+self.identity
        now = h[:,0]
        weights = torch.softmax(self.temporal(h),dim=1)
        own = (weights*h).sum(1)
        key = torch.softmax(self.key(h),dim=-1)*torch.softmax(self.age,dim=0)[None,:,None]
        value = self.value(h)
        total = torch.einsum("blsk,blsf->bkf",key,value)
        self_value = torch.einsum("blsk,blsf->bskf",key,value)
        self_mass = key.sum(1)
        # Each receiver reads only other names from the shared memory.
        slots = (total[:,None]-self_value)/(self_mass.sum(1,keepdim=True)-self_mass).clamp_min(1e-5)[...,None]
        query = torch.softmax(self.query(now),dim=-1)
        shared = torch.einsum("bsk,bskf->bsf",query,slots)
        local = self.fuse(torch.cat((now,own,torch.zeros_like(shared)),-1))
        combined = self.fuse(torch.cat((now,own,shared),-1))
        return self.head(local),self.head(combined),self.aux(combined)


def patches(x,t):
    return x[(t[:,None]-np.array(LAGS)).clip(0)]


def collect():
    OUT.mkdir(exist_ok=True)
    for d in range(16):
        raw,grid,y = [onset.settled.load(d,k) for k in ("x","time","y")]
        rows = []
        for a,b in rel.fields.parent.sessions(grid):
            data = raw[a:b]
            x,v = stream(data)
            t = np.arange(d%9,len(data)-60,9)
            aux = np.concatenate([x[t+h,:,:4].astype(np.float32)-x[t,:,:4] for h in (5,20,60)],axis=2)
            target = y[a+t]-y[a+t].mean(1,keepdims=True)
            rows.append((patches(x,t),v[t],target,aux))
        x,v,y,aux = [np.concatenate(group) for group in zip(*rows)]
        np.savez(OUT/f"day_{d:02d}.npz",x=x,v=v,y=y,aux=aux.astype(np.float16))


def train(end):
    rows = []
    for d in range(end):
        with np.load(OUT/f"day_{d:02d}.npz") as f:
            rows.append(tuple(f[k] for k in ("x","v","y","aux")))
    x,v,y,aux = [torch.from_numpy(np.concatenate(group)).cuda() for group in zip(*rows)]
    del rows
    torch.manual_seed(18062)
    net = MemoryNetwork().cuda()
    ema = MemoryNetwork().cuda()
    ema.load_state_dict(net.state_dict())
    opt = torch.optim.AdamW(net.parameters(),lr=.0012,weight_decay=.045)
    for epoch in range(8):
        for i in torch.randperm(len(x),device="cuda").split(48):
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                loc,joint,next_state = net(x[i])
            risk = (.5+.5*v[i]).unsqueeze(-1)
            correct = y[i]/risk
            def loss(p):
                p = p.float()
                scaled = p*risk
                return F.smooth_l1_loss(p,correct,beta=.85)-.065*F.cosine_similarity(scaled,y[i],dim=1,eps=1e-5).mean()
            objective = loss(joint)+.50*loss(loc)+.15*F.smooth_l1_loss(next_state.float(),aux[i].float(),beta=.7)
            opt.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),.9)
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.04)
        if epoch in (3,6):
            for group in opt.param_groups:
                group["lr"] *= .5
    print("trained",end,flush=True)
    return nn.utils.parameters_to_vector(ema.parameters()).detach().cpu().numpy()


@torch.inference_mode()
def design(raw,grid,code,p,bank):
    net = MemoryNetwork().cuda().eval()
    nn.utils.vector_to_parameters(torch.from_numpy(bank).cuda(),net.parameters())
    out = np.zeros((*p.shape[:2],len(ARMS),3))
    for a,b in rel.fields.parent.sessions(grid):
        x,v = stream(raw[a:b])
        for start in range(0,b-a,64):
            t = np.arange(start,min(start+64,b-a))
            loc,joint,_ = net(torch.from_numpy(patches(x,t)).cuda())
            z = torch.stack((loc,joint,joint-loc),2).cpu().numpy()
            z *= (.5+.5*v[t])[:,:,None,None]
            out[a+t] = z-z.mean(1,keepdims=True)
    return out


def predict(raw,grid,p,params):
    return book.path.corrected(p,design(raw,grid,params["code"],p,params["bank"]),params)


if __name__=="__main__":
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
