"""Fit joint ordered conditional quantiles, separating direction shared across horizons from sign asymmetry."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dense_stock_tail_marks_v1 as previous

reliable, basket, onset, rel, REFERENCE = previous.reliable, previous.basket, previous.onset, previous.rel, previous.REFERENCE
OUT = REFERENCE.parent/"dense_reflected_quantile_curves_v1"
ARMS = ("symmetric_direction", "asymmetric_departure", "central_vs_tail_direction")
TAUS = torch.tensor([.025,.10,.25,.50,.75,.90,.975])


class Curves(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.id = nn.Embedding(50,6)
        self.encode = nn.Sequential(nn.Linear(2*width+6,56),nn.LayerNorm(56),nn.SiLU(),
                                    nn.Linear(56,28),nn.SiLU())
        self.common = nn.Linear(28,7)
        self.specific = nn.Linear(28,21)

    def forward(self,x):
        pooled = x.mean(1,keepdim=True).expand_as(x)
        h = self.encode(torch.cat((x,pooled,self.id.weight.expand(len(x),-1,-1)),dim=-1))
        delta = self.specific(h).reshape(len(x),50,3,7)
        raw = self.common(h).unsqueeze(2)+.5*(delta-delta.mean(2,keepdim=True))
        med = raw[...,3:4]
        left = med-F.softplus(raw[...,:3]).flip((-1,)).cumsum(-1).flip((-1,))
        right = med+F.softplus(raw[...,4:]).cumsum(-1)
        return torch.cat((left,med,right),dim=-1)


def train(end, errors=None, output=OUT):
    torch.manual_seed(36189)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = [reliable.samples(d) for d in range(4,end)]
    if errors is None:
        errors = []
        for d in range(4,end):
            with np.load(basket.OUT/f"day_{d:02d}.npz") as f:
                errors.append(f["y"])
    targets,scales = [],[]
    for h in range(3):
        by_h = [(x,e[...,h].astype(float),g) for (x,_,g),e in zip(data,errors)]
        if h==0:
            # The rolling Y1 residual is already cross-sectionally centered.
            own = [r[1] for r in by_h]
            std = np.stack([np.sqrt(np.mean(np.square(np.concatenate(own[1-k::2])),axis=0)).clip(.1) for k in (0,1)])
            targets.append(np.concatenate([z/std[i%2] for i,z in enumerate(own)]))
            scales.append(np.broadcast_to(np.sqrt(np.mean(np.square(np.concatenate(own)),axis=0)).clip(.1),(12,50)))
        else:
            tables = [reliable.risk(by_h[1-k::2]) for k in range(2)]
            targets.append(np.concatenate([previous.tail.labels(x,y,g,tables[i%2]) for i,(x,y,g) in enumerate(by_h)]))
            scales.append(reliable.risk(by_h)[...,1])
    x = torch.as_tensor(np.concatenate([v[0] for v in data]),device=device)
    y = torch.as_tensor(np.stack(targets,-1),device=device,dtype=torch.float32)
    unit = x.square().mean((0,1)).sqrt().clamp_min(.03)
    x = (x/unit).clamp(-6,6)
    levels = TAUS.to(device)
    model = Curves(len(unit)).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=.0012,weight_decay=.06)
    def score(curves,truth):
        diff = truth.unsqueeze(-1)-curves
        return torch.maximum(levels*diff,(levels-1)*diff).mean()
    for _ in range(7):
        for t in torch.randperm(len(x),device=device).split(64):
            a = model(x[t])
            b = -model(-x[t]).flip((-1,))
            equivariant = .5*(a+b)
            # Both curves are ordered; the convex combination preserves order.
            general = .8*a+.2*b
            loss = score(equivariant,y[t])+score(general,y[t])+.025*(a-b).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
    torch.save(dict(state=model.cpu().state_dict(),unit=unit.cpu(),sd=torch.from_numpy(np.stack(scales,-1)),
                    rate=torch.from_numpy(basket.front.direction.table(end))),output/f"fit_{end}.pt")
    return np.array([end])


@torch.inference_mode()
def design(raw,grid,code,p,bank,output=OUT):
    pack = torch.load(output/f"fit_{int(bank[0])}.pt",map_location="cpu",weights_only=True)
    net = Curves(len(pack["unit"])).double().eval()
    net.load_state_dict(pack["state"])
    x = basket.features(raw,grid,code,p,pack["rate"].numpy())
    sd = pack["sd"].numpy()[reliable.cells(x,grid),np.arange(50)]
    cuts = np.r_[0,(TAUS.numpy()[1:]+TAUS.numpy()[:-1])/2,1]
    quad = torch.from_numpy(np.diff(cuts)).double()
    parts = []
    for t in np.array_split(np.arange(len(x)),max(1,(len(x)+95)//96)):
        z = (torch.from_numpy(x[t]).double()/pack["unit"]).clamp(-6,6)
        a = net(z)
        b = -net(-z).flip((-1,))
        shared = .5*(a+b)
        mean = shared@quad
        asym = (a-b)@quad
        skew = mean-shared[...,3]
        marks = torch.stack((mean,asym,skew),axis=2).numpy()*sd[t,:,None,:]
        parts.append(marks-marks.mean(1,keepdims=True))
    return np.concatenate(parts)


def predict(raw,grid,p,params):
    q = p.copy()
    z = design(raw,grid,params["code"],p,params["bank"])
    for h in range(3):
        q[...,h] += (rel.relative.features(p,z[...,h])@params["w"][:,h]).astype(q.dtype)
    return q


if __name__ == "__main__":
    torch.set_num_threads(1)
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
