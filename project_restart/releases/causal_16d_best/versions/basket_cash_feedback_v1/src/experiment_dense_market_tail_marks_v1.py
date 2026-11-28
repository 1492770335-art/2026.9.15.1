"""Forecast the contribution of stock-specific tail events around an independent robust market center."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dense_state_reliable_market_v1 as reliable

basket, onset, rel, REFERENCE = reliable.basket, reliable.onset, reliable.rel, reliable.REFERENCE
OUT = REFERENCE.parent/"dense_market_tail_marks_v1"
ARMS = ("centered_tail_payoff", "signed_event_frequency", "frequency_and_severity")


class TailHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.ids = nn.Embedding(50,6)
        self.net = nn.Sequential(nn.Linear(width*2+6,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,24),nn.SiLU(),nn.Linear(24,5))

    def forward(self,x):
        market = x.mean(1,keepdim=True).expand_as(x)
        ids = self.ids.weight.expand(len(x),-1,-1)
        return self.net(torch.cat((x,market,ids),axis=-1))


def labels(x,y,grid,table):
    pars = table[reliable.cells(x,grid),np.arange(50)]
    res = np.empty_like(y)
    for side in range(2):
        src = np.arange(50)%2==side
        _, common = reliable.teacher(y,pars,src)
        res[:,~src] = y[:,~src] - common[:,None]*pars[:,~src,0]
    return (res/pars[...,1]).clip(-12,12).astype(np.float32)


def train(end):
    torch.manual_seed(193017)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = [reliable.samples(d) for d in range(4,end)]
    tables = [reliable.risk(data[1-k::2]) for k in range(2)]
    target = [labels(x,y,g,tables[i%2]) for i,(x,y,g) in enumerate(data)]
    x = torch.from_numpy(np.concatenate([d[0] for d in data])).to(device)
    z = torch.from_numpy(np.concatenate(target)).to(device)
    scale = x.square().mean((0,1)).sqrt().clamp_min(.03)
    x = (x/scale).clamp(-6,6)
    # The bulk plus the two signed excesses reconstruct the standardized residual.
    marks = torch.stack(((-z-1).clamp_min(0),(z-1).clamp_min(0)),dim=-1)
    event = (marks>0).float()
    prior = marks.sum((0,1))/event.sum((0,1)).clamp_min(1)
    model = TailHead(len(scale)).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.065)
    for _ in range(6):
        for t in torch.randperm(len(x),device=device).split(72):
            raw = model(x[t])
            excess = F.softplus(raw[...,3:])
            loss = F.mse_loss(raw[...,0],z[t].clamp(-1,1))
            loss += F.binary_cross_entropy_with_logits(raw[...,1:3],event[t])
            loss += .3*((excess-marks[t]).square()*event[t]).sum()/event[t].sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
    torch.save(dict(state=model.cpu().state_dict(),scale=scale.cpu(),prior=prior.cpu(),
                    risk=torch.from_numpy(reliable.risk(data)),rate=torch.from_numpy(basket.front.direction.table(end))),
               OUT/f"fit_{end}.pt")
    return np.array([end])


@torch.inference_mode()
def design(raw,grid,code,p,bank):
    pack = torch.load(OUT/f"fit_{int(bank[0])}.pt",map_location="cpu",weights_only=True)
    net = TailHead(len(pack["scale"])).double().eval()
    net.load_state_dict(pack["state"])
    x = basket.features(raw,grid,code,p,pack["rate"].numpy())
    pars = pack["risk"].numpy()[reliable.cells(x,grid),np.arange(50)]
    parts = []
    for t in np.array_split(np.arange(len(x)),max(1,(len(x)+95)//96)):
        logits = net((torch.from_numpy(x[t]).double()/pack["scale"]).clamp(-6,6))
        prob = logits[...,1:3].sigmoid()
        excess = F.softplus(logits[...,3:])
        frequency = prob*pack["prior"]
        severity = prob*excess
        freq = frequency[...,1]-frequency[...,0]
        tails = severity[...,1]-severity[...,0]
        z = torch.stack((logits[...,0]+tails,freq,tails),dim=-1)
        parts.append((z.numpy()*pars[t,:,1,None]).mean(1))
    return np.concatenate(parts)


def predict(raw,grid,p,params):
    q = p.copy()
    q[...,2] += (design(raw,grid,params["code"],p,params["bank"])@params["w"])[:,None].astype(q.dtype)
    return q


if __name__ == "__main__":
    torch.set_num_threads(1)
    basket.rolling(OUT,ARMS,train,design)
