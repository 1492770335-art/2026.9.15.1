"""Learn the direction of a stock's return rank separately from how extreme that rank will be."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dense_balanced_panel_ranks_v1 as ranks

rel, REFERENCE=ranks.rel,ranks.REFERENCE
OUT=REFERENCE.parent/"dense_rank_direction_value_v1"
ARMS=("direction_only","radius_specific_direction","conditional_direction_increment")
USES_FORECAST=True
WIDTH=ranks.BINS//2


def encoder(n,out):
    return nn.Sequential(nn.Linear(n,40),nn.LayerNorm(40),nn.SiLU(),nn.Linear(40,out))


class DirectionValue(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.unsigned=encoder(width,WIDTH)
        self.side=encoder(width,WIDTH+1)

    def forward(self,x):
        mass=self.unsigned(x).softmax(-1)
        # Every extremity band contains the same number of names across its two signs.
        for _ in range(5):
            mass=mass*(x.shape[1]/WIDTH)/mass.sum(1,keepdim=True)
            mass=mass/mass.sum(-1,keepdim=True)
        sign=self.side(x)
        whole=sign[:,:,:1]
        return mass,whole,whole+.65*sign[:,:,1:]


def upper_probability(mass, logits):
    # Each radius bin has equal numbers of stocks above and below the median.
    offset = torch.zeros_like(mass[:, :1])
    for _ in range(4):
        probability = (logits + offset).sigmoid()
        imbalance = (mass * (probability - .5)).sum(1, keepdim=True)
        derivative = (mass * probability * (1 - probability)).sum(1, keepdim=True)
        offset -= (imbalance / derivative.clamp_min(.01)).clamp(-2, 2)
    return (logits + offset).sigmoid()


def train(end):
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(71205)
    rows=[]
    for d in range(4,end):
        with np.load(ranks.band.OUT/f"day_{d:02d}.npz") as f:
            rows.append((f["x"],f["y"][...,3],f["y"].mean(-1)))
    x,y,smooth=[torch.as_tensor(np.concatenate(col),device=device,dtype=torch.float32) for col in zip(*rows)]
    unit=x.square().mean((0,1)).sqrt().clamp_min(.04)
    x=(x/unit).clamp(-5,5)
    y=y/y.square().mean().sqrt().clamp_min(.02)
    targets=ranks.rank_targets(y)
    levels=(targets*y[...,None]).sum((0,1))/targets.sum((0,1))
    weights=(levels[WIDTH:].flip(0)-levels[:WIDTH])*.5
    target=.8*targets+.2*ranks.rank_targets(smooth)
    pos=target[:,:,WIDTH:].flip(-1)
    neg=target[:,:,:WIDTH]
    radius=pos+neg
    model=DirectionValue(x.shape[-1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0014,weight_decay=.04)
    for _ in range(6):
        for ids in torch.randperm(len(x),device=device).split(48):
            mass,whole,logit=model(x[ids])
            def signed_cost(z):
                return (pos[ids]*F.softplus(-z)+neg[ids]*F.softplus(z))
            # The two encoders have disjoint parameters. Give greater directional weight to larger payoffs.
            loss=((signed_cost(logit)+.45*signed_cost(whole))*weights).sum(-1).mean()
            loss-=.55*(radius[ids]*mass.clamp_min(1e-10).log()).sum(-1).mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
    torch.save(dict(unit=unit.cpu(),state=model.cpu().state_dict(),levels=levels.cpu()),OUT/f"net_{end}.pt")
    return np.array([end])


@torch.inference_mode()
def views(raw,grid,code,bank,p):
    pack=torch.load(OUT/f"net_{int(bank[0])}.pt",map_location="cpu",weights_only=True)
    model=DirectionValue(len(pack["unit"])).double().eval()
    model.load_state_dict(pack["state"])
    levels=pack["levels"]
    up,down=levels[WIDTH:].flip(0),levels[:WIDTH]
    x=ranks.band.pairs.net.features(raw,grid,code,p)
    result=[]
    for j in range(0,len(x),64):
        u=(torch.as_tensor(x[j:j+64]).double()/pack["unit"]).clamp(-5,5)
        mass,common,local=model(u)
        def expected(logit):
            return (mass*(down+(up-down)*upper_probability(mass, logit))).sum(-1)
        one=expected(common)
        many=expected(local)
        result.append(torch.stack((one,many,many-one),-1).numpy())
    z=np.concatenate(result)
    return z-z.mean(1,keepdims=True)


if __name__=="__main__":
    torch.set_num_threads(1)
    rel.main(OUT,ARMS,train,views,with_prediction=True,rolling_reference=REFERENCE)
