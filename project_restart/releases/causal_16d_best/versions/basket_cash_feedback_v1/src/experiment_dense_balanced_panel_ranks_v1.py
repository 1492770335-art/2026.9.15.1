"""Fit a joint assignment of the 50 forecast errors to equally occupied cross-sectional rank bins."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dense_maturity_band_pairs_v1 as band

rel, REFERENCE = band.rel, band.REFERENCE
OUT = REFERENCE.parent/"dense_balanced_panel_ranks_v1"
ARMS = ("independent_ranks", "balanced_ranks", "occupancy_adjustment")
USES_FORECAST = True
BINS = 10


def assignment(logits):
    prob = logits.softmax(-1).clamp_min(1e-12)
    for _ in range(6):
        prob = prob*(prob.shape[1]/BINS)/prob.sum(1,keepdim=True)
        prob = prob/prob.sum(-1,keepdim=True)
    return prob


class PanelRank(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(width,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,24),nn.SiLU())
        self.assignment=nn.Sequential(nn.Linear(72,32),nn.SiLU(),nn.Linear(32,BINS))
        self.spacing=nn.Sequential(nn.Linear(48,24),nn.SiLU(),nn.Linear(24,BINS-1))

    def forward(self,x):
        enc=self.encoder(x)
        mean=enc.mean(1,keepdim=True)
        dev=(enc-mean).square().mean(1,keepdim=True).clamp_min(1e-6).sqrt()
        pool=torch.cat((mean,dev),-1)
        logits=self.assignment(torch.cat((enc,pool.expand(-1,x.shape[1],-1)),-1)).clamp(-12,12)
        step=F.softplus(self.spacing(pool))+.01
        curve=F.pad(step.cumsum(-1),(1,0))
        return logits,curve-curve.mean(-1,keepdim=True)


def rank_targets(y):
    # Tied outcomes share the rank bins spanned by their tied positions.
    left=(y[:,None,:]<y[:,:,None]).sum(-1)
    right=(y[:,None,:]<=y[:,:,None]).sum(-1)
    boundaries=torch.arange(BINS,device=y.device)*5
    overlap=(torch.minimum(right[:,:,None],boundaries+5)-torch.maximum(left[:,:,None],boundaries)).clamp_min(0)
    return overlap/(right-left)[:,:,None]


def train(end):
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(60813)
    data=[]
    for day in range(4,end):
        with np.load(band.OUT/f"day_{day:02d}.npz") as f:
            data.append((f["x"],f["y"][...,3],f["y"].mean(-1)))
    x,y,smooth=[torch.as_tensor(np.concatenate(col),device=device,dtype=torch.float32) for col in zip(*data)]
    unit=x.square().mean((0,1)).sqrt().clamp_min(.04)
    x=(x/unit).clamp(-5,5)
    scale=y.square().mean().sqrt().clamp_min(.02)
    y=y/scale
    ranks=.75*rank_targets(y)+.25*rank_targets(smooth)
    values=y.sort(dim=1).values.reshape(-1,BINS,5).mean(-1)
    values-=values.mean(-1,keepdim=True)
    model=PanelRank(x.shape[-1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.045)
    for _ in range(7):
        for ids in torch.randperm(len(x),device=device).split(48):
            logits,curve=model(x[ids])
            prob=assignment(logits)
            balanced=-(ranks[ids]*prob.clamp_min(1e-12).log()).sum(-1).mean()
            independent=-(ranks[ids]*logits.log_softmax(-1)).sum(-1).mean()
            spacing=F.smooth_l1_loss(curve[:,0],values[ids],beta=.6)
            estimate=(prob*curve).sum(-1)
            mse=F.smooth_l1_loss(estimate,y[ids],beta=1.2)
            loss=.8*balanced+.2*independent+.32*spacing+.25*mse
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
    torch.save(dict(unit=unit.cpu(),state=model.cpu().state_dict()),OUT/f"net_{end}.pt")
    return np.array([end])


@torch.inference_mode()
def views(raw,grid,code,bank,p):
    params=torch.load(OUT/f"net_{int(bank[0])}.pt",map_location="cpu",weights_only=True)
    model=PanelRank(len(params["unit"])).double().eval()
    model.load_state_dict(params["state"])
    x=band.pairs.net.features(raw,grid,code,p)
    parts=[]
    for start in range(0,len(x),64):
        u=(torch.as_tensor(x[start:start+64]).double()/params["unit"]).clamp(-5,5)
        logits,curve=model(u)
        prob=assignment(logits)
        ind=(logits.softmax(-1)*curve).sum(-1)
        bal=(prob*curve).sum(-1)
        parts.append(torch.stack((ind,bal,bal-ind),-1).numpy())
    z=np.concatenate(parts)
    return z-z.mean(1,keepdims=True)


if __name__=="__main__":
    torch.set_num_threads(1)
    rel.main(OUT,ARMS,train,views,with_prediction=True,rolling_reference=REFERENCE)
