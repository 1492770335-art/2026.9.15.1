"""Fit an antisymmetric comparison network to stocks with similar incumbent horizon forecasts."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_dense_hierarchical_value_net_v1 as net
from prepare_dense_settled_reference_v1 import OUT as REFERENCE

OUT=net.OUT.parent/"dense_matched_ordinal_pairs_v1"
ARMS=("win_probability","tail_difference","ordinal_value")
USES_FORECAST=True


class Comparison(nn.Module):
    def __init__(self,width,effects=3,contexts=2):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(width,32),nn.LayerNorm(32),nn.SiLU(),nn.Linear(32,16))
        self.gate=nn.Sequential(nn.Linear(32,20),nn.SiLU(),nn.Linear(20,16*effects))
        self.uncertainty=nn.Linear(32,contexts)
        self.base=nn.Parameter(torch.randn(16,effects)*.08)

    def match(self,q):
        rms=(q.square().mean(1,keepdim=True)+.2).sqrt()
        # Match contemporaneous forecasts, independently of any future payoff, and exclude the stock itself.
        distances=(q[:,:,None]/rms[:,:,None]-q[:,None,:]/rms[:,None,:]).square().sum(-1)
        distances=distances.masked_fill(torch.eye(q.shape[1],device=q.device,dtype=torch.bool),torch.inf)
        ds,peer=distances.topk(8,largest=False)
        return peer,(-.35*ds).softmax(-1)

    def forward(self,x):
        peer,weights=self.match(x[:,:,-3:])
        h=self.encoder(x)
        donor=h[torch.arange(len(h),device=x.device)[:,None,None],peer]
        diff=h[:,:,None]-donor
        pair=torch.cat(((h[:,:,None]+donor)/2,diff.abs()),-1)
        effect=(diff.unsqueeze(-1)*(self.base+.15*self.gate(pair).reshape(*pair.shape[:-1],16,self.base.shape[1]))).sum(-2)
        uncertainty=self.uncertainty(pair)
        return effect,uncertainty,peer,weights


def tail_logits(effect,neutral):
    return torch.stack((-effect,neutral,effect),-1)


def train(days):
    torch.manual_seed(84419)
    src=[];labels=[]
    for d in range(4,days):
        with np.load(net.OUT/f"day_{d:02d}.npz") as f:
            src.append(f["x"])
            labels.append(f["y"][:,:,0])
    x=torch.tensor(np.concatenate(src),device="cuda",dtype=torch.float32)
    y=torch.tensor(np.concatenate(labels),device="cuda")
    unit=x.square().mean((0,1)).sqrt().clamp_min(.04)
    x=(x/unit).clamp(-5,5)
    model=Comparison(x.shape[-1]).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.0012,weight_decay=.04)
    for epoch in range(5):
        for idx in torch.randperm(len(x),device="cuda").split(32):
            effect,unc,peer,w=model(x[idx])
            truth=y[idx]
            donor=truth[torch.arange(len(idx),device="cuda")[:,None,None],peer]
            gap=(truth[:,:,None]-donor)/(truth.square().mean(1,keepdim=True).sqrt().clamp_min(.24)[:,:,None])
            loss=F.binary_cross_entropy_with_logits(effect[...,0],(gap>0).float(),reduction="none")
            for k,barrier in enumerate((.6,1.35)):
                cls=(gap>-barrier).long()+(gap>barrier).long()
                loss+=F.cross_entropy(tail_logits(effect[...,k+1],unc[...,k]).reshape(-1,3),cls.ravel(),reduction="none").reshape_as(gap)*.6
            loss=(w*loss).sum(-1).mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
    torch.save(dict(params=model.cpu().state_dict(),unit=unit.cpu()),OUT/f"net_{days}.pt")
    return np.array([days])


@torch.inference_mode()
def views(raw,grid,code,bank,p):
    data=torch.load(OUT/f"net_{int(bank[0])}.pt",map_location="cpu",weights_only=True)
    model=Comparison(len(data["unit"])).double().eval()
    model.load_state_dict(data["params"])
    x=net.features(raw,grid,code,p)
    result=[]
    for j in range(0,len(x),32):
        u=(torch.from_numpy(x[j:j+32]).double()/data["unit"]).clamp(-5,5)
        effect,unc,_,w=model(u)
        prob=[2*torch.sigmoid(effect[...,0])-1]
        for k in (0,1):
            probs=tail_logits(effect[...,k+1],unc[...,k]).softmax(-1)
            prob.append(probs[...,2]-probs[...,0])
        estimate=torch.stack((prob[0],prob[2],.3*prob[0]+.45*prob[1]+.25*prob[2]),-1)
        result.append((w[...,None]*estimate).sum(-2).numpy())
    z=np.concatenate(result)
    return z-z.mean(1,keepdims=True)


if __name__=="__main__":
    torch.set_num_threads(1)
    net.rel.main(OUT,ARMS,train,views,with_prediction=True,rolling_reference=REFERENCE)
