"""Estimate sensitivity to the number of training dates using two balanced delete-half neural ensembles."""
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dense_conditional_forecast_density_v1 as assessment
import experiment_dense_conditional_rank_error_net_v1 as conditional

base,REFERENCE,rel,onset=assessment.base,assessment.REFERENCE,assessment.rel,assessment.onset
OUT=REFERENCE.parent/"dense_sample_size_extrapolation_v1"
ARMS=("full_minus_balanced_halves","agreement_of_two_partitions","increment_above_inter_fit_noise")
WIDTH=conditional.WIDTH-3


def input_block(x,history):
    book=np.tanh(.55*np.arcsinh(x[:,:,:83].astype(np.float32)))
    cats=x[:,:,83:]
    return np.concatenate((book,cats==0,cats==2,history),axis=-1)


def collect():
    OUT.mkdir(exist_ok=True)
    for d in range(16):
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        t=np.concatenate([np.arange(a+d%6,b-60,6) for a,b in rel.fields.parent.sessions(grid)])
        x=input_block(raw[t],conditional.features.state(raw,grid)[t])
        u=y[t].astype(np.float32)
        u-=u.mean(1,keepdims=True)
        np.savez(OUT/f"day_{d:02d}.npz",x=x.astype(np.float16),y=u.astype(np.float16))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Sequential(nn.Linear(WIDTH,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,24),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(48,32),nn.SiLU(),nn.Linear(32,3))

    def forward(self,x):
        h=self.hidden(x)
        z=self.head(torch.cat((h,h-h.mean(1,keepdim=True)),axis=-1))
        return z-z.mean(1,keepdim=True)


def train(end):
    data=[]
    for d in range(end):
        with np.load(OUT/f"day_{d:02d}.npz") as f:
            data.append((f["x"],f["y"]))
    # Equal dates and the same number of optimization updates in each fit.
    n=min(len(x) for x,_ in data)
    x,y=[torch.as_tensor(np.stack([r[:n] for r in col]),device=conditional.features.DEVICE).float() for col in zip(*data)]
    groups=np.arange(end)
    # Fixed feature scale from the first two historical days is shared across all training fits.
    unit=x[:2].square().mean((0,1,2)).sqrt().clamp_min(.06)
    x=(x/unit).clamp(-5,5)
    packed=[unit.cpu().numpy()]
    for ids in (groups,groups[::2],groups[1::2],groups[(groups//2)%2==0],groups[(groups//2)%2==1]):
        torch.manual_seed(87503)
        days=torch.as_tensor(ids,device=x.device)
        net=Net().to(x.device)
        opt=torch.optim.AdamW(net.parameters(),lr=.0011,weight_decay=.055)
        for step in range(360):
            d=days[torch.randint(len(days),(40,),device=x.device)]
            t=torch.randint(n,(40,),device=x.device)
            p=net(x[d,t])
            loss=F.smooth_l1_loss(p,y[d,t],beta=.65)
            if step==245:
                opt.param_groups[0]["lr"]*=.4
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
        packed.extend(w.detach().cpu().numpy().ravel() for w in net.parameters())
    return np.concatenate(packed)


@torch.inference_mode()
def signals(raw,grid,p,bank):
    unit=torch.as_tensor(bank[:WIDTH]).double()
    off=WIDTH
    nets=[]
    for _ in range(5):
        net=Net().double().eval()
        for v in net.parameters():
            v.copy_(torch.as_tensor(bank[off:off+v.numel()].reshape(v.shape))); off+=v.numel()
        nets.append(net)
    assert off==len(bank)
    hist=conditional.features.state(raw,grid)
    parts=[]
    for start in range(0,len(raw),64):
        sl=slice(start,start+64)
        x=torch.from_numpy(input_block(raw[sl],hist[sl])).double()
        x=(x/unit).clamp(-5,5)
        f,a,b,c,d=[net(x)[:,:,1:] for net in nets]
        d1=f-(a+b)/2
        d2=f-(c+d)/2
        gap=(d1+d2)/2
        noise=(abs(a-b)+abs(c-d))/4
        stable=conditional.agree(d1,d2)
        beyond=stable.sign()*(abs(stable)-noise/2).clamp_min(0)
        z=torch.stack((gap,stable,beyond),axis=2)
        # Project the implied correction away from the existing forecast span in each frame.
        q=torch.as_tensor(p[sl]).double()
        q=q-q.mean(1,keepdim=True)
        z=z.reshape(len(x),50,6)
        g=q.transpose(1,2)@q+1e-5*torch.eye(3,dtype=q.dtype)
        z=z-z.mean(1,keepdim=True)-q@torch.linalg.solve(g,q.transpose(1,2)@z)
        parts.append(z.reshape(len(x),50,3,2).numpy())
    return np.concatenate(parts)


def predict(raw,grid,p,params):
    return assessment.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=="__main__":
    torch.set_num_threads(1)
    collect()
    assessment.rolling(sys.modules[__name__])
