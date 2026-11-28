"""Fit a 150-dimensional conditional Student distribution for 50 stocks and three disjoint return intervals."""
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dense_sample_size_extrapolation_v1 as features
import experiment_dense_forward_strip_observer_v1 as strip
import prepare_dense_fixed_anchor_reward_reference_v1 as best

rel,onset,REFERENCE,read_p=strip.rel,strip.onset,best.OUT,best.read_p
OUT=REFERENCE.parent/"dense_joint_strip_likelihood_v1"
ARMS=("within_stock_interval_covariance","across_stock_factor_covariance","joint_covariance")
DEVICE=features.conditional.features.DEVICE
W=features.WIDTH
K=4


class Distribution(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode=nn.Sequential(nn.Linear(W,56),nn.LayerNorm(56),nn.SiLU(),nn.Linear(56,28),nn.SiLU())
        self.membership=nn.Linear(28,K)
        self.repr=nn.Sequential(nn.Linear(84,44),nn.LayerNorm(44),nn.SiLU(),nn.Linear(44,24),nn.SiLU())
        self.mean=nn.Linear(24,3)
        self.diag=nn.Linear(24,3)
        self.off=nn.Linear(24,3)
        self.risk=nn.Linear(24,3*K)
        self.identity=nn.Parameter(torch.zeros(50,3,K))
        nn.init.normal_(self.identity,std=.05)

    def representations(self,x):
        h=self.encode(x)
        scores=self.membership(h)
        factors=scores.softmax(1).transpose(1,2)@h
        return self.repr(torch.cat((h,h.mean(1,keepdim=True).expand_as(h),scores.softmax(-1)@factors),dim=-1))

    def forward(self,x,mode):
        h=self.representations(x)
        mu=self.mean(h)
        sd=.14+F.softplus(self.diag(h))
        cross=.55*torch.tanh(self.off(h)) if mode>=1 else torch.zeros_like(sd)
        loading=None
        if mode==2:
            loading=.45*torch.tanh(self.risk(h).reshape(len(x),50,3,K)+self.identity)
        return mu,sd,cross,loading


def whiten(columns,sd,off):
    a,b,c=columns.unbind(dim=2)
    a=a/sd[...,0,None]
    b=b/sd[...,1,None]-off[...,0,None]*a
    c=c/sd[...,2,None]-off[...,1,None]*a-off[...,2,None]*b
    return torch.stack((a,b,c),dim=2).flatten(1,2)


def likelihood(params,target):
    m,sd,off,load=params
    e=whiten((target-m)[...,None],sd,off)
    maha=e.square().sum((1,2))
    logdet=2*sd.log().sum((1,2))
    if load is not None:
        f=whiten(load,sd,off)
        chol=torch.linalg.cholesky(torch.eye(K,device=f.device)+f.transpose(1,2)@f)
        r=torch.linalg.solve_triangular(chol,f.transpose(1,2)@e,upper=False)
        maha=(maha-r.square().sum((1,2))).clamp_min(0)
        logdet=logdet+2*chol.diagonal(dim1=1,dim2=2).log().sum(1)
    return ((logdet+159*torch.log1p(maha/9))/300).mean()


def train(end,modes=(0,1,2)):
    rows=[]
    _,coef,_=strip.physical()
    units=coef[2]/coef
    for d in range(end):
        with np.load(features.OUT/f"day_{d:02d}.npz") as f:
            x=f["x"]
        grid=onset.settled.load(d,"time")
        t=np.concatenate([np.arange(a+d%6,b-60,6) for a,b in rel.fields.parent.sessions(grid)])
        assert len(t)==len(x)
        y=onset.settled.load(d,"y")[t].astype(np.float32)*units.astype(np.float32)
        y[...,1:]=np.diff(y,axis=-1)
        rows.append((x,y))
    n=min(len(x) for x,y in rows)
    x,y=[torch.as_tensor(np.stack([v[:n] for v in side]),device=DEVICE) for side in zip(*rows)]
    sx=x[:,:n:3].float().square().mean((0,1,2)).sqrt().clamp_min(.06)
    ymean=y.mean((0,1,2))
    sy=y.std((0,1,2)).clamp_min(.04)
    packed=[sx.cpu().numpy(),ymean.cpu().numpy(),sy.cpu().numpy()]
    for mode in modes:
        torch.manual_seed(456718)
        model=Distribution().to(DEVICE)
        opt=torch.optim.AdamW(model.parameters(),lr=.0011,weight_decay=.055)
        for step in range(370):
            d=torch.randint(end,(40,),device=DEVICE)
            t=torch.randint(n,(40,),device=DEVICE)
            inp=(x[d,t].float()/sx).clamp(-5,5)
            target=(y[d,t]-ymean)/sy
            params=model(inp,mode)
            # Retain direct supervision on the actual endpoints under each covariance model.
            e=((params[0]-target)*sy).cumsum(-1)/torch.sqrt(sy.square().cumsum(0))
            loss=likelihood(params,target)+.5*F.smooth_l1_loss(e,torch.zeros_like(e),beta=.9)
            if step==255:
                opt.param_groups[0]["lr"]=.00042
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        packed.extend(v.detach().cpu().numpy().ravel() for v in model.parameters())
    return np.concatenate(packed)


@torch.inference_mode()
def deltas(raw,grid,p,bank):
    sx=torch.as_tensor(bank[:W]).double()
    sy=torch.as_tensor(bank[W+3:W+6]).double()
    nets=[];offset=W+6
    for _ in range(3):
        model=Distribution().double().eval()
        for v in model.parameters():
            v.copy_(torch.from_numpy(bank[offset:offset+v.numel()].reshape(v.shape)))
            offset+=v.numel()
        nets.append(model)
    assert offset==len(bank)
    _,coef,_=strip.physical()
    conversion=torch.as_tensor(coef[1:]/coef[2])
    history=features.conditional.features.state(raw,grid)
    for i in range(0,len(raw),64):
        sl=slice(i,i+64)
        v=features.input_block(raw[sl],history[sl]).astype(np.float16)
        x=(torch.from_numpy(v).double()/sx).clamp(-5,5)
        a,b,c=[(m.mean(m.representations(x))*sy).cumsum(-1)[...,1:]*conversion for m in nets]
        yield sl,torch.stack((b-a,c-b,c-a),dim=2).numpy()


def signals(raw,grid,p,bank):
    out=np.empty((*p.shape[:2],3,2))
    for sl,delta in deltas(raw,grid,p,bank):
        z=strip.center(delta).reshape(len(delta),50,6)
        q=strip.center(p[sl].astype(float))
        qt=q.transpose(0,2,1)
        out[sl]=(z-q@np.linalg.solve(qt@q+.001*np.eye(3),qt@z)).reshape(len(z),50,3,2)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=="__main__":
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
