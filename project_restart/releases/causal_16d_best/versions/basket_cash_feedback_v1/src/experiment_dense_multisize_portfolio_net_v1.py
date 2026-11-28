"""Learn a shared return law across physical portfolio sizes, with an explicit aggregation consistency loss."""
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dense_cross_section_transport_v1 as base

REFERENCE, rel, onset = base.REFERENCE, base.rel, base.onset
OUT = REFERENCE.parent / "dense_multisize_portfolio_net_v1"
ARMS = ("direct_market_network", "multi_size_network", "increment_from_multi_size_training")
CHANNELS = 6
WIDTH = 3*CHANNELS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def state(raw,grid):
    out=np.empty((*raw.shape[:2],WIDTH),np.float32)
    for a,b in rel.fields.parent.sessions(grid):
        x=raw[a:b]
        def value(i):
            return np.tanh(.65*np.arcsinh(x[:,:,i].astype(np.float32)))
        book=np.stack((value(41)-value(42),value(54)-value(55),value(26)-value(38),
                       value(11),value(0),value(64)),axis=2)
        fast,mid,slow=[base.panel.smooth(book,n).astype(np.float32) for n in (3,18,90)]
        out[a:b]=np.concatenate((fast,mid-fast,slow-mid),axis=-1)
    return out


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(2*WIDTH+1,48),nn.LayerNorm(48),nn.SiLU(),
                                  nn.Linear(48,24),nn.SiLU(),nn.Linear(24,3))

    def forward(self,mean,variance,size):
        spread=variance.clamp_min(1e-6).sqrt()
        context=torch.cat((spread,mean.new_full((len(mean),1),size**-.5)),dim=1)
        # The response of this head changes sign under a joint sign reversal of all six inputs.
        return .5*(self.layers(torch.cat((mean,context),1))-self.layers(torch.cat((-mean,context),1)))


def collect():
    OUT.mkdir(exist_ok=True)
    ratio=base.panel.ratio()
    for d in range(16):
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        t=np.concatenate([np.arange(a+d%5,b-60,5) for a,b in rel.fields.parent.sessions(grid)])
        x=state(raw,grid)[t]
        u=y[t].astype(np.float32)
        target=np.stack((u[...,1],u[...,2],u[...,2]-ratio*u[...,1]),axis=-1)
        np.savez(OUT/f"day_{d:02d}.npz",x=x.astype(np.float16),y=target.astype(np.float16))


def fit_networks(end, days=None, modes=(False,True), network_type=Net):
    records=[]
    for d in (range(end) if days is None else days):
        with np.load(OUT/f"day_{d:02d}.npz") as f:
            records.append((f["x"],f["y"]))
    x,y=[torch.from_numpy(np.concatenate(c)).float().to(DEVICE) for c in zip(*records)]
    scale=x.square().mean((0,1)).sqrt().clamp_min(.025)
    x=(x/scale).clamp(-6,6)
    models=[]
    for auxiliary in modes:
        torch.manual_seed(61091)
        net=network_type().to(DEVICE)
        opt=torch.optim.AdamW(net.parameters(),lr=.0013,weight_decay=.04)
        for step in range(400):
            t=torch.randint(len(x),(64,),device=DEVICE)
            xx,yy=x[t],y[t]
            m=xx.mean(1)
            sq=xx.square().mean(1)
            full=net(m,sq-m.square(),50)
            loss=F.smooth_l1_loss(full,yy.mean(1),beta=.45)
            perm=torch.rand((len(t),50),device=DEVICE).argsort(1)
            size=(5,10,25)[step%3]
            if auxiliary:
                ix=perm[:,:size]
                xx1=xx.gather(1,ix[:,:,None].expand(-1,-1,xx.shape[-1]))
                m1=xx1.mean(1)
                sq1=xx1.square().mean(1)
                target1=yy.gather(1,ix[:,:,None].expand(-1,-1,3)).mean(1)
                m2=(50*m-size*m1)/(50-size)
                sq2=(50*sq-size*sq1)/(50-size)
                target2=(50*yy.mean(1)-size*target1)/(50-size)
                a=net(m1,sq1-m1.square(),size)
                b=net(m2,sq2-m2.square(),50-size)
                loss+=.65*F.smooth_l1_loss(a-b,target1-target2,beta=.55)
                loss+=.6*F.mse_loss((size*a+(50-size)*b)/50,full)
            if step==250:
                opt.param_groups[0]["lr"]*=.35
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
        models.append(net.cpu().eval())
    return models,scale.cpu().numpy()


@torch.inference_mode()
def read(x, models, scale):
    preds=[]
    for start in range(0,len(x),128):
        v=(torch.as_tensor(x[start:start+128]).float()/torch.as_tensor(scale).float()).clamp(-6,6)
        m=v.mean(1)
        var=v.square().mean(1)-m.square()
        z=[net(m,var,50).numpy() for net in models]
        preds.append(np.column_stack((z[0][:,1],z[1][:,1],z[1][:,1]-z[0][:,1])))
    return np.concatenate(preds).astype(float)


def controls(f):
    return np.column_stack((np.ones(len(f)),f[...,1:].mean(1,dtype=float)))


def train(end):
    models,scale=fit_networks(end)
    gram=0.
    for d in range(4,end):
        grid=onset.settled.load(d,"time")
        t=np.concatenate([np.arange(a+d%5,b-60,5) for a,b in rel.fields.parent.sessions(grid)])
        with np.load(OUT/f"day_{d:02d}.npz") as f:
            z=read(f["x"].astype(float),models,scale)
        c=controls(base.read_p(d)[t])
        g=np.column_stack((c,z))
        gram+=g.T@g
    beta=np.linalg.lstsq(gram[:3,:3],gram[:3,3:],rcond=1e-10)[0]
    return np.concatenate([scale,beta.ravel(),*[v.detach().numpy().ravel() for m in models for v in m.parameters()]])


def signals(raw,grid,p,bank):
    scale=bank[:WIDTH].astype(np.float32)
    beta=bank[WIDTH:WIDTH+9].reshape(3,3)
    off=WIDTH+9
    models=[Net().eval(),Net().eval()]
    with torch.no_grad():
        for m in models:
            for v in m.parameters():
                v.copy_(torch.as_tensor(bank[off:off+v.numel()].reshape(v.shape)))
                off+=v.numel()
    assert off==len(bank)
    z=read(state(raw,grid),models,scale)-controls(p)@beta
    return np.broadcast_to(z[:,None,:],(*p.shape[:2],len(ARMS)))


def predict(raw,grid,p,params):
    q=p.copy()
    z=signals(raw,grid,p,params["bank"])
    q[...,2]+=(params["strength"]*z[...,int(params["choice"])]).astype(q.dtype)
    return q


if __name__=="__main__":
    torch.set_num_threads(1)
    collect()
    base.rolling(sys.modules[__name__])
