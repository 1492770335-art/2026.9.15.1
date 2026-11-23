"""Train a sequence model with damped temporal modes and peer messages between its causal temporal layers."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_execution_location_markouts_v1 as location
import experiment_execution_volume_clock_v1 as clock
import experiment_crossfit_predictable_tails_v1 as valid
from experiment_causal_cross_horizon_ridge_v1 import ART,labels

OUT=ART/'damped_market_state_v1'
WIDTH=32
WARM=96


def geometry():
    with np.load(location.OUT/'model.npz') as st:return {k:st[k].copy() for k in st.files if k not in ('market','relative')}


def frames(raw,g):
    r,q,sp,curr,prev,s,rest,units,excess,vol=location.values(raw,g)
    t=np.arange(len(r));sig=(.45+.55*np.sqrt((8+clock.flow.trailing(r*r,96))/(8+np.minimum(t+1,96))[:,None])).clip(.55,3)
    rawf=np.arcsinh(raw[:,:,:83].astype('float32')).clip(-7,7)
    z=np.stack((np.log(q),np.log1p(sp),np.tanh(curr),np.tanh(prev),s,np.arcsinh(rest),units,
                np.arcsinh(excess),np.log(sig),np.broadcast_to(np.exp(-t[:,None]/360),r.shape)),2)
    return np.concatenate((rawf,raw[:,:,83:].astype('float32')-1,z),2).astype('float16'),sig.astype('float32')


class DampedLayer(nn.Module):
    def __init__(self,width,length):
        super().__init__();self.length=length
        self.norm=nn.LayerNorm(width);self.inp=nn.Linear(width,width*2)
        self.tau=nn.Parameter(torch.linspace(-1,3,width))
        self.freq=nn.Parameter(torch.linspace(-4,0,width))
        self.out=nn.Linear(width,width)
        self.register_buffer('lag',torch.arange(length,dtype=torch.float32))
        self.msg=nn.Linear(width,12);self.mix=nn.Linear(24,width)
        self.hub=nn.Parameter(torch.randn(4,50)*.08)
        self.gate=nn.Linear(width,width)

    def forward(self,x):
        batch,time,stocks,width=x.shape;z=self.norm(x)
        y,g=self.inp(z).chunk(2,dim=-1)
        tau=1+45*torch.sigmoid(self.tau)
        kernel=torch.exp(-self.lag[None]/tau[:,None])*torch.cos(self.lag[None]*F.softplus(self.freq)[:,None])
        kernel=kernel/(.25+kernel.abs().sum(1,keepdim=True))
        y=y.permute(0,2,3,1).reshape(batch*stocks,width,time)
        y=F.conv1d(F.pad(y,(self.length-1,0)),kernel.flip(1)[:,None],groups=width)
        y=y.reshape(batch,stocks,width,time).permute(0,3,1,2)
        x=x+self.out(y*F.silu(g))
        msg=self.msg(x);mean=msg.mean(2,keepdim=True)
        weights=(self.hub*3).softmax(dim=1)
        portfolios=weights@(msg-mean)
        local=(self.hub.T*3).softmax(dim=1)@portfolios
        return x+self.mix(torch.cat((local,mean.expand_as(local)),3))*self.gate(z).sigmoid()


class SeqNet(nn.Module):
    def __init__(self,n):
        super().__init__();self.input=nn.Sequential(nn.Linear(n,WIDTH),nn.LayerNorm(WIDTH),nn.SiLU())
        self.stock=nn.Parameter(torch.zeros(50,WIDTH))
        self.temporal=nn.Sequential(DampedLayer(WIDTH,33),DampedLayer(WIDTH,65))
        self.head=nn.Sequential(nn.LayerNorm(WIDTH),nn.Linear(WIDTH,48),nn.SiLU(),nn.Dropout(.10),nn.Linear(48,6))

    def forward(self,x,v):
        h=self.temporal(self.input(x)+self.stock)
        z=self.head(h).float();a,b,c=z[...,:3].unbind(-1)
        p=torch.stack((a,.272/.559*a+b,.161/.559*a+.161/.272*b+c),-1)*v.unsqueeze(-1)
        return p,z[...,3:]


def train(end):
    torch.manual_seed(7439);rng=np.random.default_rng(7439)
    x=[];vv=[];yy=[];future=[]
    g=geometry()
    for d in range(end):
        raw=clock.pair.rawday(d);z,v=frames(raw,g)
        x.append(torch.from_numpy(z).to('cuda'));vv.append(torch.from_numpy(v).to('cuda'))
        yy.append(torch.from_numpy(labels(d).reshape(-1,50,3)).to('cuda'))
        signed=location.values(raw,g)[5];cs=np.vstack((np.zeros(50),signed.cumsum(0)))
        t=np.arange(len(raw))+1
        aux=np.stack([(cs[(t+b).clip(max=len(raw))]-cs[(t+a).clip(max=len(raw))])/(b-a) for a,b in ((0,5),(5,20),(20,60))],axis=2)
        future.append(torch.from_numpy(aux.astype('float16')).to('cuda'))
    net=SeqNet(x[0].shape[-1]).cuda();ema=SeqNet(x[0].shape[-1]).cuda();ema.load_state_dict(net.state_dict())
    optim=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.035)
    span=144
    for ep in range(10):
        for step in range(120):
            days=rng.integers(end,size=3)
            start=[int(rng.integers(0,(min(720,len(x[d])-span-61) if step%4==0 else len(x[d])-span-61))) for d in days]
            def batch(arr):return torch.stack([arr[d][s:s+span] for d,s in zip(days,start)])
            v=batch(vv);real=batch(yy)[:,WARM:];a=batch(future)[:,WARM:].float()
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                p,flow=net(batch(x).float(),v)
            p=p[:,WARM:];flow=flow[:,WARM:].float();v=v[:,WARM:]
            sh=p.shape;flat=lambda z:z.reshape(-1,*z.shape[2:])
            loss=valid.criterion(flat(p),flat(real),flat(v))+.36*F.smooth_l1_loss(flow,a,beta=.25)
            loss-=.075*F.cosine_similarity(p-p.mean(2,keepdim=True),real-real.mean(2,keepdim=True),dim=2,eps=1e-5).mean()
            optim.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);optim.step()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.015)
        if ep in (4,7):
            for group in optim.param_groups:group['lr']*=.5
        if ep in (0,4,9):print('damped temporal model',end,ep+1,flush=True)
    torch.save(ema.state_dict(),OUT/f'net_{end}.pth');return ema.eval()


def restore(st):return SeqNet(st['input.0.weight'].shape[1])


@torch.inference_mode()
def infer(net,raw):
    z,v=frames(raw,geometry());out=np.empty((len(z),50,3),np.float32)
    device=next(net.parameters()).device
    for t in range(0,len(raw),128):
        a=max(t-WARM,0);b=min(len(raw),t+128)
        xx=torch.from_numpy(z[a:b]).to(device).float()[None]
        scale=torch.from_numpy(v[a:b]).to(device)[None]
        out[t:b]=net(xx,scale)[0][0,t-a:].cpu().numpy()
    return out


if __name__=='__main__':valid.run_validation(OUT,train,restore,infer)
