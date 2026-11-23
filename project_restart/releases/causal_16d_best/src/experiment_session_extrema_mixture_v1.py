"""Forecast horizon returns conditional on whether the future price breaks today's observed high or low."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_execution_volume_clock_v1 as clock
import experiment_crossfit_predictable_tails_v1 as val
from experiment_causal_cross_horizon_ridge_v1 import ART,labels

OUT=ART/'session_extrema_mixture_v1'
H=(5,20,60)
S=np.array([.559,.272,.161],np.float32)


def geometry():
    slopes=[];hi=[];lo=[]
    for d in range(4):
        raw=clock.pair.rawday(d);r=raw[:,:,0].astype(float)-.00136;p=r.cumsum(0)-r[0]
        high=np.maximum.accumulate(p,0);low=np.minimum.accumulate(p,0)
        hi.append(np.median(raw[:,:,3][(p==high)&(high>4)]));lo.append(np.median(raw[:,:,4][(p==low)&(low<-4)]))
        k=(abs(r[1:])>.4)&(abs(r[1:])<4)
        delta=np.diff(raw[:,:,3:5].astype(float),axis=0)
        slopes.append(np.median(delta[k]/r[1:][k,None],axis=0))
    return np.r_[np.median(hi),np.median(lo),np.median(slopes,0)].astype('float32')


def path(raw,g):
    r=raw[:,:,0].astype(float)-.00136;p=r.cumsum(0)-r[0]
    hi=(g[0]-raw[0,:,3].astype(float))/g[2];lo=(g[1]-raw[0,:,4].astype(float))/g[3]
    # At the first observable frame the supplied features already contain information about the earlier range.
    top=np.maximum.accumulate(np.maximum(p,hi),axis=0);bot=np.minimum.accumulate(np.minimum(p,lo),axis=0)
    return p,top,bot


def features(raw,g):
    s,v=clock.pair.state(raw);p,top,bot=path(raw,g);n=len(raw);t=np.arange(n)
    v=(.45+.55*v).clip(.55,3).astype('float32')
    u=(top-p)/v;d=(p-bot)/v
    age_top=t[:,None]-np.maximum.accumulate(np.where(u<.5,t[:,None],0),0)
    age_bot=t[:,None]-np.maximum.accumulate(np.where(d<.5,t[:,None],0),0)
    common=(top-bot)/(v+.001)
    z=np.stack((np.log1p(u),np.log1p(d),np.log1p(common),(d-u)/(d+u+.5),np.log1p(age_top)/4,
                np.log1p(age_bot)/4,np.arcsinh((p-p[:1])/np.sqrt(t[:,None]+12)),np.tanh(raw[:,:,5].astype(float)),
                np.broadcast_to(np.exp(-t[:,None]/360),u.shape),raw[:,:,1].astype(float),raw[:,:,2].astype(float)),2)
    pieces=[s,z]
    for h in H:
        threshold=np.sqrt(h);near_u=np.exp(-u/threshold);near_d=np.exp(-d/threshold)
        prev=(t-h).clip(0)
        add=np.stack((near_u,near_d,(top-top[prev])/(threshold*v),(bot-bot[prev])/(threshold*v),
                      (near_u-near_d)*np.tanh(raw[:,:,21].astype(float)),(near_u+near_d)*np.tanh(raw[:,:,22].astype(float))),2)
        pieces.append(add)
    return np.concatenate(pieces,2).astype('float16'),v,(p,top,bot)


def breakout(path,t,v):
    p,top,bot=path;classes=[];extensions=[]
    for h in H:
        a=top[t+h]-top[t];b=bot[t]-bot[t+h]
        above=a>.25*v[t];below=b>.25*v[t]
        classes.append(above.astype(np.int64)+2*below.astype(np.int64))
        extensions.append(np.stack((np.arcsinh(a/(np.sqrt(h)*v[t])),np.arcsinh(b/(np.sqrt(h)*v[t]))),2))
    return np.stack(classes,2),np.stack(extensions,2).astype('float32')


class Net(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(n,110),nn.LayerNorm(110),nn.SiLU(),nn.Dropout(.12),nn.Linear(110,64))
        self.stock=nn.Parameter(torch.zeros(50,64));self.proj=nn.Linear(64,20);self.market=nn.Parameter(torch.randn(5,50)*.10)
        self.latent=nn.Sequential(nn.Linear(104,84),nn.LayerNorm(84),nn.SiLU(),nn.Dropout(.12))
        self.head=nn.Linear(84,3*10);self.direct=nn.Linear(84,3)
        self.register_buffer('geom',torch.zeros(4))

    def forward(self,x,v):
        h=self.enc(x)+self.stock;msg=self.proj(h);avg=msg.mean(1,keepdim=True)
        factors=(self.market*3).softmax(1)@(msg-avg);peer=(self.market.T*3).softmax(1)@factors
        z=self.latent(torch.cat((h,peer,avg.expand_as(msg)),2))
        raw=self.head(z).float().reshape(len(x),50,3,10);logits=raw[...,:4]
        centers=raw[...,4:8]*v[:,:,None,None];amount=F.softplus(raw[...,8:])
        mean=(centers*logits.softmax(-1)).sum(-1);direct=self.direct(z).float()*v[:,:,None]
        return .7*mean+.3*direct,logits,centers,amount


def fit(end):
    torch.manual_seed(4448);g=geometry();ts=[np.r_[np.arange(d%2,600,2),np.arange(600+d%6,len(clock.pair.rawday(d))-61,6)] for d in range(end)]
    n=sum(map(len,ts));idx=0
    v=torch.empty(n,50,device='cuda');y=torch.empty(n,50,3,device='cuda')
    classes=torch.empty(n,50,3,device='cuda',dtype=torch.uint8);ext=torch.empty(n,50,3,2,device='cuda',dtype=torch.float16)
    for d,t in enumerate(ts):
        z,vr,pa=features(clock.pair.rawday(d),g)
        if d==0:x=torch.empty(n,50,z.shape[-1],device='cuda',dtype=torch.float16)
        c,e=breakout(pa,t,vr);sl=slice(idx,idx+len(t));idx+=len(t)
        for dst,a in ((x,z[t]),(v,vr[t]),(y,labels(d).reshape(-1,50,3)[t]),(classes,c),(ext,e)):
            dst[sl].copy_(torch.from_numpy(a))
    net=Net(x.shape[-1]).cuda();net.geom.copy_(torch.from_numpy(g));ema=Net(x.shape[-1]).cuda();ema.load_state_dict(net.state_dict())
    opt=torch.optim.AdamW(net.parameters(),lr=.00075,weight_decay=.028)
    for ep in range(16):
        for ids in torch.randperm(n,device='cuda').split(48):
            yy=y[ids];vv=v[ids];cc=classes[ids].long()
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):p,logits,means,amount=net(x[ids].float(),vv)
            chosen=means.gather(-1,cc.unsqueeze(-1)).squeeze(-1)
            loss=.95*val.criterion(p,yy,vv)+.36*F.smooth_l1_loss(chosen/vv[:,:,None],yy/vv[:,:,None],beta=1.2)
            loss+=.23*F.cross_entropy(logits.flatten(0,2),cc.flatten())+.28*F.smooth_l1_loss(amount,ext[ids].float(),beta=.35)
            loss-=.05*F.cosine_similarity(p-p.mean(1,keepdim=True),yy-yy.mean(1,keepdim=True),dim=1,eps=1e-5).mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.016)
        if ep in (5,10):
            for group in opt.param_groups:group['lr']*=.53
        if ep in (0,7,15):print('range break mixture',end,ep+1,flush=True)
    torch.save(ema.state_dict(),OUT/f'net_{end}.pth');return ema.eval()


def restore(st):return Net(st['enc.0.weight'].shape[1])


@torch.inference_mode()
def infer(net,raw):
    z,v,_=features(raw,net.geom.cpu().numpy());out=np.empty((len(v),50,3),np.float32);dev=next(net.parameters()).device
    for i in range(0,len(raw),48):
        sl=slice(i,i+48)
        out[sl]=net(torch.from_numpy(z[sl]).to(dev).float(),torch.from_numpy(v[sl]).to(dev))[0].cpu().numpy()
    return out


if __name__=='__main__':val.run_validation(OUT,fit,restore,infer)
