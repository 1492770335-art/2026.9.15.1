"""Streamable multiscale order-book network with physical return-increment supervision."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
import cache_opening_walkforward as cache

prior=cache.opening.prior
OUT=prior.ART/'multiscale_return_net_v1'
WIDTHS=(4,16,64,256)
SCALE=np.array([.559,.272,.161],dtype=np.float32)
EPOCHS=(1,2,4,7,10)


def grid(day):
    return np.load(prior.CACHE/f'day_{day:02d}_x.npy',mmap_mode='r')


def features(x):
    # Several adjacent lags remain distinct. Longer causal windows carry price
    # displacement, return variation and slowly changing order-book pressure.
    z=np.asarray(x,dtype=np.float32)
    signed=np.arcsinh(z[:,:,:83])
    discrete=z[:,:,83:]-1.
    inputs=np.concatenate((signed,discrete),axis=2)
    tick=np.arange(len(z))
    recent=[inputs,inputs[np.maximum(0,tick-1)],inputs[np.maximum(0,tick-3)]]
    total=np.concatenate((np.zeros_like(inputs[:1],dtype=float),inputs.cumsum(0,dtype=float)))
    ret=z[:,:,0].astype(float)
    price=np.vstack((np.zeros((1,50)),ret.cumsum(0)))
    rv=np.vstack((np.zeros((1,50)),(ret**2).cumsum(0)))
    end=tick+1
    extras=[]
    for k in WIDTHS:
        start=np.maximum(0,end-k);size=(end-start)[:,None]
        avg=(total[end]-total[start])/size[:,:,None]
        recent.append(avg.astype(np.float32))
        r=(price[end]-price[start])/np.sqrt(size)
        vol=np.sqrt((rv[end]-rv[start]+4.)/(size+4.))
        extras.extend((np.arcsinh(r),np.log(vol+.05),r/(.4+vol)))
    time=np.exp(-tick/256)
    extras.append(np.broadcast_to(time[:,None],ret.shape))
    recent.append(np.stack(extras,axis=-1).astype(np.float32))
    return np.concatenate(recent,axis=2).astype(np.float16)


class ReturnNet(nn.Module):
    def __init__(self,n):
        super().__init__()
        h=96
        self.numeric=nn.Sequential(nn.Linear(n,h),nn.LayerNorm(h),nn.SiLU(),nn.Dropout(.12))
        self.proj=nn.Linear(h,48)
        self.local=nn.Linear(h,h)
        self.context=nn.Linear(96,h,bias=False)
        self.market=nn.Sequential(nn.Linear(96,64),nn.SiLU(),nn.Linear(64,4))
        self.tail=nn.Sequential(nn.LayerNorm(h),nn.SiLU(),nn.Dropout(.15),nn.Linear(h,64),nn.SiLU(),nn.Linear(64,4))

    def forward(self,x):
        x=self.numeric(x)
        msg=self.proj(x)
        context=torch.cat((msg.mean(1),msg.std(1,unbiased=False)),dim=-1)
        local=self.tail(self.local(x)+self.context(context)[:,None,:])
        return local-local.mean(1,keepdim=True)+self.market(context)[:,None,:]


def targets(y,x):
    # Four disjoint segments: t+1, +2..5, +6..20, +21..60, whose sums recover the labels.
    raw=y/SCALE
    next_r=np.zeros_like(raw[:,:,0]);next_r[:-1]=x[1:,:,0]
    return np.stack((next_r,(raw[:,:,0]-next_r)/2.,
        (raw[:,:,1]-raw[:,:,0])/np.sqrt(15.),(raw[:,:,2]-raw[:,:,1])/np.sqrt(40.)),axis=2).astype(np.float32)


def decode(v):
    totals=torch.cumsum(v*v.new_tensor([1.,2.,math.sqrt(15),math.sqrt(40)]),dim=-1)
    return totals[:,:,1:]*v.new_tensor(SCALE)


def train(end, epochs):
    torch.manual_seed(771)
    # Feature cache is generated one day at a time; validation dates are loaded
    # only by prediction after fitting. Interlaced fixed stride 3 limits memory.
    xs=[];ys=[];segs=[];vols=[]
    for d in range(end):
        x=grid(d);y=prior.labels(d).reshape(-1,50,3);sel=slice(d%3,-1,3)
        feat=np.load(OUT/f'features_{d}.npy',mmap_mode='r')
        xs.append(torch.from_numpy(np.array(feat[sel])).cuda())
        ys.append(torch.from_numpy(np.array(y[sel])).cuda())
        segs.append(torch.from_numpy(targets(y,x)[sel].copy()).cuda())
        # causal 64-frame per-stock realized volatility (third from this window's start)
        vols.append(torch.from_numpy(np.exp(feat[sel,:,-6].astype(np.float32))-.05).cuda())
    device_data=[]
    for parts in (xs,ys,segs,vols):
        device_data.append(torch.cat(parts))
        parts.clear()
    data,y,segments,vol=device_data
    net=ReturnNet(data.shape[2]).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=.0008,weight_decay=.015)
    ema=ReturnNet(data.shape[2]).cuda();ema.load_state_dict(net.state_dict())
    ws=torch.tensor([.30,.35,.35],device='cuda')
    for epoch in range(1,max(epochs)+1):
        net.train()
        for ids in torch.randperm(len(data),device='cuda').split(48):
            v=net(data[ids].float())
            total=decode(v); weight=(.45+.55*vol[ids]).clamp(.6,2.5).reciprocal().square()[:,:,None]
            err=nn.functional.smooth_l1_loss(total,y[ids],reduction='none',beta=1.4)
            segerr=nn.functional.smooth_l1_loss(v,segments[ids],reduction='none',beta=1.4)
            loss=(weight*(err*ws).sum(-1,keepdim=True)).mean()+.3*(weight*segerr).mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.015)
        if epoch==4:
            for optgroup in opt.param_groups:optgroup['lr']=.0004
        if epoch in epochs:
            torch.save(ema.state_dict(),OUT/f'net_{end}_{epoch}.pth')
        print('training',end,epoch,round(float(loss),5),flush=True)
    del data,device_data
    torch.cuda.empty_cache()


@torch.inference_mode()
def predict(d,nets):
    feat=np.load(OUT/f'features_{d}.npy',mmap_mode='r')
    result=[]
    for i in range(0,len(feat),64):
        x=torch.from_numpy(np.array(feat[i:i+64])).cuda().float()
        result.append(torch.stack([decode(model(x)) for model in nets]).cpu().numpy())
    return np.concatenate(result,axis=1)


def correlations(p,y):
    return prior.corr(p.reshape(-1,3),y.reshape(-1,3))


def blend(p,n,weights):
    delta=n-p
    market=delta.mean(1,keepdims=True)
    return p+weights[0]*market+weights[1]*(delta-market)


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    for d in range(20):
        fp=OUT/f'features_{d}.npy'
        if fp.exists():continue
        x=grid(d);feat=features(x)
        np.testing.assert_array_equal(feat[:67],features(x[:67]))
        np.save(fp,feat)
        print('features',d+1,flush=True)
    all_early=[];report={}
    for end,stop in ((12,14),(14,16),(16,20)):
        epochs=EPOCHS if end<16 else sorted(set(chosen.tolist()))
        if not all((OUT/f'net_{end}_{ep}.pth').exists() for ep in epochs):
            train(end,epochs)
        nets=[ReturnNet(np.load(OUT/'features_0.npy',mmap_mode='r').shape[-1]).cuda().eval() for _ in epochs]
        for net,ep in zip(nets,epochs):
            net.load_state_dict(torch.load(OUT/f'net_{end}_{ep}.pth',weights_only=True))
        batch=[]
        for d in range(end,stop):
            p=np.asarray(cache.load(d));y=prior.labels(d).reshape(p.shape)
            candidates=predict(d,nets)
            batch.append((p,y,candidates))
        if end<16:
            all_early.extend(batch)
            p,y=map(np.concatenate,zip(*[(v[0],v[1]) for v in all_early]))
            best=np.full(3,-np.inf);weights=np.zeros((2,3));chosen=np.zeros(3,dtype=int)
            for j,epoch in enumerate(epochs):
                n=np.concatenate([v[2][j] for v in all_early])
                for a in (0.,.1,.25,.5,.75,1.):
                    for b in (0.,.1,.25,.5,.75,1.):
                        cor=correlations(blend(p,n,np.array([[a],[b]])),y)
                        sel=cor>best;best=np.maximum(best,cor)
                        chosen[sel]=epoch;weights[0,sel]=a;weights[1,sel]=b
            print('early days through',stop,chosen,weights.tolist(),best,flush=True)
        rows=[]
        for d,(p,y,n) in zip(range(end,stop),batch):
            side=np.stack([n[epochs.index(ep),:,:,j] for j,ep in enumerate(chosen)],axis=2)
            q=blend(p,side,weights)
            if end==16:
                np.save(OUT/f'pred_{d}.npy',q.astype(np.float32))
            rows.append((p.reshape(-1,3),q.reshape(-1,3),y.reshape(-1,3)))
        result=prior.measure(rows)
        report[str(stop)]={'epochs':chosen.tolist(),'weights':weights.tolist(),'result':result}
        (OUT/'results.json').write_text(json.dumps(report,indent=2))
        print('score',end,result,flush=True)
        del nets,batch
        torch.cuda.empty_cache()


if __name__=='__main__':
    main()
