"""Train a stock-relative, exposure-normalized predictor with smoothed return endpoints."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_multiscale_return_net_v1 as v1
import experiment_peer_return_distillation_v1 as utils
import experiment_factor_exposure_return_v1 as ex
prior=v1.prior
OUT=prior.ART/'relative_price_endpoint_v1'
EPOCHS=(4,8,14,22)


def stats(end):
    x=np.concatenate([v1.grid(d)[::32,:,:83].astype(np.float32) for d in range(end)])
    lo,mid,hi=np.quantile(np.arcsinh(x),[.16,.5,.84],axis=0)
    return mid.astype(np.float32),np.maximum(.12,.5*(hi-lo)).astype(np.float32)


def features(x,fit):
    mid,scale=fit;t=np.arange(len(x));end=t+1
    z=np.arcsinh(x[:,:,:83].astype(np.float32));norm=np.arcsinh((z-mid)/scale)
    sums=np.concatenate((np.zeros_like(z[:1],dtype=float),z.cumsum(0,dtype=float)))
    r=x[:,:,0].astype(float)
    price=np.vstack((np.zeros(50),r.cumsum(0)))
    rv=np.vstack((np.zeros(50),(r*r).cumsum(0)))
    cols=[z,norm,x[:,:,83:].astype(np.float32)-1.,norm[np.maximum(0,t-1)],norm[np.maximum(0,t-3)]]
    for h in (6,30,160):
        begin=np.maximum(0,end-h)
        avg=((sums[end]-sums[begin])/(end-begin)[:,None,None]).astype(np.float32)
        cols.extend(((z-avg)/scale,np.arcsinh((avg-mid)/scale)))
    tail=[]
    for h in (5,20,60,240):
        begin=np.maximum(0,end-h);length=(end-begin)[:,None]
        ret=(price[end]-price[begin])/np.sqrt(length)
        vol=np.sqrt((rv[end]-rv[begin]+4)/(length+4)).clip(.1,8.)
        tail.extend((np.arcsinh(ret/vol),np.log(vol)))
    tail.append(np.broadcast_to(np.exp(-t/256)[:,None],r.shape))
    cols.append(np.stack(tail,axis=-1).astype(np.float32))
    return np.concatenate(cols,axis=-1).clip(-10,10).astype(np.float16)


def target(day):
    y=prior.labels(day).reshape(-1,50,3).copy()
    price=v1.grid(day)[:,:,0].cumsum(0,dtype=float)
    for j,(h,w) in enumerate([(5,2),(20,6),(60,16)]):
        t=np.arange(len(y)-h-w)
        avg=sum(price[t+h+i] for i in range(-w,w+1))/(2*w+1)
        y[t,:,j]+=.65*v1.SCALE[j]*(avg-price[t+h])
    return y


class RelativeNet(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.embed=nn.Sequential(nn.Linear(n,112),nn.LayerNorm(112),nn.SiLU())
        self.stock=nn.Parameter(torch.zeros(50,16))
        self.book=nn.Sequential(nn.Linear(128,80),nn.LayerNorm(80),nn.SiLU(),nn.Dropout(.12))
        self.message=nn.Linear(80,32)
        self.local=nn.Sequential(nn.Linear(144,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.18),nn.Linear(96,48),nn.SiLU(),nn.Linear(48,3))
        self.market=nn.Sequential(nn.Linear(64,48),nn.LayerNorm(48),nn.SiLU(),nn.Dropout(.12),nn.Linear(48,3))
        self.register_buffer('graph',torch.eye(50));self.register_buffer('beta',torch.ones(50,1))

    def forward(self,x):
        vol=(.35+.65*x[:,:,-4:-3].exp()).clamp(.5,3.5)
        h=self.book(torch.cat((self.embed(x),self.stock.expand(len(x),-1,-1)),dim=-1))
        z=self.message(h);m=z.mean(1,keepdim=True)
        market=self.market(torch.cat((m[:,0],z.std(1,unbiased=False)),dim=-1))[:,None,:]*vol.mean(1,keepdim=True)
        rel=self.local(torch.cat((h,self.graph@z-m,m.expand_as(z)),dim=-1))*vol
        rel=rel-rel.mean(1,keepdim=True)
        return market*self.beta+rel,rel,market,vol


def train(end,epochs,fit,graph,beta):
    torch.manual_seed(2301)
    ticks=[np.arange((d+1)%4,len(v1.grid(d)),4) for d in range(end)]
    n=sum(map(len,ticks));ys=torch.empty(n,50,3,device='cuda');pos=0
    for d,t in enumerate(ticks):
        raw=v1.grid(d);z=features(raw,fit)
        if d==0:
            np.testing.assert_array_equal(z[:89],features(raw[:89],fit))
            inp=torch.empty(n,50,z.shape[-1],device='cuda',dtype=torch.float16)
        sl=slice(pos,pos+len(t));pos+=len(t)
        inp[sl].copy_(torch.from_numpy(z[t]));ys[sl].copy_(torch.from_numpy(target(d)[t]))
    model=RelativeNet(inp.shape[-1]).cuda();model.graph.copy_(torch.from_numpy(graph));model.beta.copy_(torch.from_numpy(beta[:,None]))
    ema=RelativeNet(inp.shape[-1]).cuda();ema.load_state_dict(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=.0009,weight_decay=.015)
    for ep in range(1,max(epochs)+1):
        for ids in torch.randperm(n,device='cuda').split(48):
            y=ys[ids];p,r,m,vol=model(inp[ids].float())
            ym=y.mean(1,keepdim=True);yr=y-model.beta*ym
            loss=F.smooth_l1_loss(r/vol,yr/vol,beta=1.25)+.45*F.smooth_l1_loss((m-ym)/vol.mean(1,keepdim=True),torch.zeros_like(ym),beta=.6)
            loss-=.09*F.cosine_similarity(p-p.mean(1,keepdim=True),y-ym,dim=1,eps=1e-5).mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),.013)
        if ep in (6,14):
            for g in opt.param_groups:g['lr']*=.55
        if ep in epochs:torch.save(ema.state_dict(),OUT/f'net_{end}_{ep}.pth')
        print('train',end,ep,float(loss),flush=True)


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
    early={ep:[] for ep in EPOCHS};report={}
    for end,stop in [(12,14),(14,16),(16,20)]:
        fit=stats(end);beta=ex.geometry(end)[0];graph=utils.basket(end)
        epochs=EPOCHS if end<16 else sorted(set(chosen.tolist()))
        if not all((OUT/f'net_{end}_{ep}.pth').exists() for ep in epochs):
            train(end,epochs,fit,graph,beta)
        np.savez(OUT/f'stats_{end}.npz',mid=fit[0],scale=fit[1])
        rows={ep:[] for ep in epochs}
        for d in range(end,stop):
            p=ex.v1_base(d);y=prior.labels(d).reshape(p.shape);x=features(v1.grid(d),fit)
            for ep in epochs:
                model=RelativeNet(x.shape[-1]).eval().cuda()
                model.load_state_dict(torch.load(OUT/f'net_{end}_{ep}.pth',weights_only=True))
                with torch.inference_mode():
                    pred=np.concatenate([model(torch.from_numpy(x[k:k+96]).cuda().float())[0].cpu().numpy() for k in range(0,len(x),96)])
                rows[ep].append((p,y,pred))
        if end<16:
            best=np.full(3,-np.inf);weights=np.zeros((2,3));chosen=np.zeros(3,int)
            for ep in epochs:
                early[ep].extend(rows[ep]);cs,w=utils.coefficients(early[ep])
                ix=cs>best;weights[:,ix]=w[:,ix];chosen[ix]=ep;best=np.maximum(best,cs)
        entries=[]
        for i,d in enumerate(range(end,stop)):
            p,y,_=rows[epochs[0]][i]
            n=np.stack([rows[ep][i][2][:,:,j] for j,ep in enumerate(chosen)],axis=-1)
            q=v1.blend(p,n,weights)
            np.save(OUT/f'side_{d}.npy',n);np.save(OUT/f'pred_{d}.npy',q.astype(np.float32))
            entries.append((p.reshape(-1,3),q.reshape(-1,3),y.reshape(-1,3)))
        report[stop]={'epochs':chosen.tolist(),'weights':weights.tolist(),'result':prior.measure(entries)}
        (OUT/'results.json').write_text(json.dumps(report,indent=2));print('score',stop,report[stop],flush=True)

if __name__=='__main__':main()
