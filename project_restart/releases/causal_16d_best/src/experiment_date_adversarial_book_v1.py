"""Discourage a market-level date classifier from recovering training-day identity from the book representation."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_phase_future_book_v1 as phase
import experiment_full_coverage_book_v1 as cache
import experiment_causal_price_level_centering_v1 as readout

OUT=phase.inn.prior.ART/'date_adversarial_book_v1'

class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,scale):
        ctx.scale=scale
        return x.view_as(x)
    @staticmethod
    def backward(ctx,grad):
        return -ctx.scale*grad,None

def train(end):
    torch.manual_seed(281)
    ticks=[np.r_[np.arange(d%2,600,2),np.arange(600+d%6,len(phase.inn.grid(d))-61,6)] for d in range(end)]
    n=sum(map(len,ticks));at=0
    y=torch.empty(n,50,3,device='cuda');v=torch.empty(n,50,device='cuda');g=torch.empty(n,1,device='cuda')
    dates=torch.empty(n,device='cuda',dtype=torch.long);aux=torch.empty(n,50,66,device='cuda',dtype=torch.float16)
    for d,t in enumerate(ticks):
        cache.prepare(d);root=cache.CACHE/f'day_{d:02d}'
        fx=np.load(root/'x.npy',mmap_mode='r')
        if d==0:x=torch.empty(n,50,fx.shape[-1],device='cuda',dtype=torch.float16)
        sl=slice(at,at+len(t));at+=len(t);dates[sl]=d
        x[sl].copy_(torch.from_numpy(fx[t]));aux[sl].copy_(torch.from_numpy(np.load(root/'a.npy',mmap_mode='r')[t]))
        v[sl].copy_(torch.from_numpy((.45+.55*np.load(root/'v.npy',mmap_mode='r')[t]).clip(.55,3.)))
        y[sl].copy_(torch.from_numpy(phase.inn.prior.labels(d).reshape(-1,50,3)[t]))
        g[sl].copy_(torch.from_numpy(np.exp(-t[:,None]/400).astype('float32')))
    net=phase.BookNet(x.shape[-1]).cuda();net.port.copy_(torch.from_numpy(phase.impact.exposure(end)[0][-4:]*np.sqrt(50)))
    net.beta.copy_(torch.from_numpy(phase.inn.base.ex.geometry(end)[0]))
    ema=phase.BookNet(x.shape[-1]).cuda();ema.load_state_dict(net.state_dict())
    disc=nn.Sequential(nn.Linear(194,80),nn.LayerNorm(80),nn.SiLU(),nn.Dropout(.15),nn.Linear(80,end)).cuda()
    opt=torch.optim.AdamW([*net.parameters(),*disc.parameters()],lr=.0007,weight_decay=.025)
    latent={}
    handle=net.reg.register_forward_hook(lambda mod,args,result:latent.update(stock=result))
    for ep in range(1,17):
        correct=0
        for ids in torch.randperm(n,device='cuda').split(48):
            yy=y[ids];vol=v[ids];ym=yy.mean(1);yr=yy-net.beta[None,:,None]*ym[:,None]
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                p,delta,r,m=net(x[ids].float(),vol,g[ids])
                h=latent.pop('stock');market=torch.cat((h.mean(1),h.std(1,unbiased=False)),1)
                # The auxiliaries use training-day identities only. Neither classifier nor date enters inference.
                inv=Reverse.apply(market,min(.65,ep/12))
                logits=disc(torch.cat((inv,g[ids],vol.mean(1,keepdim=True)),1))
            loss=F.smooth_l1_loss(r/vol[:,:,None],yr/vol[:,:,None],beta=1.25)
            loss+=.6*F.smooth_l1_loss(m/vol.mean(1,keepdim=True),ym/vol.mean(1,keepdim=True),beta=.45)
            loss+=.35*F.smooth_l1_loss(delta.float(),aux[ids].float(),beta=.6)
            loss-=.06*F.cosine_similarity(r,yr,dim=1,eps=1e-5).mean()
            loss+=.14*F.cross_entropy(logits.float(),dates[ids])
            correct+=(logits.argmax(1)==dates[ids]).sum().item()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.016)
        if ep in (6,11):
            for group in opt.param_groups:group['lr']*=.55
        if ep in (1,8,16):print('training',end,ep,'date classifier accuracy',round(correct/n,3),flush=True)
    handle.remove();torch.save(ema.state_dict(),OUT/f'net_{end}.pth');return ema.eval()

def main():
    OUT.mkdir(exist_ok=True);hist=[];report={};mask=np.ones((2,3),bool)
    for end,stop in ((12,14),(14,16),(16,20)):
        ck=OUT/f'net_{end}.pth'
        if ck.exists():
            s=torch.load(ck,map_location='cpu',weights_only=True);net=phase.BookNet(s['current.0.weight'].shape[1]).cuda().eval();net.load_state_dict(s)
        else:net=train(end)
        if end>=14:
            coef=readout.fit(hist)*mask[:,None,:]
            if end==16:np.save(OUT/'coef.npy',coef)
        rows=[];daily=[]
        for d in range(end,stop):
            raw=phase.inn.grid(d);n=phase.infer(net,raw)
            if d==end:np.testing.assert_allclose(n[:169],phase.infer(net,raw[:169]),atol=1e-4)
            p=readout.original(d);y=phase.inn.prior.labels(d).reshape(p.shape);z=readout.basis(p,n)
            if end<16:hist.append(readout.moments(p,y,z))
            np.save(OUT/f'side_{d}.npy',n)
            if end>=14:
                terms=[(z[0]@coef[0])[:,None],z[1]@coef[1]]
                q=(p+sum(terms)).astype('float32');np.save(OUT/f'pred_{d}.npy',q)
                rows.append(tuple(a.reshape(-1,3) for a in (p,q,y)))
                if end==14:
                    old=phase.inn.prior.corr(p.reshape(-1,3),y.reshape(-1,3))
                    daily.append([phase.inn.prior.corr((p+c).reshape(-1,3),y.reshape(-1,3))-old for c in terms])
        if end>=14:
            val=phase.inn.prior.measure(rows);report[str(stop)]=val
            (OUT/'results.json').write_text(json.dumps(report,indent=2));print(stop,val,flush=True)
            if end==14:
                mask=np.min(daily,axis=0)>0
                if not mask.any():break
if __name__=='__main__':main()
