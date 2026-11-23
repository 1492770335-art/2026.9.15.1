"""Supervise distinct market, dynamic factor and stock-specific forecasts using causal stock exposures."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_phase_future_book_v1 as phase
import experiment_full_coverage_book_v1 as cache
import experiment_causal_price_level_centering_v1 as readout

OUT=phase.inn.prior.ART/'dynamic_factor_return_net_v1'
K=4

def geometry(end):
    cov=[]
    for d in range(end):
        r=phase.inn.moving_sum(phase.inn.grid(d)[:,:,0].astype(float),24)[23::24]
        r-=r.mean(1,keepdims=True);cov.append(np.cov(r.T))
    c=np.mean(cov,0);sd=np.sqrt(c.diagonal().clip(.02,None))
    _,u=np.linalg.eigh(c/np.outer(sd,sd))
    b=u[:,-K:]*sd[:,None];b-=b.mean(0)
    return (b/np.sqrt((b*b).mean(0))).astype('float32')

class FactorNet(phase.BookNet):
    def __init__(self,n):
        super().__init__(n)
        self.aux=nn.Identity()
        self.loading=nn.Linear(96,K+1)
        self.factor_latent=nn.Linear(96,16)
        self.factor_head=nn.Sequential(nn.Linear(48,48),nn.LayerNorm(48),nn.SiLU(),nn.Dropout(.14),nn.Linear(48,6))
        self.future=nn.Linear(96,66)
        self.register_buffer('base_loading',torch.zeros(50,K))
        self.register_buffer('identity',torch.eye(K))
        nn.init.zeros_(self.loading.weight);nn.init.zeros_(self.loading.bias)
    def forward(self,x,v,gate):
        _,h,r,m=super().forward(x,v,gate)
        adj=self.loading(h).float().tanh()
        beta=self.beta[None]+.35*adj[:,:,0]
        beta=beta-beta.mean(1,keepdim=True)+1
        B=self.base_loading[None]+.28*adj[:,:,1:]
        B=B-B.mean(1,keepdim=True)
        # Solve only a four-dimensional Gram system per frame.
        with torch.autocast(device_type='cuda',enabled=False):
            B=B.float();proj=torch.linalg.solve(B.transpose(1,2)@B/50+.016*self.identity,B.transpose(1,2)/50)
            r=r.float();residual=r-B@(proj@r)
        z=self.factor_latent(h);factor=proj.to(z.dtype)@z
        common=z.mean(1,keepdim=True).expand(-1,K,-1);width=z.std(1,unbiased=False)[:,None].expand_as(common)
        a,b=self.factor_head(torch.cat((factor,common,width),dim=2)).chunk(2,-1)
        f=(a+b*gate[:,None,:])*v.mean(1)[:,None,None]
        p=residual+B@f+beta[:,:,None]*m[:,None]
        return p,self.future(h),residual,m,f,proj,beta

def train(end):
    torch.manual_seed(8761)
    ts=[np.r_[np.arange(d%2,600,2),np.arange(600+d%6,len(phase.inn.grid(d))-61,6)] for d in range(end)]
    n=sum(map(len,ts));at=0;y=torch.empty(n,50,3,device='cuda');v=torch.empty(n,50,device='cuda')
    g=torch.empty(n,1,device='cuda');aux=torch.empty(n,50,66,device='cuda',dtype=torch.float16)
    for d,t in enumerate(ts):
        cache.prepare(d);root=cache.CACHE/f'day_{d:02d}';fx=np.load(root/'x.npy',mmap_mode='r')
        if d==0:x=torch.empty(n,50,fx.shape[-1],dtype=torch.float16,device='cuda')
        sl=slice(at,at+len(t));at+=len(t);x[sl].copy_(torch.from_numpy(fx[t]))
        aux[sl].copy_(torch.from_numpy(np.load(root/'a.npy',mmap_mode='r')[t]))
        v[sl].copy_(torch.from_numpy((.45+.55*np.load(root/'v.npy',mmap_mode='r')[t]).clip(.55,3)))
        y[sl].copy_(torch.from_numpy(phase.inn.prior.labels(d).reshape(-1,50,3)[t]));g[sl].copy_(torch.from_numpy(np.exp(-t[:,None]/400).astype('float32')))
    net=FactorNet(x.shape[-1]).cuda();net.port.copy_(torch.from_numpy(phase.impact.exposure(end)[0][-4:]*np.sqrt(50)))
    net.beta.copy_(torch.from_numpy(phase.inn.base.ex.geometry(end)[0]));net.base_loading.copy_(torch.from_numpy(geometry(end)))
    ema=FactorNet(x.shape[-1]).cuda();ema.load_state_dict(net.state_dict());opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.028)
    for ep in range(1,17):
        for ids in torch.randperm(n,device='cuda').split(48):
            yy=y[ids];vol=v[ids];ym=yy.mean(1)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                p,bk,loc,m,f,proj,beta=net(x[ids].float(),vol,g[ids])
            yr=yy-beta.detach()[:,:,None]*ym[:,None]
            actual_f=proj.detach()@yr
            loss=.88*F.smooth_l1_loss((p-p.mean(1,keepdim=True))/vol[:,:,None],(yy-ym[:,None])/vol[:,:,None],beta=1.15)
            loss+=.55*F.smooth_l1_loss(m/vol.mean(1,keepdim=True),ym/vol.mean(1,keepdim=True),beta=.45)
            loss+=.39*F.smooth_l1_loss(f/vol.mean(1)[:,None,None],actual_f/vol.mean(1)[:,None,None],beta=.38)
            loss+=.32*F.smooth_l1_loss(bk.float(),aux[ids].float(),beta=.6)
            loss-=.055*F.cosine_similarity(p-p.mean(1,keepdim=True),yy-ym[:,None],dim=1,eps=1e-5).mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.016)
        if ep in (6,11):
            for g1 in opt.param_groups:g1['lr']*=.55
        if ep in (1,8,16):print('training',end,ep,flush=True)
    torch.save(ema.state_dict(),OUT/f'net_{end}.pth');return ema.eval()

def main():
    OUT.mkdir(exist_ok=True);hist=[];report={};mask=np.ones((2,3),bool)
    for end,stop in ((12,14),(14,16),(16,20)):
        ck=OUT/f'net_{end}.pth'
        if ck.exists():
            state=torch.load(ck,map_location='cpu',weights_only=True)
            net=FactorNet(state['current.0.weight'].shape[1]).cuda().eval();net.load_state_dict(state)
        else:net=train(end)
        if end>=14:
            coef=readout.fit(hist)*mask[:,None,:]
            if end==16:np.save(OUT/'coef.npy',coef)
        rows=[];daily=[]
        for d in range(end,stop):
            raw=phase.inn.grid(d);n=phase.infer(net,raw);np.save(OUT/f'side_{d}.npy',n)
            if d==end:np.testing.assert_allclose(n[:153],phase.infer(net,raw[:153]),atol=1.e-4)
            p=readout.original(d);y=phase.inn.prior.labels(d).reshape(p.shape);z=readout.basis(p,n)
            if end<16:hist.append(readout.moments(p,y,z))
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
