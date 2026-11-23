"""Estimate conditional means of heavy-tailed market and stock returns with asymmetric scales."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_phase_future_book_v1 as phase
import experiment_full_coverage_book_v1 as cache
from experiment_crossfit_predictable_tails_v1 import run_validation

OUT=phase.inn.prior.ART/'asymmetric_return_distribution_v1'


class DistributionNet(phase.BookNet):
    def __init__(self,n):
        super().__init__(n)
        self.aux=nn.Identity();self.future=nn.Linear(96,66)
        self.relative_shape=nn.Linear(96,6)
        self.market_shape=nn.Sequential(nn.Linear(192,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.15),nn.Linear(64,6))
        self.df_logit=nn.Parameter(torch.zeros(2,3))

    def forward(self,x,vol,gate):
        _,h,loc,market=super().forward(x,vol,gate)
        v=vol.mean(1,keepdim=True)
        common=self.market_shape(torch.cat((h.mean(1),h.std(1,unbiased=False)),dim=1))
        geometry=torch.cat((self.relative_shape(h),common[:,None]),axis=1)
        basic,tilt=geometry.float().chunk(2,dim=2)
        sigma=.045+F.softplus(basic)
        tilt=.75*tilt.tanh()
        left=sigma*torch.exp(-tilt);right=sigma*torch.exp(tilt)
        mode=torch.cat((loc/vol[:,:,None],(market/v)[:,None]),dim=1)
        df=2.35+4*torch.sigmoid(self.df_logit)
        df=torch.cat([df[0:1].expand(50,-1),df[1:]],dim=0)
        # E|T| for a Student distribution supplies the analytic two-piece mean.
        absolute_mean=(2*torch.sqrt(df/torch.pi)*torch.exp(torch.lgamma((df+1)/2)-torch.lgamma(df/2))/(df-1))
        mean=mode+absolute_mean*(right-left)
        r=mean[:,:50]*vol[:,:,None];m=mean[:,50]*v
        return r-r.mean(1,keepdim=True)+m[:,None],self.future(h),mode,left,right,df,mean


def train(end):
    torch.manual_seed(7312)
    ts=[np.r_[np.arange(d%2,600,2),np.arange(600+d%6,len(phase.inn.grid(d))-61,6)] for d in range(end)]
    n=sum(map(len,ts));start=0
    y=torch.empty(n,50,3,device='cuda');v=torch.empty(n,50,device='cuda')
    gate=torch.empty(n,1,device='cuda');aux=torch.empty(n,50,66,device='cuda',dtype=torch.float16)
    for d,t in enumerate(ts):
        cache.prepare(d);root=cache.CACHE/f'day_{d:02d}'
        fx=np.load(root/'x.npy',mmap_mode='r')
        if d==0:x=torch.empty(n,50,fx.shape[-1],device='cuda',dtype=torch.float16)
        sl=slice(start,start+len(t));start+=len(t)
        x[sl].copy_(torch.from_numpy(fx[t]))
        aux[sl].copy_(torch.from_numpy(np.load(root/'a.npy',mmap_mode='r')[t]))
        y[sl].copy_(torch.from_numpy(phase.inn.prior.labels(d).reshape(-1,50,3)[t]))
        v[sl].copy_(torch.from_numpy((.45+.55*np.load(root/'v.npy',mmap_mode='r')[t]).clip(.55,3)))
        gate[sl].copy_(torch.from_numpy(np.exp(-t[:,None]/400).astype('float32')))
    net=DistributionNet(x.shape[-1]).cuda()
    net.port.copy_(torch.from_numpy(phase.impact.exposure(end)[0][-4:]*np.sqrt(50)))
    net.beta.copy_(torch.from_numpy(phase.inn.base.ex.geometry(end)[0]))
    ema=DistributionNet(x.shape[-1]).cuda();ema.load_state_dict(net.state_dict())
    opt=torch.optim.AdamW(net.parameters(),lr=.00063,weight_decay=.029)
    for ep in range(16):
        for ids in torch.randperm(n,device='cuda').split(48):
            yy=y[ids];vol=v[ids];ym=yy.mean(1,keepdim=True)
            target=torch.cat(((yy-ym)/vol[:,:,None],ym/vol.mean(1)[:,None,None]),dim=1)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                p,bk,mode,left,right,df,mean=net(x[ids].float(),vol,gate[ids])
            error=target-mode
            s=torch.where(error<0,left,right)
            constant=.5*torch.log(df*torch.pi)+torch.lgamma(.5*df)-torch.lgamma(.5*(df+1))
            nll=torch.log((left+right)/2)+constant+.5*(df+1)*torch.log1p((error/s)**2/df)
            loss=.44*(nll[:,:50].mean()+.56*nll[:,50].mean())
            loss+=.67*F.smooth_l1_loss(mean[:,:50],target[:,:50],beta=1.35)+.38*F.smooth_l1_loss(mean[:,50],target[:,50],beta=.56)
            loss+=.26*F.smooth_l1_loss(bk.float(),aux[ids].float(),beta=.6)
            loss-=.055*F.cosine_similarity(p-p.mean(1,keepdim=True),yy-ym,dim=1,eps=1e-5).mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.017)
        if ep in (5,10):
            for group in opt.param_groups:group['lr']*=.52
        if ep in (0,7,15):print('training',end,ep+1,flush=True)
    torch.save(ema.state_dict(),OUT/f'net_{end}.pth')
    return ema.eval()


if __name__=='__main__':run_validation(OUT,train,lambda st:DistributionNet(st['current.0.weight'].shape[1]))

