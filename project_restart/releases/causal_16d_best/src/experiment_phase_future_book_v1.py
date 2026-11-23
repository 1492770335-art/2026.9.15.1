"""Jointly forecast returns and future anonymous quote-state changes, with phase-specific heads."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_book_spline_network_v1 as spline
import experiment_endpoint_innovation_v1 as inn
import experiment_intraday_relative_calibration_v1 as ref
import experiment_pooled_price_impact_v1 as impact
import experiment_peer_return_distillation_v1 as fits
import experiment_multiscale_return_net_v1 as v1
OUT=inn.prior.ART/'phase_future_book_v1'
OBS=[11,14,16,23,26,29,34,36,38,39,40,41,42,50,51,54,55,60,64,72,74,94]
def auxiliary(raw,t):
    f=np.arcsinh(raw[:,:,OBS].astype(np.float32))
    prefix=np.concatenate((np.zeros_like(f[:1],dtype=float),f.cumsum(0,dtype=float)))
    out=[]
    for lo,hi in [(1,6),(6,21),(21,61)]:
        avg=(prefix[t+hi]-prefix[t+lo])/(hi-lo)
        out.append(np.clip(avg-f[t],-4,4))
    return np.concatenate(out,axis=-1).astype(np.float16)

class BookNet(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.current=nn.Sequential(nn.Linear(n,144),nn.LayerNorm(144),nn.SiLU(),nn.Dropout(.12),nn.Linear(144,96))
        self.stock=nn.Parameter(torch.zeros(50,96))
        self.norm=nn.LayerNorm(96)
        self.attn=nn.MultiheadAttention(96,4,dropout=.1,batch_first=True)
        self.factors=nn.Linear(96,24)
        self.reg=nn.Sequential(nn.Linear(144,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.15))
        self.idio=nn.Linear(96,6)
        self.market=nn.Sequential(nn.Linear(144,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.2),nn.Linear(64,6))
        self.aux=nn.Linear(96,3*len(OBS))
        self.register_buffer('port',torch.zeros(4,50));self.register_buffer('beta',torch.ones(50))
    def forward(self,x,vol,gate):
        h=self.current(x)+self.stock
        n=self.norm(h)
        h=h+self.attn(n,n,n,need_weights=False)[0]
        z=self.factors(h);f=self.port@z;m=z.mean(1,keepdim=True)
        out=self.reg(torch.cat((h,self.port.T@f,m.expand_as(z)),axis=2))
        a,b=self.idio(out).chunk(2,dim=-1)
        loc=(a+gate[:,None,:]*b)*vol[:,:,None]
        loc=loc-loc.mean(1,keepdim=True)
        a,b=self.market(torch.cat((m[:,0],z.std(1,unbiased=False),f.flatten(1)),dim=1)).chunk(2,dim=-1)
        market=(a+gate*b)*vol.mean(1,keepdim=True)
        return loc+self.beta[None,:,None]*market[:,None,:],self.aux(out),loc,market

def train(end, seed=281, offset=0, out=OUT, model_class=BookNet, make_features=spline.features, make_aux=auxiliary):
    torch.manual_seed(seed)
    # Sample the high-variance first 600 frames more densely and retain all training dates.
    ticks=[np.r_[np.arange((d+offset)%2,600,2),np.arange(600+(d+offset)%6,len(inn.grid(d))-61,6)] for d in range(end)]
    n=sum(len(t) for t in ticks);at=0
    y=torch.empty(n,50,3,device='cuda');v=torch.empty(n,50,device='cuda')
    aux=torch.empty(n,50,3*len(OBS),device='cuda',dtype=torch.float16);g=torch.empty(n,1,device='cuda')
    for d,t in enumerate(ticks):
        raw=inn.grid(d);a,s=make_features(raw,t)
        if d==0:x=torch.empty(n,50,a.shape[-1],device='cuda',dtype=torch.float16)
        sl=slice(at,at+len(t));at+=len(t)
        x[sl].copy_(torch.from_numpy(a));y[sl].copy_(torch.from_numpy(inn.prior.labels(d).reshape(-1,50,3)[t]))
        v[sl].copy_(torch.from_numpy((.45+.55*s).clip(.55,3)))
        aux[sl].copy_(torch.from_numpy(make_aux(raw,t)));g[sl].copy_(torch.from_numpy(np.exp(-t[:,None]/400).astype(np.float32)))
    net=model_class(x.shape[-1]).cuda();net.port.copy_(torch.from_numpy(impact.exposure(end)[0][-4:]*np.sqrt(50)))
    net.beta.copy_(torch.from_numpy(inn.base.ex.geometry(end)[0]))
    ema=model_class(x.shape[-1]).cuda();ema.load_state_dict(net.state_dict())
    opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.025)
    for ep in range(1,17):
        for ids in torch.randperm(n,device='cuda').split(48):
            yy=y[ids];vol=v[ids];ym=yy.mean(1);yr=yy-net.beta[None,:,None]*ym[:,None]
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                p,delta,r,m=net(x[ids].float(),vol,g[ids])
            loss=F.smooth_l1_loss(r/vol[:,:,None],yr/vol[:,:,None],beta=1.25)+.6*F.smooth_l1_loss(m/vol.mean(1,keepdim=True),ym/vol.mean(1,keepdim=True),beta=.45)
            loss+=.35*F.smooth_l1_loss(delta.float(),aux[ids].float(),beta=.6)
            loss-=.06*F.cosine_similarity(r,yr,dim=1,eps=1e-5).mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.016)
        if ep in (6,11):
            for group in opt.param_groups:group['lr']*=.55
        if ep in (1,8,16):print('training',end,ep,flush=True)
    torch.save(ema.state_dict(),out/f'net_{end}.pth');return ema.eval()

@torch.inference_mode()
def infer(net,raw,make_features=spline.features):
    n=len(raw);out=np.empty((n,50,3),np.float32)
    for k in range(4):
        t=np.arange(k,n,4);a,v=make_features(raw,t)
        for i in range(0,len(t),48):
            ti=t[i:i+48];x=torch.from_numpy(a[i:i+48]).cuda().float()
            vol=torch.from_numpy(v[i:i+48]).cuda();g=torch.from_numpy(np.exp(-ti[:,None]/400).astype(np.float32)).cuda()
            out[ti]=net(x,(.45+.55*vol).clamp(.55,3),g)[0].cpu().numpy()
    return out

def main():
    OUT.mkdir(exist_ok=True);early=[];report={}
    for end,stop in ((12,14),(14,16),(16,20)):
        ck=OUT/f'net_{end}.pth'
        if ck.exists():
            state=torch.load(ck,weights_only=True,map_location='cpu')
            net=BookNet(state['current.0.weight'].shape[1]).cuda().eval();net.load_state_dict(state)
        else:net=train(end)
        rows=[]
        for d in range(end,stop):
            p=np.load(ref.OUT/f'pred_{d}.npy');raw=inn.grid(d);n=infer(net,raw)
            if d==end:np.testing.assert_allclose(n[:145],infer(net,raw[:145]),atol=1e-4)
            rows.append((p,inn.prior.labels(d).reshape(p.shape),n))
        if end==12:_,w=fits.coefficients(rows)
        entries=[]
        for day,(p,y,n) in zip(range(end,stop),rows):
            q=v1.blend(p,n,w);np.save(OUT/f'side_{day}.npy',n);np.save(OUT/f'pred_{day}.npy',q.astype(np.float32))
            entries.append(tuple(a.reshape(-1,3) for a in (p,q,y)))
        val=inn.prior.measure(entries);report[stop]={'w':w.tolist(),'result':val}
        (OUT/'results.json').write_text(json.dumps(report,indent=2));print(stop,report[stop],flush=True)
        if end<16:early.extend(rows)
        if end==14:
            if val['gain']<=0:break
            _,w=fits.coefficients(early)
if __name__=='__main__':main()
