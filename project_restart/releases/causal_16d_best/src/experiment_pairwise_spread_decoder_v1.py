"""Train an antisymmetric pairwise return model, then average pair differences to infer stock-relative returns."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_phase_future_book_v1 as phase
import experiment_variance_ratio_calibration_v1 as variance

OUT=phase.inn.prior.ART/'pairwise_spread_decoder_v1'

class PairNet(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.state=nn.Sequential(nn.Linear(width,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.12),nn.Linear(128,72))
        self.identity=nn.Parameter(torch.zeros(50,72))
        self.query=nn.Linear(72,16)
        self.local=nn.Sequential(nn.Linear(88,72),nn.LayerNorm(72),nn.SiLU(),nn.Dropout(.14))
        self.potential=nn.Linear(72,6)
        self.direction=nn.Linear(72,16,bias=False)
        self.reliability=nn.Linear(72,16)
        self.spread=nn.Linear(16,3,bias=False)
        self.aux=nn.Linear(72,3*len(phase.OBS))

    def forward(self,x,vol,gate):
        h=self.state(x)+self.identity
        m=self.query(h).mean(1,keepdim=True).expand(-1,50,-1)
        latent=self.local(torch.cat([h,m],2))
        a,b=self.potential(latent).chunk(2,dim=-1)
        potential=(a+b*gate[:,None])*vol[:,:,None]
        direction=self.direction(latent)
        confidence=self.reliability(latent)
        # Interchanging the stocks keeps the gate fixed and reverses the signed forecast.
        si=direction[:,:,None,:];sj=direction[:,None,:,:]
        u=confidence[:,:,None,:];v=confidence[:,None,:,:]
        mask=torch.sigmoid((u+v)*.5-(u-v).abs()*.45)
        spread=self.spread(torch.tanh(si-sj)*mask)
        pair_vol=(vol[:,:,None]+vol[:,None,:])*.5
        edges=potential[:,:,None]-potential[:,None]+spread*pair_vol[:,:,:,None]
        return edges.mean(2),edges,self.aux(latent)

def train(end):
    torch.manual_seed(6813)
    ticks=[np.r_[np.arange(d%3,600,3),np.arange(600+d%7,len(phase.inn.grid(d))-61,7)] for d in range(end)]
    n=sum(map(len,ticks));at=0
    y=torch.empty(n,50,3,device='cuda');v=torch.empty(n,50,device='cuda');g=torch.empty(n,1,device='cuda')
    aux=torch.empty(n,50,3*len(phase.OBS),device='cuda',dtype=torch.float16)
    for d,t in enumerate(ticks):
        raw=phase.inn.grid(d);a,s=phase.spline.features(raw,t)
        if d==0:x=torch.empty(n,50,a.shape[-1],dtype=torch.float16,device='cuda')
        sl=slice(at,at+len(t));at+=len(t);x[sl].copy_(torch.from_numpy(a))
        y[sl].copy_(torch.from_numpy(phase.inn.prior.labels(d).reshape(-1,50,3)[t]))
        v[sl].copy_(torch.from_numpy((.45+.55*s).clip(.55,3)))
        g[sl].copy_(torch.from_numpy(np.exp(-t[:,None]/400).astype('float32')))
        aux[sl].copy_(torch.from_numpy(phase.auxiliary(raw,t)))
    net=PairNet(x.shape[-1]).cuda();ema=PairNet(x.shape[-1]).cuda();ema.load_state_dict(net.state_dict())
    opt=torch.optim.AdamW(net.parameters(),lr=.00085,weight_decay=.025)
    for ep in range(1,15):
        for ids in torch.randperm(n,device='cuda').split(40):
            vol=v[ids];yy=y[ids]
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                rel,edges,auxpred=net(x[ids].float(),vol,g[ids])
            truth=yy-yy.mean(1,keepdim=True)
            diff=yy[:,:,None]-yy[:,None]
            pair_scale=(vol[:,:,None]+vol[:,None,:])*.5
            err=(edges-diff)/pair_scale[:,:,:,None]
            loss=.64*F.smooth_l1_loss(err,torch.zeros_like(err),beta=1.3)
            loss+=.4*F.smooth_l1_loss(rel/vol[:,:,None],truth/vol[:,:,None],beta=1.1)
            loss+=.32*F.smooth_l1_loss(auxpred.float(),aux[ids].float(),beta=.6)
            loss-=.07*F.cosine_similarity(rel,truth,dim=1,eps=1e-5).mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.017)
        if ep in (5,10):
            for opt_group in opt.param_groups:opt_group['lr']*=.55
        if ep in (1,7,14):print('training',end,ep,flush=True)
    torch.save(ema.state_dict(),OUT/f'net_{end}.pth')
    return ema.eval()

def main():
    OUT.mkdir(exist_ok=True);hist=[];report={}
    for end,stop in ((12,14),(14,16),(16,20)):
        ck=OUT/f'net_{end}.pth'
        if ck.exists():
            state=torch.load(ck,map_location='cpu',weights_only=True)
            net=PairNet(state['state.0.weight'].shape[1]).cuda().eval();net.load_state_dict(state)
        else:net=train(end)
        rows=[]
        for d in range(end,stop):
            raw=phase.inn.grid(d);r=phase.infer(net,raw)
            if d==end:np.testing.assert_allclose(r[:177],phase.infer(net,raw[:177]),atol=1e-4)
            p=variance.baseline(d);n=r+p.mean(1,keepdims=True);y=phase.inn.prior.labels(d).reshape(p.shape)
            np.save(OUT/f'side_{d}.npy',r);rows.append((p,y,n))
        if end==12:
            _,w=phase.fits.coefficients(rows);w[0]=0;w[1]=np.minimum(w[1],.16)
        entries=[]
        for d,(p,y,n) in zip(range(end,stop),rows):
            q=phase.v1.blend(p,n,w).astype(np.float32);np.save(OUT/f'pred_{d}.npy',q)
            entries.append(tuple(a.reshape(-1,3) for a in (p,q,y)))
        result=phase.inn.prior.measure(entries);report[str(stop)]={'w':w.tolist(),'result':result}
        (OUT/'results.json').write_text(json.dumps(report,indent=2));print(stop,report[str(stop)],flush=True)
        if end==12:hist.extend(rows)
        if end==14:
            g=np.array(result['candidate'])-result['base']
            w[1]*=(g>0)
            if not np.any(w[1]):break

if __name__=='__main__':main()
