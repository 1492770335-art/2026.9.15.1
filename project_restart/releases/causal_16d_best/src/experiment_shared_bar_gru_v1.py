"""Train one fixed five-bar return predictor across different causal aggregations of the input stream."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_phase_future_book_v1 as phase
import experiment_variance_ratio_calibration_v1 as prior_model

OUT=phase.inn.prior.ART/'shared_bar_gru_v1'
FIELDS=(7,10,11,14,16,23,26,34,38,41,42,51,54,55,60,64,72,74)
BARS=6
SCALES=(1,2,4,8,12)
N_BAR=len(FIELDS)+4


class Day:
    def __init__(self,raw):
        self.raw=raw
        r=raw[:,:,0].astype(float)-.00136
        self.price=np.concatenate((np.zeros_like(r[:1]),r.cumsum(0)))
        self.energy=np.concatenate((np.zeros_like(r[:1]),(r*r).cumsum(0)))
        self.active=np.concatenate((np.zeros_like(r[:1]),(abs(r)>.018).cumsum(0)))
        a=np.arcsinh(raw[:,:,FIELDS].astype(np.float32))
        self.cs=np.concatenate((np.zeros_like(a[:1]),a.cumsum(0,dtype=float)))
        self.vol=np.sqrt((10+phase.inn.moving_sum(r*r,128))/(10+np.minimum(np.arange(len(r))+1,128))[:,None]).clip(.45,4.).astype(np.float32)

    def inputs(self,t,w):
        now=np.arcsinh(self.raw[t,:,:83].astype(np.float32))
        pieces=[now,self.raw[t,:,83:].astype(np.float32)-1.]
        v=self.vol[t]
        for k in reversed(range(BARS)):
            hi=np.maximum(t+1-w*k,0);lo=np.maximum(t+1-w*(k+1),0)
            length=(hi-lo).clip(1)[:,None]
            ret=(self.price[hi]-self.price[lo])/(np.sqrt(length)*v)
            energy=np.sqrt((self.energy[hi]-self.energy[lo])/length)/v
            hit=(self.active[hi]-self.active[lo])/length
            book=(self.cs[hi]-self.cs[lo])/length[:,:,None]
            valid=np.broadcast_to(((hi-lo)/w)[:,None],v.shape)
            pieces.extend([book,np.stack((np.arcsinh(ret),np.log1p(energy),hit,valid),axis=2)])
        return np.concatenate(pieces,axis=2).clip(-8,8).astype(np.float16),v


class ScaleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bar=nn.Sequential(nn.Linear(N_BAR,28),nn.LayerNorm(28),nn.SiLU())
        self.gru=nn.GRU(28,32,batch_first=True)
        self.current=nn.Sequential(nn.Linear(100,72),nn.LayerNorm(72),nn.SiLU())
        self.encoder=nn.Sequential(nn.Linear(72+32+3,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.12))
        self.identity=nn.Parameter(torch.zeros(50,96))
        self.msg=nn.Linear(96,20)
        self.local=nn.Sequential(nn.Linear(136,72),nn.LayerNorm(72),nn.SiLU(),nn.Dropout(.13),nn.Linear(72,2))
        self.market=nn.Sequential(nn.Linear(123,72),nn.LayerNorm(72),nn.SiLU(),nn.Dropout(.18),nn.Linear(72,2))
        self.register_buffer('port',torch.zeros(4,50));self.register_buffer('beta',torch.ones(50))
    def forward(self,x,v,meta):
        grid=x[:,:,100:].reshape(-1,BARS,N_BAR)
        seq=self.bar(grid)
        tail=self.gru(seq)[1][0].reshape(len(x),50,32)
        ctx=meta[:,None,:].expand(-1,50,-1)
        z=self.encoder(torch.cat([self.current(x[:,:,:100]),tail,ctx],2))+self.identity
        msg=self.msg(z);market=msg.mean(1,keepdim=True);fac=self.port@msg
        a,b=self.local(torch.cat([z,market.expand_as(msg),self.port.T@fac],2)).unbind(-1)
        loc=(a+.45*(meta[:,0:1]-.5)*b)*v
        loc-=loc.mean(1,keepdim=True)
        q,k=self.market(torch.cat([market[:,0],msg.std(1,unbiased=False),fac.flatten(1),meta],1)).unbind(-1)
        common=(q+.45*(meta[:,0]-.5)*k)*v.mean(1)
        return loc,common


def meta(t,w):
    return np.stack((np.full(len(t),np.log(w)/np.log(12)),np.exp(-t/350),
                     np.exp(-t/(10*w))),axis=1).astype(np.float32)


def train(end):
    torch.manual_seed(7221)
    rng=np.random.default_rng(7221)
    count=0
    total=end*5*318
    data=torch.empty(total,50,100+BARS*N_BAR,device='cuda',dtype=torch.float16)
    vol=torch.empty(total,50,device='cuda');out=torch.empty_like(vol);ctx=torch.empty(total,3,device='cuda')
    for d in range(end):
        raw=phase.inn.grid(d);day=Day(raw);y=phase.inn.prior.labels(d).reshape(-1,50,3);n=len(raw)
        for w in SCALES:
            h=5*w
            t=np.sort(np.r_[rng.choice(600,95,replace=False),rng.choice(np.arange(600,n-h),223,replace=False)])
            x,v=day.inputs(t,w);sl=slice(count,count+len(t));count+=len(t)
            if w in (1,4,12):
                j=(1,4,12).index(w);target=y[t,:,j]/(phase.inn.SCALE[j]*np.sqrt(h))
            else:
                target=(day.price[t+1+h]-day.price[t+1])/np.sqrt(h)
            data[sl].copy_(torch.from_numpy(x));vol[sl].copy_(torch.from_numpy((.45+.55*v).clip(.55,3.)))
            out[sl].copy_(torch.from_numpy(target));ctx[sl].copy_(torch.from_numpy(meta(t,w)))
    net=ScaleNet().cuda();net.port.copy_(torch.from_numpy(phase.impact.exposure(end)[0][-4:]*np.sqrt(50)))
    net.beta.copy_(torch.from_numpy(phase.inn.base.ex.geometry(end)[0]))
    ema=ScaleNet().cuda();ema.load_state_dict(net.state_dict())
    opt=torch.optim.AdamW(net.parameters(),lr=.0009,weight_decay=.025)
    for ep in range(1,13):
        for ids in torch.randperm(count,device='cuda').split(72):
            y=out[ids];ym=y.mean(1);yr=y-net.beta*ym[:,None];v=vol[ids]
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                loc,m=net(data[ids].float(),v,ctx[ids])
            loss=F.smooth_l1_loss(loc/v,yr/v,beta=.95)+.58*F.smooth_l1_loss(m/v.mean(1),ym/v.mean(1),beta=.40)
            loss-=.048*F.cosine_similarity(loc,yr,dim=1,eps=1e-5).mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.015)
        if ep in (5,9):
            for g in opt.param_groups:g['lr']*=.53
        if ep in (1,6,12):print('training',end,ep,flush=True)
    torch.save(ema.state_dict(),OUT/f'net_{end}.pth')
    return ema.eval()


@torch.inference_mode()
def predict(net,raw):
    d=Day(raw);out=np.empty((len(raw),50,3),np.float32)
    for j,w in enumerate((1,4,12)):
        for i in range(0,len(raw),72):
            t=np.arange(i,min(i+72,len(raw)));x,v=d.inputs(t,w)
            loc,m=net(torch.from_numpy(x).cuda().float(),(.45+.55*torch.from_numpy(v).cuda()).clamp(.55,3),
                      torch.from_numpy(meta(t,w)).cuda())
            out[t,:,j]=(loc+net.beta*m[:,None]).cpu().numpy()*phase.inn.SCALE[j]*np.sqrt(5*w)
    return out


def main():
    OUT.mkdir(exist_ok=True);report={};hist=[]
    for end,stop in ((12,14),(14,16),(16,20)):
        ck=OUT/f'net_{end}.pth'
        if ck.exists():
            net=ScaleNet().cuda().eval();net.load_state_dict(torch.load(ck,weights_only=True))
        else:net=train(end)
        rows=[]
        for d in range(end,stop):
            raw=phase.inn.grid(d);n=predict(net,raw)
            if d==end:np.testing.assert_allclose(n[:143],predict(net,raw[:143]),atol=1e-4)
            p=prior_model.baseline(d);y=phase.inn.prior.labels(d).reshape(p.shape)
            np.save(OUT/f'side_{d}.npy',n);rows.append((p,y,n))
        if end==12:_,w=phase.fits.coefficients(rows)
        entries=[]
        for d,(p,y,n) in zip(range(end,stop),rows):
            q=phase.v1.blend(p,n,w).astype('float32')
            np.save(OUT/f'pred_{d}.npy',q);entries.append(tuple(a.reshape(-1,3) for a in (p,q,y)))
        val=phase.inn.prior.measure(entries);report[str(stop)]={'result':val,'w':w.tolist()}
        (OUT/'results.json').write_text(json.dumps(report,indent=2));print(stop,report[str(stop)],flush=True)
        if end<16:hist.extend(rows)
        if end==14:
            if val['gain']<=0:break
            _,w=phase.fits.coefficients(hist)

if __name__=='__main__':main()
