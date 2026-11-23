"""Train a numerical spline encoder on stock-specific book distributions and causal rolling context."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_quote_event_gru_v1 as event
import experiment_peer_return_distillation_v1 as fitting
import experiment_pooled_price_impact_v1 as impact
import experiment_multiscale_return_net_v1 as base
import experiment_endpoint_innovation_v1 as inn
OUT=inn.prior.ART/'book_spline_network_v1'
CUTS=[.03,.12,.28,.5,.72,.88,.97]
FIELDS=[0,5,7,10,11,12,13,14,16,18,20,21,22,23,24,25,26,27,28,29,30,31,32,34,36,37,38,39,40,41,42,43,46,48,50,51,52,53,54,55,56,57,60,64,68,72,74,75,76,77,78,79,80,81,82]
def geometry(end):
    x=np.concatenate([np.arcsinh(inn.grid(d)[d%23::23,:,:83].astype(np.float32)) for d in range(end)])
    global_c=np.quantile(x.reshape(-1,83),CUTS,axis=0)[:,None,:]
    stock_c=np.quantile(x,CUTS,axis=0)
    cuts=.75*stock_c+.25*global_c
    return np.moveaxis(cuts,0,-1).astype(np.float32)

def features(raw,t):
    r=raw[:,:,0].astype(float)-.00136;n=len(raw)
    state=np.arcsinh(raw[:,:,FIELDS].astype(np.float32))
    total=np.concatenate((np.zeros_like(state[:1],dtype=float),state.cumsum(0,dtype=float)))
    parts=[np.arcsinh(raw[t,:,:83].astype(np.float32)),raw[t,:,83:].astype(np.float32)-1]
    for h in (5,24,120):
        left=np.maximum(0,t+1-h)
        avg=(total[t+1]-total[left])/(t+1-left)[:,None,None]
        parts.append((state[t]-avg).astype(np.float32))
    v=np.sqrt((8+inn.moving_sum(r*r,96))/(np.minimum(np.arange(n)+1,96)+8)[:,None]).clip(.45,4)
    tail=[np.log(v[t])]
    for h in (5,20,60,240):
        ret=inn.moving_sum(r,h)[t]/np.sqrt(h)
        tail.extend([np.arcsinh(ret),np.arcsinh((ret-ret.mean(1,keepdims=True))/v[t])])
    tail.append(np.broadcast_to(np.exp(-t/300)[:,None],v[t].shape))
    parts.append(np.stack(tail,axis=2).astype(np.float32))
    return np.concatenate(parts,axis=2).clip(-10,10).astype(np.float16),v[t].astype(np.float32)

class SplineNet(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.register_buffer('cuts',torch.zeros(50,83,len(CUTS)))
        self.register_buffer('port',torch.zeros(4,50));self.register_buffer('beta',torch.ones(50))
        self.current=nn.Linear(83*(len(CUTS)-1),80,bias=False)
        self.hist=nn.Linear(n,80)
        self.block=nn.Sequential(nn.LayerNorm(80),nn.Linear(80,96),nn.SiLU(),nn.Dropout(.12),nn.Linear(96,80))
        self.msg=nn.Linear(80,32)
        self.local=nn.Sequential(nn.Linear(144,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.16),nn.Linear(96,3))
        self.market=nn.Sequential(nn.Linear(192,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.2),nn.Linear(64,3))
    def forward(self,x,vol):
        c=self.cuts
        b=((x[:,:,:83,None]-c[:,:,:-1])/(c[:,:,1:]-c[:,:,:-1]).clamp_min(.045)).clamp(0,1)-.5
        h=self.current(b.flatten(2))+self.hist(x);h=h+self.block(h)
        v=self.msg(h);m=v.mean(1,keepdim=True);f=self.port@v
        r=self.local(torch.cat((h,self.port.T@f,m.expand_as(v)),2))*vol[:,:,None]
        r=r-r.mean(1,keepdim=True)
        market=self.market(torch.cat((m[:,0],v.std(1,unbiased=False),f.flatten(1)),dim=1))*vol.mean(1,keepdim=True)
        return r+self.beta[None,:,None]*market[:,None],r,market

def train(end):
    torch.manual_seed(7461)
    sel=[np.arange((d*5+1)%4,len(inn.grid(d)),4) for d in range(end)]
    size=sum(len(t) for t in sel);pos=0
    y=torch.empty(size,50,3,device='cuda');vol=torch.empty(size,50,device='cuda')
    for d,t in enumerate(sel):
        a,v=features(inn.grid(d),t)
        if d==0:data=torch.empty(size,50,a.shape[-1],dtype=torch.float16,device='cuda')
        sl=slice(pos,pos+len(t));pos+=len(t)
        data[sl].copy_(torch.from_numpy(a));vol[sl].copy_(torch.from_numpy((.45+.55*v).clip(.55,3)))
        y[sl].copy_(torch.from_numpy(inn.prior.labels(d).reshape(-1,50,3)[t]))
    model=SplineNet(data.shape[-1]).cuda();model.cuts.copy_(torch.from_numpy(geometry(end)))
    model.port.copy_(torch.from_numpy(impact.exposure(end)[0][-4:]*np.sqrt(50)))
    model.beta.copy_(torch.from_numpy(inn.base.ex.geometry(end)[0]))
    ema=SplineNet(data.shape[-1]).cuda();ema.load_state_dict(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=.00085,weight_decay=.025)
    for ep in range(1,17):
        for ids in torch.randperm(size,device='cuda').split(64):
            yy=y[ids];v=vol[ids];p,r,m=model(data[ids].float(),v)
            ym=yy.mean(1);yr=yy-model.beta[None,:,None]*ym[:,None]
            loss=F.smooth_l1_loss(r/v[:,:,None],yr/v[:,:,None],beta=1.2)+.6*F.smooth_l1_loss(m/v.mean(1,keepdim=True),ym/v.mean(1,keepdim=True),beta=.45)
            loss-=.08*F.cosine_similarity(r,yr,dim=1,eps=1e-5).mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),.018)
        if ep in (6,11):
            for g in opt.param_groups:g['lr']*=.55
        if ep in (1,8,16):print('train',end,ep,flush=True)
    torch.save(ema.state_dict(),OUT/f'net_{end}.pth');return ema.eval()

@torch.inference_mode()
def infer(net,raw):
    n=len(raw);out=np.empty((n,50,3),np.float32)
    for k in range(4):
        t=np.arange(k,n,4);a,v=features(raw,t)
        for i in range(0,len(t),64):
            x=torch.from_numpy(a[i:i+64]).cuda().float();vol=torch.from_numpy(v[i:i+64]).cuda()
            out[t[i:i+64]]=net(x,(.45+.55*vol).clamp(.55,3))[0].cpu().numpy()
    return out

def main():
    OUT.mkdir(exist_ok=True);early=[];report={}
    for end,stop in ((12,14),(14,16),(16,20)):
        path=OUT/f'net_{end}.pth'
        if path.exists():
            state=torch.load(path,map_location='cpu',weights_only=True)
            net=SplineNet(state['hist.weight'].shape[1]).cuda().eval();net.load_state_dict(state)
        else:net=train(end)
        rows=[]
        for d in range(end,stop):
            raw=inn.grid(d);n=infer(net,raw);p=np.load(event.REFERENCE/f'pred_{d}.npy')
            if d==end:np.testing.assert_allclose(n[:145],infer(net,raw[:145]),atol=1e-4)
            rows.append((p,inn.prior.labels(d).reshape(p.shape),n))
        if end==12:_,w=fitting.coefficients(rows)
        entries=[]
        for d,(p,y,n) in zip(range(end,stop),rows):
            q=base.blend(p,n,w)
            np.save(OUT/f'side_{d}.npy',n);np.save(OUT/f'pred_{d}.npy',q.astype(np.float32))
            entries.append(tuple(a.reshape(-1,3) for a in (p,q,y)))
        val=inn.prior.measure(entries);report[stop]={'weights':w.tolist(),'result':val}
        (OUT/'results.json').write_text(json.dumps(report,indent=2));print(stop,report[stop],flush=True)
        if end<16:early.extend(rows)
        if end==14:
            if val['gain']<=0:break
            _,w=fitting.coefficients(early)
if __name__=='__main__':main()
