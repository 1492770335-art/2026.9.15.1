"""Train an event-time GRU with per-stock quote histories and synchronous portfolio information."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_endpoint_innovation_v1 as inn
import experiment_pooled_price_impact_v1 as impact
import experiment_peer_return_distillation_v1 as fitting
import experiment_multiscale_return_net_v1 as base

OUT=inn.prior.ART/'quote_event_gru_v1'
REFERENCE=inn.prior.ART/'curve_quantile_market_v1'
LAGS=np.array([48,32,20,12,8,5,3,2,1,0])

def inputs(raw):
    time=np.arange(len(raw))
    quote=raw[:,:,[0,20,23,26,38,39,41,42]]
    changed=np.any(quote[1:]!=quote[:-1],axis=2)
    event=np.vstack((np.ones((1,50),bool),changed))
    count=event.cumsum(axis=0)-1
    tick,code=np.nonzero(event)
    locations=np.zeros((len(raw),50),np.int32)
    locations[count[tick,code],code]=tick
    indices=locations[np.maximum(0,count[:,:,None]-LAGS),np.arange(50)[None,:,None]]
    r=raw[:,:,0].astype(float)-.00136
    price=r.cumsum(0)
    vol=np.sqrt((5+inn.moving_sum(r*r,96))/(np.minimum(time+1,96)+5)[:,None]).clip(.35,4)
    spans=[]
    for h in (5,20,60,240):
        move=inn.moving_sum(r,h)/np.sqrt(h)
        spans.extend((np.arcsinh(move/vol),np.arcsinh(move.mean(1,keepdims=True)-move)))
    frame=np.concatenate((np.arcsinh(raw[:,:,:83].astype(np.float32)),raw[:,:,83:].astype(np.float32)-1,
                          np.stack(spans+[np.log(vol)],axis=-1)),axis=2).astype(np.float16)
    return tuple(torch.from_numpy(np.ascontiguousarray(a)).cuda() for a in (frame,indices,price.astype(np.float32),vol.astype(np.float32)))

def gather(data,t):
    frame,ids,price,vol=data
    ix=ids[t].long(); stocks=torch.arange(50,device=t.device)[None,:,None]
    elapsed=(t[:,None,None]-ix).float()
    own=price[t,:,None]-price[ix,stocks]
    m=price.mean(1); drift=m[t,None,None]-m[ix]
    context=torch.stack((torch.log1p(elapsed)/4,torch.asinh(own),torch.asinh(drift),
                         torch.asinh((drift-own)/torch.sqrt(1+elapsed))),dim=-1)
    events=torch.cat((frame[ix,stocks].float(),context),dim=-1)
    return frame[t].float(),events,vol[t].clamp(.5,3)

class EventNet(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(n+4,56),nn.LayerNorm(56),nn.SiLU())
        self.rnn=nn.GRU(56,64,batch_first=True)
        self.current=nn.Sequential(nn.Linear(n,64),nn.LayerNorm(64),nn.SiLU())
        self.message=nn.Linear(128,32)
        self.head=nn.Sequential(nn.LayerNorm(192),nn.Linear(192,112),nn.SiLU(),nn.Dropout(.16),nn.Linear(112,3))
        self.common=nn.Sequential(nn.Linear(160,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.22),nn.Linear(64,3))
        self.register_buffer('port',torch.zeros(4,50))
        self.register_buffer('beta',torch.ones(50))
    def forward(self,x,events,vol):
        b,s,k,_=events.shape
        encoded=self.encoder(events).reshape(b*s,k,-1)
        _,last=self.rnn(encoded)
        h=torch.cat((self.current(x),last[0].reshape(b,s,-1)),dim=-1)
        v=self.message(h);m=v.mean(1,keepdim=True);factors=self.port@v
        local=self.head(torch.cat((h,m.expand_as(v),self.port.T@factors),dim=-1))*vol[:,:,None]
        local=local-local.mean(1,keepdim=True)
        market=self.common(torch.cat((m[:,0],factors.flatten(1)),dim=1))*vol.mean(1,keepdim=True)
        return local+self.beta[None,:,None]*market[:,None,:],local,market

def train(end):
    rng=np.random.default_rng(539);torch.manual_seed(539)
    days=[inputs(inn.grid(d)) for d in range(end)]
    labels=[torch.from_numpy(inn.prior.labels(d).reshape(-1,50,3)).cuda() for d in range(end)]
    net=EventNet(days[0][0].shape[-1]).cuda()
    net.port.copy_(torch.from_numpy(impact.exposure(end)[0][-4:]*np.sqrt(50)))
    net.beta.copy_(torch.from_numpy(inn.base.ex.geometry(end)[0]))
    ema=EventNet(days[0][0].shape[-1]).cuda();ema.load_state_dict(net.state_dict())
    opt=torch.optim.AdamW(net.parameters(),lr=.00085,weight_decay=.025)
    for ep in range(1,13):
        batches=[]
        for d in range(end):
            ticks=rng.permutation(np.arange((ep+d)%8,len(labels[d]),8))
            batches.extend((d,ticks[k:k+56]) for k in range(0,len(ticks),56))
        rng.shuffle(batches)
        for d,t in batches:
            ix=torch.as_tensor(t,device='cuda')
            x,events,vol=gather(days[d],ix);y=labels[d][ix];ym=y.mean(1)
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                _,z,m=net(x,events,vol)
            rel=y-net.beta[None,:,None]*ym[:,None]
            loss=F.smooth_l1_loss(z/vol[:,:,None],rel/vol[:,:,None],beta=1.15)
            loss+=.55*F.smooth_l1_loss(m/vol.mean(1,keepdim=True),ym/vol.mean(1,keepdim=True),beta=.5)
            loss-=.055*F.cosine_similarity(z,rel,dim=1,eps=1e-5).mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.02)
        if ep in (5,9):
            for g in opt.param_groups:g['lr']*=.55
        if ep in (1,6,12):print('training days',end,'epoch',ep,flush=True)
    torch.save(ema.state_dict(),OUT/f'net_{end}.pth')
    return ema.eval()

@torch.inference_mode()
def infer(net,raw):
    data=inputs(raw)
    output=[]
    for t in torch.arange(len(raw),device='cuda').split(56):
        valid=len(t)
        if valid<56:t=torch.cat((t,t[-1:].expand(56-valid)))
        output.append(net(*gather(data,t))[0][:valid].cpu().numpy())
    return np.concatenate(output)

def main():
    OUT.mkdir(exist_ok=True);early=[];report={}
    for end,stop in ((12,14),(14,16),(16,20)):
        checkpoint=OUT/f'net_{end}.pth'
        if checkpoint.exists():
            net=EventNet(109).cuda().eval()
            net.load_state_dict(torch.load(checkpoint,weights_only=True))
        else:net=train(end)
        rows=[]
        for d in range(end,stop):
            raw=inn.grid(d);side=infer(net,raw)
            if d==end:np.testing.assert_allclose(side[:129],infer(net,raw[:129]),atol=1e-5)
            p=np.load(REFERENCE/f'pred_{d}.npy');y=inn.prior.labels(d).reshape(p.shape)
            rows.append((p,y,side))
        if end==12:_,w=fitting.coefficients(rows)
        entries=[]
        for d,(p,y,n) in zip(range(end,stop),rows):
            q=base.blend(p,n,w);entries.append(tuple(a.reshape(-1,3) for a in (p,q,y)))
            np.save(OUT/f'side_{d}.npy',n);np.save(OUT/f'pred_{d}.npy',q.astype(np.float32))
        val=inn.prior.measure(entries);report[stop]={'weights':w.tolist(),'result':val}
        (OUT/'results.json').write_text(json.dumps(report,indent=2));print(stop,report[stop],flush=True)
        if end<16:early.extend(rows)
        if end==14:
            if val['gain']<=0:break
            _,w=fitting.coefficients(early)
if __name__=='__main__':main()
