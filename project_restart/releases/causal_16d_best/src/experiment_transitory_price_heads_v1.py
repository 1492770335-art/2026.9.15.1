"""Compare direct return heads with an exact drift/transitory decomposition using smoothed training targets."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_consensus_horizon_portfolios_v1 as blend
import experiment_auxiliary_quote_decoder_v1 as decoder
import experiment_phase_future_book_v1 as phase
import experiment_full_coverage_book_v1 as cache
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure
from experiment_replicated_revision_slope_v1 import day_weights

OUT=ART/'transitory_price_heads_v1'
HORIZONS=np.array([5,20,60])
HALF=16


def targets(raw,t,y):
    price=np.cumsum(raw[:,:,0].astype(float)-.00136,axis=0)
    prefix=np.vstack((np.zeros(50),price.cumsum(0)))
    s=np.arange(len(price));left=(s-HALF).clip(0);right=(s+HALF+1).clip(0,len(price))
    displacement=.25*((prefix[right]-prefix[left])/(right-left)[:,None]-price)
    current=displacement[t,:,None]
    terminal=np.stack([displacement[t+h] for h in HORIZONS],axis=2)
    trend=y-(current-terminal)*(blend.UNITS/.25)
    return np.concatenate((trend,current,terminal),axis=2).astype('float32')


@torch.inference_mode()
def encoding(raw,t,model,cache_day=None,fixed_batch=False):
    parts=[];vols=[]
    if cache_day is not None:
        cache.prepare(cache_day);root=cache.CACHE/f'day_{cache_day:02d}'
        fx=np.load(root/'x.npy',mmap_mode='r');fv=np.load(root/'v.npy',mmap_mode='r')
    latent=[]
    hook=model.reg.register_forward_hook(lambda mod,args,v:latent.append(v))
    try:
        for start in range(0,len(t),800):
            ids=t[start:start+800]
            x,v=(fx[ids],fv[ids]) if cache_day is not None else phase.spline.features(raw,ids)
            risk=(.45+.55*v).clip(.55,3)
            for j in range(0,len(ids),64):
                sl=slice(j,j+64);a=ids[sl]
                batch_x=x_to_tensor(x[sl])
                age=torch.from_numpy(np.exp(-a[:,None]/400).astype('float32')).cuda()
                r=torch.from_numpy(risk[sl]).cuda()
                if fixed_batch and len(a)<64:
                    # Keep the same GEMM shapes for a prefix and a full session before half rounding.
                    def pad(v):
                        return torch.cat((v,v[-1:].expand(64-len(a),*v.shape[1:])))
                    batch_x,r,age=map(pad,(batch_x,r,age))
                pred=model(batch_x,r,age)[0]
                h=latent.pop()
                features=torch.cat((h,pred,batch_x[:,:,:31],age[:,None].expand(-1,50,-1)),dim=2)
                parts.append(features[:len(a)].half());vols.append(r[:len(a)])
    finally:hook.remove()
    return torch.cat(parts),torch.cat(vols)


def x_to_tensor(x):
    return torch.from_numpy(x).cuda().float()


class Head(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(n,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.16),nn.Linear(64,40),nn.SiLU())
        self.stock=nn.Sequential(nn.Linear(80,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,7))
        self.market=nn.Sequential(nn.Linear(80,48),nn.LayerNorm(48),nn.SiLU(),nn.Dropout(.16),nn.Linear(48,7))
        self.register_buffer('scale',torch.from_numpy(blend.UNITS.copy()/.25))

    def forward(self,x,vol):
        h=self.enc(x.float());m=h.mean(1)
        local=self.stock(torch.cat((h,m[:,None].expand_as(h)),dim=2)).float()*vol[:,:,None]
        shared=self.market(torch.cat((m,h.std(1,unbiased=False)),dim=1)).float()*vol.mean(1,keepdim=True)
        return local-local.mean(1,keepdim=True)+shared[:,None]

    def return_from(self,p,decompose):
        if decompose:return p[:,:,:3]+(p[:,:,3:4]-p[:,:,4:])*self.scale
        return p[:,:,:3]


def criterion(p,y,v):
    m=p.mean(1);ym=y.mean(1)
    return (F.smooth_l1_loss((p-m[:,None])/v[:,:,None],(y-ym[:,None])/v[:,:,None],beta=1.15)
            +.52*F.smooth_l1_loss(m/v.mean(1,keepdim=True),ym/v.mean(1,keepdim=True),beta=.52))


def training_set(end,encoder):
    blocks=[[] for _ in range(4)]
    for d in range(end):
        raw=phase.inn.grid(d)
        t=np.r_[np.arange(d%4,600,4),np.arange(600+d%8,len(raw)-61-HALF,8)]
        obs=labels(d).reshape(-1,50,3)[t];latent,v=encoding(raw,t,encoder,d)
        for dst,a in zip(blocks,(latent,v,torch.from_numpy(obs).cuda(),torch.from_numpy(targets(raw,t,obs)).cuda())):
            dst.append(a)
    return tuple(torch.cat(b) for b in blocks)


def fit(data,decompose):
    torch.manual_seed(50715)
    x,v,truth,parts=data
    model=Head(x.shape[2]).cuda();ema=Head(x.shape[2]).cuda();ema.load_state_dict(model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.035)
    for epoch in range(13):
        for ids in torch.randperm(len(x),device='cuda').split(64):
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                p=model(x[ids],v[ids])
            q=model.return_from(p,decompose)
            loss=criterion(q,truth[ids],v[ids])
            if decompose:
                loss+=.48*criterion(p[:,:,:3],parts[ids,:,:3],v[ids])
                loss+=.42*criterion(p[:,:,3:],parts[ids,:,3:],v[ids])
            loss-=.045*F.cosine_similarity(q-q.mean(1,keepdim=True),truth[ids]-truth[ids].mean(1,keepdim=True),dim=1).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),.022)
        if epoch in (5,9):
            for group in optimizer.param_groups:group['lr']*=.55
    return ema.eval()


@torch.inference_mode()
def infer(x,v,model,decompose):
    return torch.cat([model.return_from(model(x[k:k+72],v[k:k+72]),decompose).cpu() for k in range(0,len(x),72)]).numpy()


def mix(p,side,w):
    delta=side-p;m=delta.mean(1,keepdims=True)
    return (p+m*w[0]+(delta-m)*w[1]).astype('float32')


def main():
    OUT.mkdir(exist_ok=True);report={'folds':[]};hist=[[],[]];priors=[];gains=[]
    for end,stop in ((8,12),(12,14),(14,16)):
        encoder=decoder.net(end);data=training_set(end,encoder)
        models=[fit(data,bool(i)) for i in range(2)];del data
        w=[.68*np.median(h,axis=0).clip(0,.45) if h else None for h in hist]
        c=blend.fixed_weight(priors);rows=[[],[]]
        for d in range(end,stop):
            p,y,fs,stat=blend.training_day(d,priors);priors.append(stat)
            p=blend.parent.fixed_forecast(p,fs,c)
            raw=phase.inn.grid(d);t=np.arange(d%3,len(p),3);p=p[t];y=y[t]
            x,v=encoding(raw,t,encoder)
            for i,model in enumerate(models):
                side=infer(x,v,model,bool(i));q=mix(p,side,w[i]) if w[i] is not None else p
                if d==end:
                    small=t[t<193];xx,vv=encoding(raw[:193],small,encoder)
                    np.testing.assert_allclose(infer(xx,vv,model,bool(i)),side[:len(small)],rtol=0,atol=.0001)
                row=tuple(a.reshape(-1,3) for a in (p,q,y));rows[i].append((row,side.reshape(-1,3)))
                if w[i] is not None and i==1:
                    s=measure([row]);gains.append(np.subtract(s['candidate'],s['base']))
                hist[i].append(day_weights(p,side,y))
        fold={}
        for name,group in zip(('direct','decomposed'),rows):
            fold[name]={'combined':measure([r for r,_ in group]),'standalone':measure([(r[0],f,r[2]) for r,f in group])}
        report['folds'].append(fold);print('days',end+1,stop,fold,flush=True)
        (OUT/'results.json').write_text(json.dumps(report,indent=2))
    def gains_for(a,key,b,other):
        return np.array([np.subtract(f[a][key]['candidate'],f[b][other]['candidate']) for f in report['folds'][1:]])
    daily=np.stack(gains)
    gains=np.array([np.subtract(f['decomposed']['combined']['candidate'],f['decomposed']['combined']['base']) for f in report['folds'][1:]])
    better=gains_for('decomposed','combined','direct','combined').mean(0)>0
    gate=(gains.min(0)>0)&((daily>0).sum(0)>=3)&better
    report['retained']=gate.tolist();path=OUT/'results.json';path.write_text(json.dumps(report,indent=2))
    if not gate.any():return
    encoder=decoder.net(16);data=training_set(16,encoder);model=fit(data,True);del data
    w=.68*np.median(hist[1],axis=0).clip(0,.45)*gate
    torch.save(model.state_dict(),OUT/'head.pth');np.save(OUT/'weights.npy',w)
    from experiment_replicated_expert_consensus_v1 import sources
    from experiment_predictive_book_modes_v1 import validation
    experts=sources();c=np.load(blend.PARENT/'weights.npy')[0]
    def advance(raw,p):
        p=blend.parent.fixed_forecast(p,[fn(raw) for fn in experts],c);q=p.copy()
        for k in range(4):
            t=np.arange(k,len(p),4)
            if len(t):
                x,v=encoding(raw,t,encoder);q[t]=mix(p[t],infer(x,v,model,True),w)
        return q
    result=validation(advance)
    ref=json.loads((blend.PARENT/'raw_replay.json').read_text())
    result['gain_over_fixed']=result['pooled']['score']-ref['pooled']['score']
    report['development']=result;path.write_text(json.dumps(report,indent=2));print(result,flush=True)


if __name__=='__main__':main()
