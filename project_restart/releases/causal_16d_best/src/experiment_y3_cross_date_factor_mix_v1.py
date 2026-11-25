"""Train a joint panel model with stock-relative and market-factor examples paired across historical sessions."""
import copy
import json
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_y3_polynomial_patch_gru_v1 as patch
import y3_flow_exchange_references as history

base,parent=history.base,history.incumbent
OUT=parent.OUT.parent/'y3_cross_date_factor_mix_v1'
NAMES=('unaugmented','within-date','between-date')
ARMS=('unchanged','unaugmented full','unaugmented far','within-date full','between-date full',
      'between-date far','augmented minus unaugmented','between minus within','relative augmentation',
      'relative versus within','common augmentation','common versus within','augmented relative forecast',
      'augmented market forecast','agreement of augmentation contrasts')


def features(raw,t):
    x,h,vol=patch.prepare(raw,t)
    coarse=h[:,:,:,0].reshape(len(t),50,-1)
    # The tilt of the fastest and slowest patches represents recent movement versus slow drift.
    tilt=h[:,:,::3,1].reshape(len(t),50,-1)
    clock=np.where(t<2401,t,t-2401)
    phase=np.stack((np.exp(-clock/300),np.sin(2*np.pi*clock/20),np.cos(2*np.pi*clock/20)),1).astype('float32')
    return np.concatenate((x,coarse,tilt,np.log(vol)[:,:,None]),2).astype('float16'),phase


def examples(day):
    raw=history.trial.clock.book.rawday(day)
    y=history.trial.clock.labels(day).reshape(-1,50,3)
    t=np.concatenate([np.arange(lo+(day*3)%10,hi-60,10) for lo,hi in base.segments(len(raw))])
    x,ph=features(raw,t)
    target=np.stack((y[t,:,0],y[t,:,1],y[t,:,2],y[t,:,2]-base.RATIO*y[t,:,1]),2)
    return x,ph,target.astype('float32'),np.full(len(t),day),t


class FactorNet(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.local=nn.Sequential(nn.Linear(n,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.13),nn.Linear(96,48))
        self.risk=nn.Sequential(nn.Linear(2*n+3,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.15),nn.Linear(96,48))
        self.stock=nn.Embedding(50,12)
        self.relative=nn.Sequential(nn.Linear(108,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.15),nn.Linear(64,4))
        self.market=nn.Sequential(nn.Linear(48,32),nn.LayerNorm(32),nn.SiLU(),nn.Linear(32,4))

    def forward(self,local_x,market_x,phase):
        x=local_x.float()
        z=market_x.float()
        # Synthetic pairs are formed only after subtracting each input's own contemporaneous market.
        a=self.local(x-x.mean(1,keepdim=True))
        b=self.risk(torch.cat((z.mean(1),z.std(1,unbiased=False),phase),1))
        h=torch.cat((a,b[:,None].expand_as(a),self.stock.weight[None].expand(len(a),-1,-1)),2)
        r=self.relative(h)
        return r-r.mean(1,keepdim=True),self.market(b)


def day_betas(y):
    m=y.mean(1)
    # Modest shrinkage towards unit exposure; stock exposures average to one per horizon.
    v=m-m.mean(0)
    beta=((y-y.mean(0))*v[:,None]).sum(0)/v.square().sum(0).clamp_min(1e-6)
    return .65+.35*beta


def partner_indices(day,t,mode,rng):
    if mode==0: return np.arange(len(t))
    bin_id=t//120
    out=np.empty(len(t),int)
    dates=np.unique(day)
    for d in dates:
        for b in np.unique(bin_id[day==d]):
            idx=np.flatnonzero((day==d)&(bin_id==b))
            keep=(day!=d) if mode==2 else (day==d)
            pool=np.flatnonzero(keep&(bin_id==b))
            if not len(pool):pool=np.flatnonzero(keep)
            if mode==1 and len(pool)==1:out[idx]=pool[0]
            else:out[idx]=rng.choice(pool,size=len(idx))
    return out


def fit(rows):
    cols=[np.concatenate(a) for a in zip(*rows)]
    x,ph,y=[torch.from_numpy(a).cuda() for a in cols[:3]]
    day,t=cols[3:]
    beta=torch.stack([day_betas(y[torch.from_numpy(np.flatnonzero(day==k)).cuda()]) for k in np.unique(day)]).mean(0)
    m=y.mean(1)
    rel=y-beta[None]*m[:,None]
    models=[]
    for mode in range(3):
        torch.manual_seed(52881)
        rng=np.random.default_rng(16121)
        net=FactorNet(x.shape[-1]).cuda()
        ema=copy.deepcopy(net).eval()
        opt=torch.optim.AdamW(net.parameters(),lr=.0009,weight_decay=.045)
        for epoch in range(7):
            if epoch in (4,6):opt.param_groups[0]['lr']*=.55
            partner=torch.from_numpy(partner_indices(day,t,mode,rng)).cuda()
            for ix in torch.randperm(len(x),device='cuda').split(40):
                # Keep half the batch intact to anchor predictions on actual market states.
                mate=partner[ix].clone();mate[::2]=ix[::2]
                r,c=net(x[ix],x[mate],ph[mate])
                goal=rel[ix]+beta[None]*m[mate,None]
                q=r+beta[None]*c[:,None]
                def err(a,b):return F.smooth_l1_loss(a,b,beta=.65)
                loss=.45*err(r,rel[ix])+.40*err(c,m[mate])+.50*err(q[:,:,2],goal[:,:,2])+.45*err(q[:,:,3],goal[:,:,3])
                opt.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.)
                opt.step()
                with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.05)
        models.append(ema)
    return models,beta.detach().cpu().numpy()


@torch.inference_mode()
def predict(raw,models,beta):
    x,ph=features(raw,np.arange(len(raw)))
    blocks=[]
    for start in range(0,len(raw),96):
        v=torch.from_numpy(x[start:start+96]).cuda()
        phase=torch.from_numpy(ph[start:start+96]).cuda()
        rss,ms,full=[],[],[]
        for net in models:
            r,m=net(v,v,phase)
            ri=r.cpu().numpy();mi=m.cpu().numpy()[:,None]*beta
            rss.append(ri);ms.append(mi);full.append(ri+mi)
        a,b,c=full
        d1=c[:,:,2]-a[:,:,2];d2=c[:,:,2]-b[:,:,2]
        z=np.stack((a[:,:,2],a[:,:,3],b[:,:,2],c[:,:,2],c[:,:,3],d1,d2,rss[2][:,:,2]-rss[0][:,:,2],
                    rss[2][:,:,2]-rss[1][:,:,2],ms[2][:,:,2]-ms[0][:,:,2],ms[2][:,:,2]-ms[1][:,:,2],
                    rss[2][:,:,2],ms[2][:,:,2],np.where(d1*d2>0,np.sign(d1)*np.minimum(abs(d1),abs(d2)),0.)),2)
        blocks.append(z)
    return np.concatenate(blocks)


def train():
    OUT.mkdir(exist_ok=True)
    rows,grams,folds,daily=[],[],[],[]
    for d in range(16):
        if d in (4,8,12):
            nets,b=fit(rows)
            if d>=8:
                w=base.calibrate(grams);fold=[]
        if d>=4:
            raw=history.trial.clock.book.rawday(d)
            y=history.trial.clock.labels(d).reshape(-1,50,3)
            ps=history.references(d,raw) if d>=8 else parent.parent.warm.read(d,raw)
            z=predict(raw,nets,b)
            if d==4:np.testing.assert_allclose(z[:111],predict(raw[:111],nets,b),rtol=0,atol=8e-6)
            r=[base.records(p,y,z) for p in ps];grams.append(r)
            if d>=8:
                fold.append(r);daily.append([base.score(m,w) for m in r])
                if d in (11,15):folds.append([base.score(m,w) for m in np.sum(fold,axis=0)])
        rows.append(examples(d))
        if d%4==3: print('complete through date',d+1,flush=True)
    f=np.asarray(folds);days=np.asarray(daily)
    good=[i for i in range(1,len(ARMS)) if f[:,:,0,i].min()>0 and (days[:,:,0,i]>0).sum(0).min()>=5]
    pick=max(good,key=lambda i:f[:,:,0,i].min()) if good else 0
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,selected=pick,folds=f.tolist(),daily=days.tolist()),indent=2))
    print('selected',ARMS[pick],f[:,:,0].round(6),flush=True)
    if pick:
        nets,b=fit(rows)
        np.savez(OUT/'model.npz',n=rows[0][0].shape[-1],b=b,pick=pick-1,coef=base.calibrate(grams)[pick-1],
                 **{str(i)+'_'+k:v.cpu().numpy() for i,m in enumerate(nets) for k,v in m.state_dict().items()})
    return bool(pick)


def correction():
    with np.load(OUT/'model.npz') as f:
        nets=[FactorNet(int(f['n'])).cuda().eval() for _ in NAMES]
        for i,m in enumerate(nets):m.load_state_dict({k:torch.from_numpy(f[str(i)+'_'+k]) for k in m.state_dict()})
        b,idx,coef=f['b'],int(f['pick']),float(f['coef'])
    return lambda raw:(predict(raw,nets,b)[:,:,idx]*coef).astype('float32')


def layer():
    before,delta=parent.layer(),correction()
    def predict_all(raw,initial):
        p=before(raw,initial).copy();p[:,:,2]+=delta(raw)
        return p
    return predict_all


def replay_with():
    delta=correction()
    def read(d,raw):
        p,h,n=history.read(d,raw);p[:,:,2]+=delta(raw);h[:,:,2]+=delta(raw[:n])
        return p,h,n
    return history.stack.verifier,read


if __name__=='__main__' and (sys.argv[1:]==['replay'] or train()):
    base.replay.replay(sys.modules[__name__])
