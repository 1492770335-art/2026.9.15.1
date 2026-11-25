"""Predict long-horizon returns with a GRU over disjoint book-history patches and adaptive stock portfolios."""
import copy
import json
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import y3_panel_message_origin_references as history
import experiment_y3_peer_arrival_alignment_v1 as book

parent,base=history.incumbent,history.base
OUT=parent.OUT.parent/'y3_polynomial_patch_gru_v1'
SPANS=((60,180),(20,60),(5,20),(0,5))
EXTRA=(16,21,25,34,39,52,60,64)
SEED=31357
ARMS=('unchanged','means-only far readout','shuffled derivatives far readout','full far readout',
      'shape beyond mean','shape beyond reassigned derivatives','relative shape beyond mean',
      'relative shape beyond reassigned derivatives','market shape beyond mean','market shape beyond reassigned',
      'full Y3 predicted from three parts','full Y3 shape beyond mean','full Y3 shape beyond reassigned',
      'late shape shared by both contrasts','late shape versus average of controls')


def prepare(raw,ticks):
    now=np.arcsinh(raw[ticks,:,:83].astype('float32'))
    cols=np.concatenate((now,raw[ticks,:,83:].astype('float32')-1),2)
    data=np.concatenate((book.signals(raw),np.tanh(np.arcsinh(raw[:,:,EXTRA].astype('float32')))),2)
    hist=np.empty((len(ticks),50,4,3,data.shape[2]),np.float16)
    volatility=np.empty((len(ticks),50),np.float32)
    for lo,hi in base.segments(len(raw)):
        valid=(ticks>=lo)&(ticks<hi)
        if not valid.any():continue
        t=ticks[valid]-lo+1
        a=data[lo:hi].astype(float)
        r=base.raw_returns(raw[lo:hi])
        pos=np.arange(len(a))[:,None,None]
        def prefix(z):
            return np.concatenate((np.zeros_like(z[:1]),z.cumsum(0)))
        p0,p1,p2=(prefix(v) for v in (a,pos*a,pos*pos*a))
        rv=prefix(r*r)
        start=(t-96).clip(0)
        volatility[valid]=np.sqrt((rv[t]-rv[start]+8)/((t-start+8)[:,None])).clip(.45,4)
        for j,(close,far) in enumerate(SPANS):
            right=(t-close).clip(0);left=(t-far).clip(0)
            n=(right-left).clip(1)[:,None,None]
            center=((right+left-1)*.5)[:,None,None]
            s0=p0[right]-p0[left];s1=p1[right]-p1[left];s2=p2[right]-p2[left]
            avg=s0/n
            tilt=np.sqrt(12.)*(s1-center*s0)/n**2
            curve=np.sqrt(180.)*(s2-2*center*s1+(center**2-(n**2-1)/12)*s0)/n**3
            hist[valid,:,j]=np.stack((avg,tilt,curve),2).astype('float16')
    return cols.clip(-10,10).astype('float16'),hist,volatility


class PatchModel(nn.Module):
    def __init__(self,mode):
        super().__init__()
        self.mode=mode
        self.now=nn.Sequential(nn.Linear(101,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.13),nn.Linear(64,48))
        self.patch=nn.Sequential(nn.Linear(48,36),nn.LayerNorm(36),nn.SiLU())
        self.time=nn.GRU(36,36,batch_first=True)
        self.keys=nn.Linear(84,4)
        self.receiver=nn.Linear(84,4)
        self.message=nn.Linear(84,24)
        self.idio=nn.Sequential(nn.Linear(132,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.12),nn.Linear(64,3))
        self.market=nn.Sequential(nn.Linear(120,48),nn.LayerNorm(48),nn.SiLU(),nn.Dropout(.18),nn.Linear(48,3))

    def forward(self,x,h,v):
        b,s=h.shape[:2]
        avg=h[:,:,:,0]
        shape=h[:,:,:,1:]
        if self.mode==0:
            shape=torch.zeros_like(shape)
        elif self.mode==1:
            shape=shape.roll(17,dims=1)
        patch=self.patch(torch.cat((avg,shape.flatten(3)),3).float())
        _,mem=self.time(patch.reshape(b*s,4,36))
        local=self.now(torch.cat((x.float(),v.log()[:,:,None]),2))
        state=torch.cat((local,mem[0].reshape(b,s,36)),2)
        scores=self.keys(state).softmax(1).transpose(1,2)
        value=self.message(state)
        groups=scores@value
        broad=value.mean(1,keepdim=True)
        incoming=self.receiver(state).softmax(2)@groups
        r=self.idio(torch.cat((state,incoming,broad.expand_as(value)),2))*v[:,:,None]
        r=r-r.mean(1,keepdim=True)
        market=self.market(torch.cat((groups.flatten(1),broad[:,0]),1))*v.mean(1,keepdim=True)
        return r,market


def example(d):
    raw=history.trial.clock.book.rawday(d)
    t=np.concatenate([np.arange(lo+(d*5+1)%11,hi-60,11) for lo,hi in base.segments(len(raw))])
    x,h,v=prepare(raw,t)
    y=history.trial.clock.labels(d).reshape(-1,50,3)[t]
    # Three increments reconstruct Y3 in its own normalized return units.
    short=.160943/.558942*y[:,:,0]
    middle=base.RATIO*y[:,:,1]
    target=np.stack((short,middle-short,y[:,:,2]-middle),2)
    return x,h,v,target.astype('float32')


def fit(data):
    x,h,v,y=[torch.from_numpy(np.concatenate(a)).cuda() for a in zip(*data)]
    models=[]
    for mode in range(3):
        torch.manual_seed(SEED)
        m=PatchModel(mode).cuda()
        avg=copy.deepcopy(m).eval()
        opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.04)
        for epoch in range(7):
            if epoch in (4,6):
                opt.param_groups[0]['lr']*=.55
            for tick in torch.randperm(len(y),device='cuda').split(40):
                rel,market=m(x[tick],h[tick],v[tick])
                total=y[tick]
                common=total.mean(1)
                local=total-common[:,None]
                def err(pred,truth):
                    return F.smooth_l1_loss(pred,truth,beta=.55)
                loss=err(rel,local)+.8*err(rel.sum(2),local.sum(2))+.7*err(rel[:,:,2],local[:,:,2])
                loss+=.5*err(market,common)+.5*err(market.sum(1),common.sum(1))
                opt.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(),1.0)
                opt.step()
                with torch.no_grad():
                    torch._foreach_lerp_(list(avg.parameters()),list(m.parameters()),.04)
        models.append(avg)
    return models


@torch.inference_mode()
def signals(raw,models):
    x,h,v=prepare(raw,np.arange(len(raw)))
    result=[]
    for idx in range(0,len(x),96):
        sl=slice(idx,idx+96)
        states=[torch.from_numpy(a[sl]).cuda() for a in (x,h,v)]
        pred=[m(*states) for m in models]
        local=[r.cpu().numpy() for r,_ in pred]
        market=[c.cpu().numpy()[:,None] for _,c in pred]
        both=[a+b for a,b in zip(local,market)]
        f=[p[:,:,2] for p in both]
        t=[p.sum(2) for p in both]
        a,b=f[2]-f[0],f[2]-f[1]
        def mkt(i):return np.broadcast_to(market[2][:,:,2]-market[i][:,:,2],f[2].shape)
        out=np.stack((*f,a,b,local[2][:,:,2]-local[0][:,:,2],local[2][:,:,2]-local[1][:,:,2],
                      mkt(0),mkt(1),t[2],t[2]-t[0],t[2]-t[1],
                      np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0.),.5*(a+b)),axis=2)
        result.append(out)
    return np.concatenate(result)


def train():
    OUT.mkdir(exist_ok=True)
    rows,folds,daily,data=[],[],[],[]
    for d in range(16):
        if d in (4,8,12):
            models=fit(data)
            if d>=8:
                w=base.calibrate(rows);fold=[]
        if d>=4:
            raw=history.trial.clock.book.rawday(d)
            y=history.trial.clock.labels(d).reshape(-1,50,3)
            pp=history.references(d,raw) if d>=8 else parent.warm.read(d,raw)
            z=signals(raw,models)
            if d==4:
                np.testing.assert_allclose(z[:131],signals(raw[:131],models),rtol=0,atol=8e-6)
            record=[base.records(p,y,z) for p in pp]
            rows.append(record)
            if d>=8:
                fold.append(record)
                daily.append([base.score(m,w) for m in record])
                if d in (11,15):
                    folds.append([base.score(m,w) for m in np.sum(fold,axis=0)])
        data.append(example(d))
        print('finished training-date pass',d+1,flush=True)
    f=np.asarray(folds);ds=np.asarray(daily)
    good=[j for j in range(1,len(ARMS)) if f[:,:,0,j].min()>0 and (ds[:,:,0,j]>0).sum(0).min()>=5]
    pick=max(good,key=lambda j:f[:,:,0,j].min()) if good else 0
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,selected=pick,folds=f.tolist(),daily=ds.tolist()),indent=2))
    print('selected',ARMS[pick],flush=True)
    if pick:
        models=fit(data)
        np.savez(OUT/'model.npz',pick=pick-1,coef=base.calibrate(rows)[pick-1],
                 **{str(i)+'_'+k:v.cpu().numpy() for i,m in enumerate(models) for k,v in m.state_dict().items()})
    return bool(pick)


def correction():
    models=[PatchModel(i).cuda().eval() for i in range(3)]
    with np.load(OUT/'model.npz') as f:
        for i,m in enumerate(models):
            m.load_state_dict({k:torch.from_numpy(f[str(i)+'_'+k]) for k in m.state_dict()})
        pick,coef=int(f['pick']),float(f['coef'])
    return lambda raw:(signals(raw,models)[:,:,pick]*coef).astype('float32')


def layer():
    before,delta=parent.layer(),correction()
    def predict(raw,initial):
        q=before(raw,initial).copy();q[:,:,2]+=delta(raw)
        return q
    return predict


def replay_with():
    delta=correction()
    def read(d,raw):
        p,h,n=history.read(d,raw);p[:,:,2]+=delta(raw);h[:,:,2]+=delta(raw[:n])
        return p,h,n
    return history.stack.verifier,read


if __name__=='__main__' and (sys.argv[1:]==['replay'] or train()):
    base.replay.replay(sys.modules[__name__])
