"""Train multi-horizon panel forecasts with oriented area between learned pairs of book trajectories."""
import copy
import json
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_y3_frequency_path_loss_v1 as path
import y3_morning_bias_references as history

parent,base=history.incumbent,history.base
OUT=parent.OUT.parent/'y3_oriented_book_paths_v1'
FIELDS=(0,10,11,16,21,25,34,37,39,40,45,52,54,55,60,64)
MODES=('levels and variation','displaced patch order','chronological signed area')
ARMS=('unchanged','levels terminal','displaced terminal','chronological terminal','relative terminal',
      'relative last forty','market last forty','terminal above levels','terminal above displaced','remote above displaced',
      'area ablation','remote area ablation','endpoint and remote agreement')


def features(raw,t):
    vals=np.arcsinh(raw[:,:,FIELDS].astype('float32'))
    x=np.empty((len(t),50,8,2*len(FIELDS)),np.float16)
    phase=np.empty((len(t),3),np.float32)
    for lo,hi in base.segments(len(raw)):
        mask=(t>=lo)&(t<hi)
        local=t[mask]-lo
        end=local[:,None]+1
        c=np.concatenate((np.zeros_like(vals[:1]),vals[lo:hi].cumsum(0,dtype=float)))
        for k,step in enumerate((3,16)):
            right=(end-np.arange(7,-1,-1)*step).clip(0)
            left=(right-step).clip(0)
            v=(c[right]-c[left])/(right-left).clip(1)[:,:,None,None]
            x[mask,:,:,k*len(FIELDS):(k+1)*len(FIELDS)]=v.transpose(0,2,1,3)
        phase[mask]=np.stack((np.exp(-local/240),local/2400,np.full_like(local,lo>0)),1)
    return x,phase


def sample(d):
    raw=history.trial.clock.book.rawday(d)
    y=history.trial.clock.labels(d).reshape(-1,50,3)
    t=np.concatenate([np.arange(lo+(3*d+1)%12,hi-60,12) for lo,hi in base.segments(len(y))])
    x,phase=features(raw,t)
    curve=path.targets(raw,y,t)[:,:,[0,3,7,11]]
    return x,phase,np.diff(curve,axis=2,prepend=np.zeros_like(curve[:,:,:1]))


class PathHead(nn.Module):
    def __init__(self,mode):
        super().__init__()
        self.mode=mode
        self.encode=nn.Sequential(nn.Linear(2*len(FIELDS),40),nn.LayerNorm(40),nn.SiLU(),nn.Linear(40,24))
        self.local=nn.Sequential(nn.Linear(84,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.12),nn.Linear(64,32))
        self.attention=nn.Linear(32,4)
        self.receive=nn.Linear(32,4)
        self.relative=nn.Sequential(nn.Linear(64,48),nn.SiLU(),nn.Dropout(.1),nn.Linear(48,4))
        self.common=nn.Sequential(nn.Linear(131,48),nn.LayerNorm(48),nn.SiLU(),nn.Dropout(.14),nn.Linear(48,4))

    def forward(self,x,phase,disable=False):
        trace=self.encode(x.float())
        spread=trace.std(2,unbiased=False)
        ordered=trace[:,:,[0,4,1,6,2,5,3,7]] if self.mode==1 else trace
        # Signed area of the closed polygon is invariant to translation and reverses sign under time reversal.
        pos=ordered-ordered[:,:,:1]
        xy=pos.reshape(*pos.shape[:-1],12,2)
        a,b=xy[:,:,:-1],xy[:,:,1:]
        area=(a[...,0]*b[...,1]-a[...,1]*b[...,0]).sum(2)*.5
        if self.mode==0 or disable:
            area=torch.zeros_like(area)
        h=self.local(torch.cat((trace[:,:,-1],trace.mean(2),spread,area),2))
        pools=self.attention(h).softmax(1).transpose(1,2)@h
        peer=self.receive(h).softmax(2)@pools
        rel=self.relative(torch.cat((h,peer),2))
        market=self.common(torch.cat((pools.flatten(1),phase),1))
        return rel-rel.mean(1,keepdim=True),market


def fit(rows,modes=range(len(MODES))):
    # Inputs remain in host memory; the device only holds one panel batch at a time.
    x,ph,y=[torch.from_numpy(np.concatenate(v)) for v in zip(*rows)]
    models=[]
    for mode in modes:
        torch.manual_seed(75019)
        net=PathHead(mode).cuda()
        ema=copy.deepcopy(net).eval()
        opt=torch.optim.AdamW(net.parameters(),lr=.00085,weight_decay=.06)
        for epoch in range(7):
            if epoch in (4,6):
                opt.param_groups[0]['lr']*=.5
            for ids in torch.randperm(len(y)).split(40):
                rel,common=net(x[ids].cuda(),ph[ids].cuda())
                truth=y[ids].cuda()
                market=truth.mean(1)
                errors=(rel-(truth-market[:,None]),common-market)
                loss=0.
                for e,weight in zip(errors,(1.,.65)):
                    scaled=e/e.new_tensor([.32,.54,.63,.63])
                    bands=torch.stack((e.sum(-1),e[...,2:].sum(-1),e[...,:2].sum(-1)),dim=-1)
                    loss+=weight*(F.smooth_l1_loss(scaled,torch.zeros_like(scaled),beta=.65)+
                                  .65*F.smooth_l1_loss(bands,torch.zeros_like(bands),beta=.65))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.)
                opt.step()
                with torch.no_grad():
                    torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.06)
        models.append(ema)
    return models


@torch.inference_mode()
def signals(raw,models):
    result=[]
    x,phase=features(raw,np.arange(len(raw)))
    for i in range(0,len(x),64):
        args=torch.from_numpy(x[i:i+64]).cuda(),torch.from_numpy(phase[i:i+64]).cuda()
        outputs=[m(*args) for m in models]+[models[2](*args,disable=True)]
        levels=[r+c[:,None] for r,c in outputs]
        y=[v.sum(2) for v in levels]
        far=[v[:,:,2:].sum(2) for v in levels]
        rel,market=outputs[2]
        d,e=y[2]-y[1],far[2]-far[1]
        agree=torch.where(d*e>0,d.sign()*torch.minimum(abs(d),abs(e)),0)
        v=(y[0],y[1],y[2],rel.sum(2),rel[:,:,2:].sum(2),market[:,2:].sum(1)[:,None].expand_as(d),
           y[2]-y[0],d,e,y[2]-y[3],far[2]-far[3],agree)
        result.append(torch.stack(v,2).cpu().numpy())
    return np.concatenate(result)


def train():
    OUT.mkdir(exist_ok=True)
    rows,grams,folds,daily,alone=[],[],[],[],[]
    for d in range(16):
        if d in (4,8,12):
            models=fit(rows)
            if d>=8:
                w=base.calibrate(grams)
                group=[]
            print('fitted through day',d,flush=True)
        if d>=4:
            raw,y,warm=parent.records(d)
            ps=history.references(d,raw) if d>=8 else warm
            z=signals(raw,models)
            if d==4:
                for n in (197,2413):
                    np.testing.assert_allclose(z[:n],signals(raw[:n],models),atol=5e-6,rtol=0)
            ms=[base.records(p,y,z) for p in ps]
            grams.append(ms)
            alone.append([float(np.corrcoef(z[:,:,j].ravel(),y[:,:,2].ravel())[0,1]) for j in range(3)])
            if d>=8:
                group.append(ms)
                daily.append([base.score(m,w)[0] for m in ms])
                if d in (11,15):
                    folds.append([base.score(m,w) for m in np.sum(group,0)])
        rows.append(sample(d))
    f,ds=np.array(folds),np.array(daily)
    ok=[j for j in range(3,len(ARMS)) if f[:,:,:,j].min()>0 and (ds[:,:,j]>0).sum(0).min()>=5 and
        (f[:,:,0,j].mean(0)>f[:,:,0,1:3].mean(0).max(1)).all()]
    pick=max(ok,key=lambda j:f[:,:,0,j].min()) if ok else 0
    np.savez(OUT/'training_moments.npz',grams=np.array(grams))
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,selected=pick,folds=f.tolist(),daily=ds.tolist(),
        standalone_y3=alone,windows=[[9,12],[13,16]]),indent=2))
    print('selected',ARMS[pick],flush=True)
    if pick:
        models=fit(rows)
        state={str(i)+'_'+k:v.cpu().numpy() for i,m in enumerate(models) for k,v in m.state_dict().items()}
        np.savez(OUT/'model.npz',pick=pick-1,w=base.calibrate(grams)[pick-1],**state)
    return bool(pick)


def correction():
    with np.load(OUT/'model.npz') as f:
        models=[PathHead(i).cuda().eval() for i in range(3)]
        for i,m in enumerate(models):
            m.load_state_dict({k:torch.from_numpy(f[str(i)+'_'+k]) for k in m.state_dict()})
        pick,w=int(f['pick']),float(f['w'])
    return lambda raw:signals(raw,models)[:,:,pick]*w


def layer():
    before,update=parent.layer(),correction()
    def apply(raw,initial):
        p=before(raw,initial).copy()
        p[:,:,2]+=update(raw)
        return p
    return apply


def replay_with():
    update=correction()
    def read(day,raw):
        p,h,n=history.read(day,raw)
        p[:,:,2]+=update(raw)
        h[:,:,2]+=update(raw[:n])
        return p,h,n
    return history.stack.verifier,read


if __name__=='__main__' and (sys.argv[1:]==['replay'] or train()):
    base.replay.replay(sys.modules[__name__])
