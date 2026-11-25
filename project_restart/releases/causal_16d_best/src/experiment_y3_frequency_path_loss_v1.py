"""Compare time-domain and complex Fourier losses for a twelve-endpoint causal return-path decoder."""
import copy
import json
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_y3_cross_date_factor_mix_v1 as featurizer
import y3_flow_exchange_references as history

base,parent=history.base,history.incumbent
OUT=parent.OUT.parent/'y3_frequency_path_loss_v1'
MODES=('pointwise','permuted Fourier','ordered Fourier','separate near and remote Fourier')
ARMS=('unchanged','pointwise Y3','permuted-frequency Y3','ordered-frequency Y3','split-frequency Y3',
      'ordered minus pointwise','ordered minus permuted','split minus pointwise','split minus permuted',
      'ordered remote return','split remote return','ordered remote improvement','relative ordered improvement',
      'relative split improvement','common ordered improvement','common split improvement','low-frequency path average')
SCALE=np.array([.558942,.272164,.160943])


def targets(raw,y,t):
    price=np.vstack((np.zeros(50),base.raw_returns(raw).cumsum(0)))
    pts=t[:,None]+np.arange(0,61,5)+1
    increments=np.diff(price[pts],axis=1).transpose(0,2,1)*SCALE[2]
    previous=0.
    for end,idx,begin in ((1,0,0),(4,1,1),(12,2,4)):
        actual=(SCALE[2]/SCALE[idx])*y[t,:,idx]
        increments[:,:,begin:end]+=((actual-previous-increments[:,:,begin:end].sum(2))/(end-begin))[:,:,None]
        previous=actual
    path=increments.cumsum(2)
    np.testing.assert_allclose(path[:,:,-1],y[t,:,2],rtol=1e-6,atol=2e-6)
    return path.astype('float32')


def data(day):
    raw=history.trial.clock.book.rawday(day);y=history.trial.clock.labels(day).reshape(-1,50,3)
    t=np.concatenate([np.arange(lo+(3+day*7)%10,hi-60,10) for lo,hi in base.segments(len(raw))])
    x,phase=featurizer.features(raw,t)
    return x,phase,targets(raw,y,t)


class Decoder(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.local=nn.Sequential(nn.Linear(n,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.12),nn.Linear(96,48))
        self.port=nn.Linear(48,4)
        self.send=nn.Linear(48,24)
        self.receive=nn.Linear(48,4)
        self.head=nn.Sequential(nn.Linear(96,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.13),nn.Linear(64,12))
        self.market=nn.Sequential(nn.Linear(123,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.15),nn.Linear(64,12))

    def forward(self,x,phase):
        h=self.local(x.float())
        v=self.send(h);mid=v.mean(1,keepdim=True)
        factors=self.port(h).softmax(1).transpose(1,2)@v
        read=self.receive(h).softmax(2)@factors
        rel=self.head(torch.cat((h,read,mid.expand_as(read)),2))
        common=self.market(torch.cat((factors.flatten(1),mid[:,0],phase),1))
        return rel-rel.mean(1,keepdim=True),common


def path_loss(error,mode):
    scale=torch.sqrt(torch.arange(1,13,device=error.device)/12)
    normalized=error/scale
    point=F.smooth_l1_loss(normalized,torch.zeros_like(normalized),beta=.7)
    if mode==0: return point
    def spectrum(e):
        return torch.fft.fft(e,dim=-1,norm='ortho').abs().mean()
    if mode==1: spec=spectrum(normalized[..., [0,9,3,7,2,11,5,8,1,10,4,6]])
    elif mode==2: spec=spectrum(normalized)
    else: spec=.35*spectrum(normalized[:,:,:4])+.65*spectrum((error[:,:,4:]-error[:,:,3:4])/scale[4:])
    return .6*point+.4*spec


def fit(records):
    x,ph,y=[torch.from_numpy(np.concatenate(v)).cuda() for v in zip(*records)]
    models=[]
    for mode in range(len(MODES)):
        torch.manual_seed(87341)
        m=Decoder(x.shape[-1]).cuda()
        opt=torch.optim.AdamW(m.parameters(),lr=.0009,weight_decay=.045)
        ema=copy.deepcopy(m).eval()
        for ep in range(7):
            if ep in (4,6):opt.param_groups[0]['lr']*=.55
            for ids in torch.randperm(len(x),device='cuda').split(42):
                rel,market=m(x[ids],ph[ids]);truth=y[ids]
                common=truth.mean(1);loc=truth-common[:,None]
                e=rel-loc;em=market-common
                loss=path_loss(e,mode)+.55*path_loss(em[:,None],mode)
                total=e+em[:,None]
                loss+=.5*F.smooth_l1_loss(total[:,:,-1],torch.zeros_like(total[:,:,-1]),beta=.75)
                remote=total[:,:,-1]-total[:,:,3]
                loss+=.35*F.smooth_l1_loss(remote,torch.zeros_like(remote),beta=.6)
                opt.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(),.95);opt.step()
                with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(m.parameters()),.05)
        models.append(ema)
    return models


@torch.inference_mode()
def signals(raw,models):
    x,ph=featurizer.features(raw,np.arange(len(raw)))
    result=[]
    for i in range(0,len(raw),96):
        xx=torch.from_numpy(x[i:i+96]).cuda();phase=torch.from_numpy(ph[i:i+96]).cuda()
        pair=[net(xx,phase) for net in models]
        rel=[v.cpu().numpy() for v,_ in pair];market=[v.cpu().numpy()[:,None] for _,v in pair]
        levels=[v+m for v,m in zip(rel,market)]
        a,b,c,d=[q[:,:,-1] for q in levels]
        far=[q[:,:,-1]-q[:,:,3] for q in levels]
        shape=a.shape
        z=np.stack((a,b,c,d,c-a,c-b,d-a,d-b,far[2],far[3],far[2]-far[0],rel[2][:,:,-1]-rel[0][:,:,-1],
                    rel[3][:,:,-1]-rel[0][:,:,-1],np.broadcast_to(market[2][:,:,-1]-market[0][:,:,-1],shape),
                    np.broadcast_to(market[3][:,:,-1]-market[0][:,:,-1],shape),
                    (levels[2][:,:,4:]-levels[0][:,:,4:]).mean(2)),2)
        result.append(z)
    return np.concatenate(result)


def train():
    OUT.mkdir(exist_ok=True)
    records,grams,folds,daily,standalone=[],[],[],[],[]
    warm=parent.parent.warm
    for d in range(16):
        if d in (4,8,12):
            models=fit(records)
            if d>=8:w=base.calibrate(grams);group=[]
        if d>=4:
            raw=history.trial.clock.book.rawday(d);y=history.trial.clock.labels(d).reshape(-1,50,3)
            z=signals(raw,models)
            if d==4:np.testing.assert_allclose(z[:117],signals(raw[:117],models),rtol=0,atol=6e-6)
            ps=history.references(d,raw) if d>=8 else warm.read(d,raw)
            rec=[base.records(p,y,z) for p in ps];grams.append(rec)
            standalone.append([float(np.corrcoef(z[:,:,k].ravel(),y[:,:,2].ravel())[0,1]) for k in range(4)])
            if d>=8:
                group.append(rec);daily.append([base.score(m,w) for m in rec])
                if d in (11,15):folds.append([base.score(m,w) for m in np.sum(group,axis=0)])
        records.append(data(d))
        if d%4==3:print('trained through day',d+1,flush=True)
    f=np.asarray(folds);days=np.asarray(daily)
    good=[j for j in range(1,len(ARMS)) if f[:,:,0,j].min()>0 and (days[:,:,0,j]>0).sum(0).min()>=5]
    pick=max(good,key=lambda j:f[:,:,0,j].min()) if good else 0
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,modes=MODES,selected=pick,folds=f.tolist(),daily=days.tolist(),
        daily_standalone_y3=standalone,reference='https://arxiv.org/abs/2402.02399'),indent=2))
    print('selected',ARMS[pick],flush=True)
    if pick:
        mods=fit(records)
        np.savez(OUT/'model.npz',n=records[0][0].shape[-1],coef=base.calibrate(grams)[pick-1],pick=pick-1,
                 **{str(k)+'_'+name:v.cpu().numpy() for k,m in enumerate(mods) for name,v in m.state_dict().items()})
    return bool(pick)


def correction():
    with np.load(OUT/'model.npz') as f:
        mods=[Decoder(int(f['n'])).cuda().eval() for _ in MODES]
        for k,m in enumerate(mods):m.load_state_dict({key:torch.from_numpy(f[str(k)+'_'+key]) for key in m.state_dict()})
        pick,coef=int(f['pick']),float(f['coef'])
    return lambda raw:(signals(raw,mods)[:,:,pick]*coef).astype('float32')


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
