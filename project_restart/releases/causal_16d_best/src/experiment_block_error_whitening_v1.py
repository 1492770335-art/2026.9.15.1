"""Fit four nearby origins jointly with a training-only covariance for their errors."""
import copy
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_verified_near_carry_v1 as best
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

segment = best.incumbent.segment
paths = segment.paths
OUT = ART/'block_error_whitening_v1'
UNITS = np.array([.558942,.272164,.160943],dtype='float32')
GAPS = np.array([0,4,8,12])


def data(end,enc):
    xs,ys=[],[]
    for d in range(end):
        raw=paths.book.rawday(d)
        origins=np.arange(d%7,len(raw)-61-GAPS[-1],28)
        t=(origins[:,None]+GAPS).ravel()
        x,v=paths.frozen.encoding(raw,t,enc,d,fixed_batch=True)
        y=labels(d).reshape(-1,50,3)[t].astype('float32')
        y[:,:,1:]-=y[:,:,:2].copy()*(UNITS[1:]/UNITS[:-1])
        y=torch.from_numpy(y).cuda()
        m=y.mean(1,keepdim=True)
        truth=torch.cat(((y-m)/v[:,:,None],m/v.mean(1)[:,None,None]),dim=1)
        xs.append(x.reshape(len(origins),4,50,-1))
        ys.append(truth.reshape(len(origins),4,51,3))
    return torch.cat(xs),torch.cat(ys)


def geometry(truth):
    mats=[]
    for y in (truth[:,:,:50],truth[:,:,50:]):
        source=y.permute(3,0,2,1).reshape(3,-1,4).double()
        source-=source.mean(1,keepdim=True)
        cov=source.transpose(1,2)@source/source.shape[1]
        sd=cov.diagonal(dim1=1,dim2=2).clamp_min(1e-6).sqrt()
        corr=cov/(sd[:,:,None]*sd[:,None])
        eig,vec=torch.linalg.eigh(corr)
        # Shrink toward independent errors, then normalize expected squared error.
        precision=(.5+.5*eig).rsqrt()
        energy=(precision.square()*eig).mean(1).sqrt()
        w=(vec*(precision/energy[:,None])[:,None,:])@vec.transpose(1,2)
        mats.append(w.float())
    return torch.stack(mats)


class Head(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(width,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,40),nn.SiLU())
        def head():
            return nn.Sequential(nn.Linear(80,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,3))
        self.local=head()
        self.market=head()

    def forward(self,x):
        h=self.enc(x.float())
        m=h.mean(1,keepdim=True)
        local=self.local(torch.cat((h,m.expand_as(h)),dim=2))
        common=self.market(torch.cat((m[:,0],h.std(1,unbiased=False)),dim=1))
        return torch.cat((local,common[:,None]),dim=1)


def train(end,enc, objective="whiten", seed=7806):
    x,y=data(end,enc)
    w=geometry(y)
    group_scales=[]
    for target in (y[:,:,:50],y[:,:,50:]):
        group_var=target.mean(1).var((0,1))
        single_var=target.var((0,1,2))
        group_scales.append((group_var/single_var.clamp_min(1e-8)).sqrt().clamp(.3,1.))
    nets=[]
    for whiten in (False,True):
        torch.manual_seed(seed)
        net=Head(x.shape[-1]).cuda()
        avg=copy.deepcopy(net).eval()
        opt=torch.optim.AdamW(net.parameters(),lr=.00065,weight_decay=.045)
        for epoch in range(9):
            for ids in torch.randperm(len(x),device='cuda').split(16):
                err=net(x[ids].flatten(0,1)).reshape(-1,4,51,3)-y[ids]
                loss=0.
                for j,(part,factor) in enumerate(((err[:,:,:50],1.),(err[:,:,50:],.6))):
                    if whiten and objective == 'whiten':
                        part=torch.einsum('bisk,kij->bjsk',part,w[j])
                    term=F.smooth_l1_loss(part,torch.zeros_like(part),beta=.85)
                    if whiten and objective == 'block_mean':
                        mean=part.mean(1)/group_scales[j]
                        term=.65*term+.35*F.smooth_l1_loss(mean,torch.zeros_like(mean),beta=.85)
                    loss+=factor*term
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),.8)
                opt.step()
                with torch.no_grad():
                    torch._foreach_lerp_(list(avg.parameters()),list(net.parameters()),.024)
            if epoch in (4,7):
                for group in opt.param_groups:group['lr']*=.55
        nets.append(avg)
    return nets


@torch.inference_mode()
def changes(raw,t,enc,nets):
    x,v=paths.frozen.encoding(raw,t,enc,fixed_batch=True)
    out=[]
    for start in range(0,len(t),64):
        sl=slice(start,start+64)
        delta=nets[1](x[sl])-nets[0](x[sl])
        scale=torch.cat((v[sl],v[sl].mean(1,keepdim=True)),dim=1)
        out.append((delta*scale[:,:,None]).cpu().numpy())
    bands=np.concatenate(out)
    cumulative=np.cumsum(bands/UNITS,axis=2)*UNITS
    loc=cumulative[:,:50]
    return loc-loc.mean(1,keepdims=True),cumulative[:,50:]


def load():
    parent=best.load()
    enc=paths.decoder.net(16)
    keep=np.load(OUT/'mask.npy')
    nets=[]
    for j in (0,1):
        s=torch.load(OUT/f'net_{j}.pth',map_location='cpu',weights_only=True)
        net=Head(s['enc.0.weight'].shape[1]).cuda().eval()
        net.load_state_dict(s)
        nets.append(net)
    def predict(raw,p):
        q=parent(raw,p).copy()
        for delta,k in zip(changes(raw,np.arange(len(raw)),enc,nets),keep):
            q+=.15*delta*k
        return q
    return predict


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    folds,daily=[],[]
    for end,stop in ((8,12),(12,14),(14,16)):
        enc=paths.decoder.net(end)
        nets=train(end,enc)
        for begin in range(end,stop,2):
            rows=[[],[]]
            for d in (begin,begin+1):
                raw=paths.book.rawday(d)
                t=np.arange(d%4,len(raw),4)
                y=labels(d).reshape(-1,50,3)[t]
                p=np.load(segment.best.refit.OUT/f'reference_{d}.npy',mmap_mode='r')[t]
                sides=changes(raw,t,enc,nets)
                if d==begin:
                    early=t[t<177]
                    for a,b in zip(changes(raw[:177],early,enc,nets),sides):
                        np.testing.assert_allclose(a,b[:len(early)],rtol=0,atol=3e-5)
                for side,g in zip(sides,rows):
                    g.append(tuple(a.reshape(-1,3) for a in (p,p+.15*side,y)))
                def gain(g):
                    s=measure(g)
                    return np.subtract(s['candidate'],s['base']).tolist()
                daily.append([gain(g[-1:]) for g in rows])
            folds.append([gain(g) for g in rows])
            print(begin+1,'IC gains local/market',folds[-1],flush=True)
    keep=(np.min(folds,axis=0)>0)&((np.array(daily)>0).sum(0)>=5)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=daily,retained=keep.tolist()),indent=2))
    if keep.any():
        np.save(OUT/'mask.npy',keep)
        enc=paths.decoder.net(16)
        for j,net in enumerate(train(16,enc)):
            torch.save(net.cpu().state_dict(),OUT/f'net_{j}.pth')
        baseline=json.loads((best.OUT/'reload_verification.json').read_text())['pooled']['score']
        for name in ('raw_replay','reload_verification'):
            res=segment.best.roll.gls.best.walk.replay(load())
            if name=='reload_verification':
                old=json.loads((OUT/'raw_replay.json').read_text())
                np.testing.assert_allclose(res['pooled']['candidate'],old['pooled']['candidate'],rtol=0,atol=1e-10)
            (OUT/f'{name}.json').write_text(json.dumps(res,indent=2))
            print(name,res['pooled']['score'],flush=True)
            if res['pooled']['score']<=baseline:break


if __name__=='__main__':
    main()
