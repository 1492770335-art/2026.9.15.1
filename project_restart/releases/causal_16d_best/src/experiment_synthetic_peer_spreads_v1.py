"""Train on returns relative to a replicating portfolio of peers, then reconstruct stock forecasts."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_maturing_band_slopes_v1 as audit
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'synthetic_peer_spreads_v1'
roll=audit.roll
book=audit.book
RATIO=audit.RATIO
REG=.28


def portfolios():
    covs=[]
    for d in range(8):
        r=book.rawday(d)[:,:,0].astype(float)-.00136
        blocks=r[:len(r)//40*40].reshape(-1,40,50).sum(1)
        covs.append(np.cov(blocks.T))
    cov=np.mean(covs,0)
    cov+=.22*np.diag(cov.diagonal())
    weights=np.empty_like(cov)
    for i in range(50):
        other=np.arange(50)!=i
        g=cov[np.ix_(other,other)]
        b=cov[other,i]
        step=1/np.linalg.eigvalsh(g)[-1]
        w=np.ones(49)/49
        for _ in range(400):
            u=roll.simplex(w-step*(g@w-b))
            if np.max(abs(u-w))<1e-8: break
            w=u
        weights[i,other]=w;weights[i,i]=0
    return weights.astype('float32')


def operators(peer):
    diff=np.eye(50)-peer
    center=np.eye(50)-np.ones((50,50))/50
    inverse=center@np.linalg.solve(diff.T@diff+REG*np.eye(50),diff.T)
    return tuple(torch.tensor(v,dtype=torch.float32,device='cuda') for v in (peer,diff,inverse))


def tensors(raw,t):
    s,v=book.state(raw)
    x=s[t].clip(-5,5)
    return torch.from_numpy(x).cuda(),torch.from_numpy(v[t]).cuda()


class SpreadNet(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(n,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,32))
        self.head=nn.Sequential(nn.Linear(98,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.12),nn.Linear(64,4))

    def forward(self,x,v,P):
        h=self.encoder(x)
        neighbors=P@h
        z=torch.cat((h,h-neighbors,neighbors-h.mean(1,keepdim=True),
                     torch.log(v)[:,:,None],torch.log(v@P.T)[:,:,None]),dim=2)
        return self.head(z)


def dataset(end):
    data=[]
    for d in range(end):
        raw=book.rawday(d)
        t=np.arange(d%11,len(raw)-61,11)
        x,v=tensors(raw,t)
        y=torch.from_numpy(labels(d).reshape(-1,50,3)[t].copy()).cuda()
        y=torch.stack((y[:,:,0],y[:,:,1],y[:,:,2],y[:,:,2]-RATIO*y[:,:,1]),dim=2)
        data.append((x,v,y))
    return tuple(map(torch.cat,zip(*data)))


def train(ds,ops):
    x,v,y=ds
    torch.manual_seed(5423)
    model=SpreadNet(x.shape[-1]).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.04)
    weights=torch.tensor([.18,.30,.46,1.],device='cuda')
    for ep in range(8):
        for ix in torch.randperm(len(x),device='cuda').split(48):
            P,D,_=ops
            target=D@y[ix]
            noise=(.5+.32*(v[ix]+v[ix]@P.T)).clamp(.6,3)[:,:,None]
            out=model(x[ix],v[ix],P)
            loss=(weights*F.smooth_l1_loss(out/noise,target/noise,beta=.8,reduction='none')).mean()
            loss-=.09*(weights*F.cosine_similarity(out,target,dim=1,eps=1e-5)).mean()
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        if ep in (4,6):
            for group in optimizer.param_groups: group['lr']*=.5
    return model.eval()


@torch.inference_mode()
def infer(model,ops,raw,t):
    x,v=tensors(raw,t)
    P,_,inv=ops
    ys=[]
    for start in range(0,len(t),64):
        sl=slice(start,start+64)
        ys.append((inv@model(x[sl],v[sl],P)).cpu().numpy())
    return np.concatenate(ys)


def innovation(p,n):
    # Keep only directions new to all three stock-relative incumbent forecasts.
    x=p.astype(float)-p.mean(1,keepdims=True)
    gram=x.transpose(0,2,1)@x
    reg=np.eye(3)[None]*(.00001+.00001*gram.trace(axis1=1,axis2=2))[:,None,None]
    slope=np.linalg.solve(gram+reg,x.transpose(0,2,1)@n)
    return (n-x@slope).astype('float32')


def changed(p,n):
    z=innovation(p,n)
    q=p.copy()
    q[:,:,2]+=.18*(.30*z[:,:,2]+.70*z[:,:,3])
    return q


def load():
    parent=audit.incumbent.load()
    ops=operators(np.load(OUT/'portfolio.npy'))
    st=torch.load(OUT/'net.pth',map_location='cpu',weights_only=True)
    model=SpreadNet(st['encoder.0.weight'].shape[1]).cuda().eval()
    model.load_state_dict(st)
    def apply(raw,p):
        base=parent(raw,p)
        n=infer(model,ops,raw,np.arange(len(raw)))
        return changed(base,n)
    return apply


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    uni=(np.ones((50,50))-np.eye(50))/49
    peer=portfolios()
    np.testing.assert_allclose(peer.sum(1),1,atol=1e-6)
    ops=[operators(w) for w in (uni,peer)]
    folds=[];mixing=[];daily=[];variance=[];tail_skill=[]
    for end,stop in ((8,12),(12,14),(14,16)):
        ds=dataset(end)
        models=[train(ds,o) for o in ops]
        del ds
        for begin in range(end,stop,2):
            block=[[],[]];coef=roll.fit(mixing)
            for d in (begin,begin+1):
                raw=book.rawday(d);fs=roll.sources(d);all_t=np.arange(len(raw))
                t=all_t[d%4::4]
                yfull=labels(d).reshape(-1,50,3)
                p=roll.predict(fs,t,coef);y=yfull[t]
                mixing.append(roll.moments(fs,yfull,all_t[d%5::5]))
                true=y[:,:,2]-RATIO*y[:,:,1]
                variance.append([np.mean((true-true@w.T)**2).item() for w in (uni,peer)])
                day_skill=[]
                for j,(m,o) in enumerate(zip(models,ops)):
                    n=infer(m,o,raw,t)
                    if d==begin:
                        short=infer(m,o,raw[:185],t[t<185])
                        np.testing.assert_allclose(n[:len(short)],short,rtol=0,atol=2e-6)
                    q=changed(p,n)
                    row=tuple(a.reshape(-1,3) for a in (p,q,y))
                    block[j].append(row)
                    yrel=true-true.mean(1,keepdims=True)
                    day_skill.append(float(np.corrcoef(n[:,:,3].ravel(),yrel.ravel())[0,1]))
                    if j:
                        s=measure([row]);daily.append(s['candidate'][2]-s['base'][2])
                tail_skill.append(day_skill)
            folds.append([measure(rs) for rs in block])
            print(begin+1,[(sc['candidate'][2]-sc['base'][2]) for sc in folds[-1]],flush=True)
    g=np.array([[s['candidate'][2]-s['base'][2] for s in f] for f in folds])
    keep=bool((g[:,1]>0).all() and (g[:,1]-g[:,0]).mean()>0 and (np.array(daily)>0).sum()>=5)
    report=dict(arms=['equal peer weights','synthetic peer basket'],folds=folds,train_cutoffs=[8,8,12,14],
                far_spread_variance=variance,relative_far_ic=tail_skill,daily=daily,retained=keep)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    print('retained',keep,flush=True)
    if keep:
        np.save(OUT/'portfolio.npy',peer)
        model=train(dataset(16),ops[1])
        torch.save(model.cpu().state_dict(),OUT/'net.pth')
        replay=roll.gls.best.walk.replay
        result=replay(load())
        (OUT/'raw_replay.json').write_text(json.dumps(result,indent=2))
        old=json.loads((audit.incumbent.OUT/'reload_verification.json').read_text())['pooled']['score']
        print('official',result['pooled']['score'],result['pooled']['score']-old,flush=True)
        if result['pooled']['score']>old:
            check=replay(load())
            np.testing.assert_allclose(check['pooled']['candidate'],result['pooled']['candidate'],rtol=0,atol=1e-10)
            (OUT/'reload_verification.json').write_text(json.dumps(check,indent=2))


if __name__=='__main__':
    main()
