"""Forecast spreads between matched stocks from the signed lag of their past book paths."""
import argparse
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_y3_echo_vintage_ensemble_v1 as ensemble
from y3_portfolio_walkforward import replay,scores

parent=ensemble.parent
OUT=parent.ART/'y3_temporal_pair_alignment_v1'
ARMS=('incumbent','own history','paired time alignment','alignment above own histories',
      'cross-stock lag alone','two horizons agree','shuffled patch order')


def prefix(raw):
    rng=np.random.default_rng(13313)
    proj=rng.normal(size=(100,8)).astype('float32')*.10
    out=[]
    for a,b in parent.base.segments(len(raw)):
        chunks=[]
        for start in range(a,b,128):
            r=raw[start:min(start+128,b)]
            z=np.tanh(.7*np.arcsinh(r.astype('float32')))
            q=z[:,:,[26,38,41,42,54,55]]
            s=np.concatenate((z@proj,q[:,:,::2]-q[:,:,1::2],z[:,:,:1]),2)
            s-=s.mean(1,keepdims=True)
            chunks.append(s)
        s=np.concatenate(chunks)
        out.append((a,b,np.concatenate((np.zeros_like(s[:1],dtype=float),s.cumsum(0,dtype=float)))))
    return out

def patches(pref,t):
    out=np.empty((len(t),50,2,8,12),dtype=np.float32)
    for a,b,cs in pref:
        ix=np.flatnonzero((t>=a)&(t<b))
        for idx in np.array_split(ix,max(1,(len(ix)+63)//64)):
            end=(t[idx]-a+1)[:,None]
            for j,width in enumerate((8,32)):
                ends=(end-np.arange(7,-1,-1)*width).clip(0)
                starts=(ends-width).clip(0)
                v=(cs[ends]-cs[starts])/(ends-starts).clip(1)[:,:,None,None]
                out[idx,:,j]=v.transpose(0,2,1,3)
    return out

def target(p,y):
    p,y=parent.exposure.bands(p),parent.exposure.bands(y)
    p=p-p.mean(1,keepdims=True);y=y-y.mean(1,keepdims=True)
    beta=((p*y).sum((0,1))/np.square(p).sum((0,1)).clip(.001)).clip(.45,2.5)
    return (y/beta-p).astype('float32')

def panels():
    graphs=[ensemble.peer.graph_stat(ensemble.peer.rawday(d)) for d in range(4)]
    hist,bank=[],[]
    for d,raw,ps,y in ensemble.reference_days():
        if d%2==0:
            graph=ensemble.peer.graph(graphs)[2]
            if d>=8:bank.append((graph,ensemble.fit(hist)))
        ref=ps[-1] if d>=8 else np.mean(ps[:2],0)
        hist.append(ensemble.record(ensemble.design(raw,graph),raw,ref,y,d))
        if d>=8:
            ps=ps.copy()
            ps[:,:,:,2]+=ensemble.changes(raw,bank)[None,:,:,1]
        yield d,raw,ps,y,graph
        graphs.append(ensemble.peer.graph_stat(raw))

def neighborhood(p):
    scale=p.std(1,keepdim=True,unbiased=False).clamp_min(.06)
    v=p/scale
    distance=torch.cdist(v,v)
    distance.diagonal(dim1=1,dim2=2).fill_(float('inf'))
    k=distance.kthvalue(7,dim=2).values.clamp_min(.15)
    kernel=torch.exp(-(distance/k[:,:,None]).square())*(distance<=k[:,:,None])
    return (kernel+kernel.transpose(1,2))*.5


class Spread(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Linear(24,12)
        self.probe=nn.Linear(24,8,bias=False)
        self.price=nn.Sequential(nn.Linear(32,20),nn.SiLU(),nn.Linear(20,2))
        self.orientation=nn.Linear(20,6,bias=False)
        self.read=nn.Linear(6,2,bias=False)

    def forward(self,x,p):
        seq=x.transpose(2,3).flatten(3)
        h=self.embed(seq).tanh()
        q=self.probe(seq)
        now=q[:,:,-1]
        early=q[:,:,:-1].mean(2)
        own=self.price(torch.cat((h[:,:,-1],h.mean(2),now-early),2))
        orientation=self.orientation(torch.cat((h.mean(2),now-early),2))
        direction=(orientation[:,:,None]+orientation[:,None,:]).tanh()
        corr=[]
        for k in (1,2,4):
            for sig in (h,q):
                past=F.normalize(sig[:,:,:-k].flatten(2),dim=2,eps=.02)
                recent=F.normalize(sig[:,:,k:].flatten(2),dim=2,eps=.02)
                match=recent@past.transpose(1,2)
                corr.append(match-match.transpose(1,2))
        skew=self.read(torch.stack(corr,3)*direction)
        diff=own[:,:,None]-own[:,None,:]
        return diff,skew


def sample(pref,p,y,d):
    t=np.concatenate([np.arange(a+d%9,b-60,9) for a,b in parent.base.segments(len(p))])
    return patches(pref,t).astype('float16'),p[t],target(p[t],y[t])


def fit(hist,dev):
    x,p,y=[torch.from_numpy(np.concatenate(v)).to(dev) for v in zip(*hist)]
    rng=np.random.default_rng(35027)
    torch.manual_seed(35027)
    net=Spread().to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=.0018,weight_decay=.08)
    for _ in range(340):
        ids=torch.as_tensor(rng.integers(len(p),size=20),device=dev)
        w=neighborhood(p[ids])[:,:,:,None]
        truth=y[ids,:,None]-y[ids,None,:]
        local,lag=net(x[ids].float(),p[ids])
        both=local+lag
        magnitude=F.smooth_l1_loss(both,truth,beta=.8,reduction='none')
        ranking=F.softplus(-truth.sign()*both)*truth.abs().clamp(max=1.5)
        loss=((magnitude+.28*ranking+.15*lag.square())*w).sum()/(2*w.sum())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step()
    return net.eval()


@torch.inference_mode()
def shifts(pref,p,net,y=None):
    device=next(net.parameters()).device
    out=[]
    tally=np.zeros(4)
    for i in range(0,len(p),48):
        t=np.arange(i,min(i+48,len(p)))
        x=torch.from_numpy(patches(pref,t)).to(device)
        pr=torch.from_numpy(p[t]).to(device)
        w=neighborhood(pr)
        def score(edge):
            degree=w.sum(2).clamp_min(.25)[:,:,None]
            v=(edge*w[:,:,:,None]).sum(2)/degree
            return v-v.mean(1,keepdim=True)
        local,lag=net(x,pr)
        full,late=score(local+lag).unbind(2)
        item=score(local)[:,:,0]
        paired=score(lag)[:,:,0]
        bad=score(net(x[:,:,:,[3,6,1,4,7,0,5,2]],pr)[1])[:,:,0]
        agreement=torch.where(full*late>0,full.sign()*torch.minimum(full.abs(),late.abs()),0.)
        delta=.15*torch.stack((item,full,full-item,paired,agreement,bad),2)
        out.append((delta-delta.mean(1,keepdim=True)).cpu().numpy())
        if y is not None:
            target=torch.from_numpy(y[t,:,2]).to(device)
            edge=target[:,:,None]-target[:,None,:]
            base=pr[:,:,2,None]-pr[:,None,:,2]
            changed=base+delta[:,:,1,None]-delta[:,None,:,1]
            mask=(w>0)&(edge!=0)
            tally+=np.array([mask.sum().item(),((edge*base>0)*mask).sum().item(),
                    ((edge*changed>0)*mask).sum().item(),((edge*(local+lag)[:,:,:,0]>0)*mask).sum().item()])
    return np.concatenate(out),tally


def train():
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hist,days,tallies=[],[],[]
    for d,raw,ps,y,g in panels():
        pref=prefix(raw)
        if d>=8:
            if d%2==0:net=fit(hist,dev)
            p=ps[-1]
            delta,acc=shifts(pref,p,net,y)
            if d==8:
                cut=2451
                np.testing.assert_allclose(delta[:cut],shifts(prefix(raw[:cut]),p[:cut],net)[0],rtol=0,atol=3e-6)
            days.append([parent.base.records(p,y,delta)])
            tallies.append(acc)
        p=ps[-1] if d>=8 else np.mean(ps[:2],0)
        hist.append(sample(pref,p,y,d))
        print('day',d+1,flush=True)
    h=np.stack([scores(v,len(ARMS))[0] for v in (days[:4],days[4:])])
    ds=np.stack([scores([v],len(ARMS))[0] for v in days])
    acc=np.stack([sum(v) for v in (tallies[:4],tallies[4:])]);acc=acc[:,1:]/acc[:,:1]
    j=max((2,3,4,5),key=lambda i:h[0,0,i])
    keep=h[:,0,j].min()>0 and (ds[:,0,j]>0).sum()>=5 and (h[:,0,j]-h[:,0,6]).mean()>0
    OUT.mkdir(exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,choice=j,selected=j if keep else 0,halves=h.tolist(),
                   daily=ds.tolist(),paired_accuracy=acc.tolist(),accuracy_models=['incumbent','incumbent plus paired','paired correction alone']),indent=2))
    print('choice',ARMS[j],'selected',j if keep else 0,'gain',h[:,:,j],'pair accuracy',acc,flush=True)
    if keep:torch.save(dict(state=fit(hist,dev).cpu().state_dict(),pick=j),OUT/'model.pt')


def correction():
    prior=ensemble.correction()
    f=torch.load(OUT/'model.pt',weights_only=True,map_location='cpu')
    net=Spread();net.load_state_dict(f['state'])
    def update(raw,p):
        delta=prior(raw,p)
        q=p.copy();q[:,:,2]+=delta
        return delta+shifts(prefix(raw),q,net.eval())[0][:,:,f['pick']-1]
    return update


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('train','screen','verify'))
    parser.add_argument('--upstream-dir')
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.mode=='train':train()
    else:replay(OUT,correction,args.upstream_dir)
