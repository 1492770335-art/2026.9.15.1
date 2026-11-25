"""Train a joint book encoder with a cross-date moment loss for fixed-terminal forecast revisions."""
import copy
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_full_coverage_book_v1 as cache
import experiment_y3_terminal_revision_chain_v1 as chain

clock=chain.clock
incumbent=chain.incumbent
OUT=clock.ART/'y3_martingale_factor_net_v1'
SEED=91381


class FactorNet(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(width,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.12),nn.Linear(96,48))
        self.stock=nn.Parameter(torch.zeros(50,48))
        self.assign=nn.Linear(48,4)
        self.value=nn.Linear(48,16)
        self.read=nn.Sequential(nn.Linear(96,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.10),nn.Linear(64,3))

    def forward(self,x,v):
        h=F.silu(self.enc(x.float())+self.stock)
        a=self.assign(h)
        keys=a.softmax(1).transpose(1,2)@self.value(h)
        neighbors=a.softmax(2)@keys
        common=self.value(h).mean(1,keepdim=True).expand_as(neighbors)
        spread=self.value(h).std(1,keepdim=True,unbiased=False).expand_as(common)
        z=self.read(torch.cat((h,neighbors,common,spread),2))*v[:,:,None]
        return z-z.mean(1,keepdim=True)


def dataset(end):
    xs=[];vol=[];labels=[];by_split=[[],[]];offset=0
    for d in range(end):
        cache.prepare(d);root=cache.CACHE/f'day_{d:02d}'
        fx=np.load(root/'x.npy',mmap_mode='r');fv=np.load(root/'v.npy',mmap_mode='r')
        t=np.arange(d%5,len(fx),5)
        xs.append(torch.tensor(fx[t],device='cuda'))
        vol.append(torch.tensor((.45+.55*fv[t]).clip(.55,3),device='cuda'))
        y=clock.labels(d).reshape(-1,50,3)[t].astype('float32')
        labels.append(torch.tensor(clock.center(y),device='cuda'))
        by_split[d%2].extend(offset+np.arange(len(t)-8));offset+=len(t)
    return torch.cat(xs),torch.cat(vol),torch.cat(labels),[torch.tensor(r,device='cuda') for r in by_split]


def instruments(x):
    selected=x[:,:,[0,5,11,14,21,26,38,41,42,54,55,77,78,-2]].float()
    z=torch.stack((selected[:,:,0],selected[:,:,1],selected[:,:,2],selected[:,:,3],selected[:,:,4],
                   selected[:,:,5]-selected[:,:,6],selected[:,:,7]-selected[:,:,8],
                   selected[:,:,9]-selected[:,:,10],selected[:,:,11]-selected[:,:,12],selected[:,:,13]),2)
    z=z-z.mean(1,keepdim=True)
    return (z/(.18+z.std(1,keepdim=True,unbiased=False))).tanh()


def train(end,arms=(False,True),factory=FactorNet,extra_loss=None):
    x,v,y,groups=dataset(end)
    nets=[]
    for restricted in arms:
        torch.manual_seed(SEED)
        m=factory(x.shape[2]).cuda();avg=copy.deepcopy(m).eval()
        opt=torch.optim.AdamW(m.parameters(),lr=.0007,weight_decay=.04)
        for ep in range(11):
            sets=[g[torch.randperm(len(g),device='cuda')] for g in groups]
            for u in range(0,min(map(len,sets))-31,32):
                i=torch.cat([g[u:u+32] for g in sets])
                prediction=m(x[i],v[i])
                e=(y[i]-prediction)/v[i,:,None]
                loss=F.smooth_l1_loss(e,torch.zeros_like(e),beta=1.)
                tail=e[:,:,2]-clock.RATIO*e[:,:,1]
                loss+=.45*F.smooth_l1_loss(tail,torch.zeros_like(tail),beta=.9)
                if extra_loss is not None:
                    loss+=extra_loss(m,prediction,y[i],v[i])
                if restricted and ep>=2:
                    with torch.no_grad():
                        # Five-tick origin stride: 8 and 3 advances retain the same terminal date.
                        err20=(y[i+8,:,1]-avg(x[i+8],v[i+8])[:,:,1])/v[i]
                        err5=(y[i+3,:,0]-avg(x[i+3],v[i+3])[:,:,0])/v[i]
                    rev=torch.stack((e[:,:,2]-clock.RATIO*err20,
                                     e[:,:,1]-(chain.UNIT[1]/chain.UNIT[0])*err5),2)
                    h=instruments(x[i])
                    moment=torch.einsum('bsi,bsk->bik',h,rev)/50
                    # Independent date groups keep the same observation's squared noise out of the moment loss.
                    a,b=moment.chunk(2)
                    loss+=6.*(a.mean(0)*b.mean(0)).mean()
                opt.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(),.9);opt.step()
                with torch.no_grad():torch._foreach_lerp_(list(avg.parameters()),list(m.parameters()),.025)
            if ep in (5,8):opt.param_groups[0]['lr']*=.55
        nets.append(avg)
    return nets


@torch.inference_mode()
def forecasts(raw,t,nets):
    scores=[[] for _ in nets]
    for ids in np.array_split(t,max(1,(len(t)+511)//512)):
        x,v=cache.phase.spline.features(raw,ids)
        x=torch.tensor(x,device='cuda');v=torch.tensor((.45+.55*v).clip(.55,3),device='cuda')
        for j,m in enumerate(nets):
            scores[j].append(torch.cat([m(a,b) for a,b in zip(x.split(64),v.split(64))]).cpu().numpy())
    return [np.concatenate(v) for v in scores]


def revision(p,signals):
    a,b=(out[:,:,2]-clock.RATIO*out[:,:,1] for out in signals)
    far=clock.center(p[:,:,2]-clock.RATIO*p[:,:,1])
    return .15*(b-a),.12*(b-far)


def run():
    folds=[];days=[]
    for end,stop in ((8,12),(12,14),(14,16)):
        nets=train(end)
        for begin in range(end,stop,2):
            group=[[[],[]],[[],[]]]
            for d in (begin,begin+1):
                raw=clock.book.rawday(d);t=np.arange(d%4,len(raw),4)
                signals=forecasts(raw,t,nets)
                if d==begin:
                    st=t[t<237]
                    for sub,full in zip(forecasts(raw[:237],st,nets),signals):
                        np.testing.assert_allclose(sub,full[:len(st)],rtol=0,atol=2e-5)
                y=clock.labels(d).reshape(-1,50,3)[t]
                for j,ref in enumerate(incumbent.REFS):
                    p=ref(d).reshape(-1,50,3)[t]
                    for k,z in enumerate(revision(p,signals)):group[j][k].append((p,y,z))
                days.append([chain.metric.gain(g[-1:]) for g in group[0]])
            folds.append([[chain.metric.gain(g) for g in side] for side in group])
            print(begin+1,folds[-1][0],flush=True)
    good=[k for k in (0,1) if all(sum(r[j][k]['y3']>0 for r in folds)>=3 and folds[-1][j][k]['y3']>0 and
         np.mean([r[j][k]['y3'] for r in folds])>0 and np.mean([r[j][k]['relative_far'] for r in folds])>0
         for j in (0,1)) and sum(d[k]['y3']>0 for d in days)>=5]
    pick=max(good,key=lambda k:np.mean([r[0][k]['y3'] for r in folds])) if good else None
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=days,chosen=pick,
         strategies=['moment loss minus supervised','replace 12% of relative far forecast']),indent=2))
    if pick is not None:
        torch.save(dict(models=[m.cpu().state_dict() for m in train(16)],chosen=pick),OUT/'models.pth')
    return pick is not None


def layer():
    parent=incumbent.layer();package=torch.load(OUT/'models.pth',map_location='cpu',weights_only=True)
    nets=[]
    for s in package['models']:
        m=FactorNet(s['enc.0.weight'].shape[1]).cuda().eval();m.load_state_dict(s);nets.append(m)
    def predict(raw,p):
        q=parent(raw,p);s=forecasts(raw,np.arange(len(q)),nets)
        q[:,:,2]+=revision(q,s)[package['chosen']]
        return q
    return predict


if __name__=='__main__':
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
    if run():chain.replay.verify(OUT,layer,incumbent.OUT)
