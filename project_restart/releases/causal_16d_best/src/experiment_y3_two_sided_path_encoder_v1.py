"""Learn forward-looking stock signals from paired returns on opposite sides of the same observed path."""
from functools import lru_cache
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import y3_crossing_validation as history
import y3_early_score_neighborhoods as early

trial=history.trial
OUT=trial.clock.ART/'y3_two_sided_path_encoder_v1'
ARMS=['direct forward forecast','shared orientation','even component','odd component','signed plus even and odd',
      'unrelated backward target','mirrored input without price reversal','orientation beyond direct','half orientation effect']


def stream(raw):
    idx=[0,1,2,4,7,11,12,13,14,15,17,18,21,25,26,38,41,42,50,51,54,55,66,69,74,77,78]
    x=np.tanh(.65*np.arcsinh(raw[:,:,idx].astype('float32')))
    x=np.concatenate((x,raw[:,:,83:].astype('float32')-1),2)
    return np.concatenate((np.zeros_like(x[:1],dtype=float),x.cumsum(0,dtype=float)))


def patches(s,t):
    end=t[:,None]+1-8*np.arange(7,-1,-1)[None]
    start=(end-8).clip(0)
    end=end.clip(0)
    x=(s[end]-s[start])/(end-start).clip(1)[:,:,None,None]
    return x.transpose(0,2,1,3).astype('float16')


class Path(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.patch=nn.Sequential(nn.Linear(n,28),nn.LayerNorm(28),nn.SiLU())
        self.path=nn.Sequential(nn.Linear(8*28,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,24),nn.SiLU())
        self.read=nn.Sequential(nn.Linear(48,32),nn.SiLU(),nn.Linear(32,3))

    def forward(self,x):
        h=self.path(self.patch(x.float()).flatten(2))
        p=self.read(torch.cat((h,h.mean(1,keepdim=True).expand_as(h)),2))
        return p-p.mean(1,keepdim=True)


def reverse(x, signed):
    xr=x.flip(2)
    if signed:
        xr[:,:,:,0]*=-1
    return xr


def fit_paths(days, variants):
    records=[]
    for d in days:
        raw=trial.clock.book.rawday(d)
        y=trial.clock.labels(d).reshape(-1,50,3)
        t=np.arange(130+d%7,len(y)-60,7)
        # The window contains ticks t-63 through t. Both targets lie strictly outside that window.
        forward=np.stack([y[t+i,:,1] for i in (0,20,40)],2)
        backward=-np.stack([y[t-64-20-i,:,1] for i in (0,20,40)],2)
        records.append((patches(stream(raw),t),trial.clock.center(forward),trial.clock.center(backward)))
    x,y,b=[torch.from_numpy(np.concatenate(v)).cuda() for v in zip(*records)]
    bank={}
    for name in variants:
        torch.manual_seed(9031)
        net=Path(x.shape[-1]).cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.04)
        target=b.roll(17,1) if name=='wrong' else b
        for ep in range(7):
            for ix in torch.randperm(len(x),device='cuda').split(48):
                a=net(x[ix])
                loss=F.smooth_l1_loss(a,y[ix],beta=.9)
                if name!='direct':
                    back=net(reverse(x[ix],name!='naive'))
                    loss+=.7*F.smooth_l1_loss(back,target[ix],beta=.9)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),.8)
                opt.step()
            if ep in (3,5):opt.param_groups[0]['lr']*=.5
        bank.update({name+'/'+k:v.detach().cpu().numpy() for k,v in net.state_dict().items()})
    return bank


@lru_cache(maxsize=1)
def trained():
    return fit_paths(range(4), ('direct','matched','wrong','naive'))


@torch.inference_mode()
def encode(raw,m):
    s=stream(raw)
    net=Path(s.shape[-1]).cuda().eval()
    outputs=[]
    for name in ('direct','matched','wrong','naive'):
        net.load_state_dict({k.split('/',1)[1]:torch.from_numpy(v) for k,v in m.items() if k.startswith(name+'/')})
        rows=[]
        for t in np.array_split(np.arange(len(raw)),(len(raw)+95)//96):
            x=torch.from_numpy(patches(s,t)).cuda()
            a=net(x)[:,:,1:]
            if name=='matched':
                b=net(reverse(x,True))[:,:,1:]
                a=torch.cat((a,.5*(a+b),.5*(a-b)),2)
            rows.append(a.cpu().numpy())
        outputs.append(np.concatenate(rows))
    return np.concatenate(outputs,2)


def state(raw):
    return encode(raw,trained())


def record_with_state(z,p,y,d):
    t=np.arange(72+d%7,len(p)-60,7)
    q=trial.clock.center(p[t])
    x=np.concatenate((q,z[t]),2).reshape(len(t)*50,-1)
    o=trial.clock.center(y[t,:,2])
    f=q[:,:,2]
    b=(f*o).sum(dtype=float)/(f*f).sum(dtype=float).clip(1e-7)
    return x.T@x/len(x),x.T@(o/b.clip(.4,2.)-f).ravel()/len(x)


def warm_history():
    rows=[]
    for d in range(4,8):
        raw=trial.clock.book.rawday(d)
        rows.append(record_with_state(state(raw),early.load(d,raw),trial.clock.labels(d).reshape(-1,50,3),d))
    return rows


def readout(hist):
    g,b=[np.stack(v).astype(float) for v in zip(*hist)]
    gram=g.mean(0)
    def solve(cols,n=3):
        basis=np.eye(len(gram))[:,cols]
        basis[:n]=-trial.curve.solve(gram[:n,:n],gram[:n,cols],.03)
        days=b@basis
        mu=days.mean(0)
        mu*=abs(mu)/(abs(mu)+days.std(0)/len(days)**.5+1e-7)
        return basis@trial.curve.solve(basis.T@gram@basis,mu,.85)
    effect=solve([7,8,9,10],5)
    weights=np.column_stack((solve([3,4]),solve([5,6]),solve([7,8]),solve([9,10]),solve([7,8,9,10]),
                             solve([11,12]),solve([13,14]),effect,.5*effect))
    return (.44*weights).astype('float32')


def fit(hist):
    return {**trained(),'w':readout(hist)}


def changes_with_reference(z,m,p):
    return list(np.moveaxis(np.concatenate((trial.clock.center(p),z),2)@m['w'],2,0))


def select(f,d):
    good=[]
    for i in (1,2,3,4,7,8):
        v=f[:,:,0,i]
        if ((v>0).sum(0).min()>=3 and v.mean(0).min()>0 and v[-1].min()>0 and (d[:,0,i]>0).sum()>=5
                and f[:,:,1,i].mean(0).min()>0 and all((v-f[:,:,0,k]).mean(0).min()>0 for k in (0,5))):
            good.append(i)
    return int(max(good,key=lambda i:f[:,0,0,i].mean())) if good else None


def correction():
    with np.load(OUT/'model.npz') as f:
        m={k:f[k] for k in f.files if k!='pick'}
        i=int(f['pick'])
    return lambda raw,p:changes_with_reference(encode(raw,m),m,p)[i]


def layer():
    return history.layer(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2)
    history.run(sys.modules[__name__])
