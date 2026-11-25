"""Train a smooth horizon-conditioned orientation encoder on all past dates and overlapping twenty-tick outcomes."""
from functools import lru_cache
import sys
import numpy as np
import torch
from torch.nn import functional as F

import y3_path_orientation_validation as history

parent=history.incumbent
trial=history.trial
OUT=trial.clock.ART/'y3_dense_orientation_curve_v1'
ARMS=['sparse expanding encoder','dense horizon curve','dense final window','dense versus sparse',
      'dense curvature','wrong backward windows','dense forward only','half dense increment']
OFFSETS=np.arange(0,41,5)
PHI=np.stack([np.cos(np.pi*k*OFFSETS/40) for k in range(3)],0).astype('float32')


@lru_cache(maxsize=4)
def train(end,names=('sparse','dense','wrong')):
    records=[]
    for d in range(end):
        raw=trial.clock.book.rawday(d)
        y=trial.clock.labels(d).reshape(-1,50,3)
        t=np.arange(130+d%11,len(y)-60,11)
        f=y[t[:,None]+OFFSETS,:,1].transpose(0,2,1)
        b=-y[t[:,None]-84-OFFSETS,:,1].transpose(0,2,1)
        records.append((parent.patches(parent.stream(raw),t),trial.clock.center(f),trial.clock.center(b)))
    x,y,b=[torch.from_numpy(np.concatenate(v)).cuda() for v in zip(*records)]
    del records
    phi=torch.from_numpy(PHI).cuda()
    bank={}
    for name in names:
        torch.manual_seed(3586)
        net=parent.Path(x.shape[-1]).cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.055)
        ids=[0,4,8] if name=='sparse' else list(range(9))
        reverse_target=b.flip(2) if name=='wrong' else b
        for ep in range(6):
            for idx in torch.randperm(len(x),device='cuda').split(48):
                front=net(x[idx])@phi
                back=net(parent.reverse(x[idx],True))@phi
                loss=F.smooth_l1_loss(front[:,:,ids],y[idx][:,:,ids],beta=.9)
                loss+=.7*F.smooth_l1_loss(back[:,:,ids],reverse_target[idx][:,:,ids],beta=.9)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),.85)
                opt.step()
            if ep in (3,4):
                opt.param_groups[0]['lr']*=.52
        bank.update({name+'/'+k:v.detach().cpu().numpy() for k,v in net.state_dict().items()})
    return bank


@torch.inference_mode()
def encode(raw,m):
    s=parent.stream(raw)
    phi=torch.from_numpy(PHI).cuda()
    net=parent.Path(s.shape[-1]).cuda().eval()
    blocks=[]
    for name in ('sparse','dense','wrong'):
        net.load_state_dict({k.split('/',1)[1]:torch.from_numpy(v) for k,v in m.items() if k.startswith(name+'/')})
        rows=[]
        for t in np.array_split(np.arange(len(raw)),(len(raw)+95)//96):
            x=torch.from_numpy(parent.patches(s,t)).cuda()
            a=net(x)@phi
            odd=.5*(a-net(parent.reverse(x,True))@phi)
            views=[odd[:,:,[4,8]]]
            if name=='dense':
                views.extend((odd[:,:,4:5]-.5*(odd[:,:,:1]+odd[:,:,8:9]),a[:,:,[4,8]]))
            rows.append(torch.cat(views,2).cpu().numpy())
        blocks.append(np.concatenate(rows))
    z=np.concatenate(blocks,2)
    return np.concatenate((z,z[:,:,2:4]-z[:,:,:2]),2)


def historical_state(raw,d):
    return encode(raw,train(4*(d//4)))


def state(raw):
    return historical_state(raw,16)


def record_with_state(z,p,y,d):
    t=np.arange(72+d%7,len(y)-60,7)
    q=trial.clock.center(p[t])
    x=np.concatenate((q,z[t]),2).reshape(len(t)*50,-1)
    actual=trial.clock.center(y[t,:,2])
    f=q[:,:,2]
    scale=(actual*f).sum(dtype=float)/(f*f).sum(dtype=float).clip(1e-7)
    return x.T@x/len(x),x.T@(actual/scale.clip(.4,2.)-f).ravel()/len(x)


def warm_history():
    rows=[]
    for d in range(4,8):
        raw=trial.clock.book.rawday(d)
        rows.append(record_with_state(historical_state(raw,d),parent.early.load(d,raw),trial.clock.labels(d).reshape(-1,50,3),d))
    return rows


def fit(hist):
    gs,b=[np.stack(v).astype(float) for v in zip(*hist)]
    g=gs.mean(0)
    def solve(ids,n=3):
        w=np.eye(len(g))[:,ids]
        w[:n]=-trial.curve.solve(g[:n,:n],g[:n,ids],.04)
        days=b@w
        mu=days.mean(0)
        mu*=abs(mu)/(abs(mu)+days.std(0)/len(days)**.5+1e-7)
        return w@trial.curve.solve(w.T@g@w,mu,.85)
    effect=solve([12,13],5)
    w=np.column_stack((solve([3,4]),solve([5,6]),solve([6]),effect,solve([7]),solve([10,11]),
                       solve([8,9]),.5*effect))
    m=dict(w=(.40*w).astype('float32'))
    if len(hist)==12:
        m.update(train(16))
    return m


def changes_with_reference(z,m,p):
    return list(np.moveaxis(np.concatenate((trial.clock.center(p),z),2)@m['w'],2,0))


def select(f,d):
    good=[]
    for i in (1,2,3,4,7):
        v=f[:,:,0,i]
        if ((v>0).sum(0).min()>=3 and v.mean(0).min()>0 and v[-1].min()>0 and (d[:,0,i]>0).sum()>=5
                and f[:,:,1,i].mean(0).min()>0 and all((v-f[:,:,0,k]).mean(0).min()>0 for k in (0,5))):
            good.append(i)
    return int(max(good,key=lambda i:f[:,0,0,i].mean())) if good else None


def correction():
    with np.load(OUT/'model.npz') as f:
        m={k:f[k] for k in f.files if k!='pick'}
        pick=int(f['pick'])
    return lambda raw,p:changes_with_reference(encode(raw,m),m,p)[pick]


def layer():
    return history.layer(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2)
    history.run(sys.modules[__name__])
