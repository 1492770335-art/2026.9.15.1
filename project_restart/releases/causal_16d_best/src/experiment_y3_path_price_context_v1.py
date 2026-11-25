"""Attribute signed-price path responses to old and recent returns under the observed and peer book histories."""
import sys

import numpy as np
import torch

import audit_y3_temporal_orientation as orientation
import y3_path_orientation_validation as history

parent=history.incumbent
trial=history.trial
OUT=trial.clock.ART/'y3_path_price_context_v1'
ARMS=['linear old and recent prices','price response in typical book','own price and book response','book interaction',
      'old-specific interaction','recent-specific interaction','misassigned interaction','half book interaction']


def split_response(net,x):
    def averaged(a):
        return .5*(net(a)[:,:,1:]+net(a.flip(2))[:,:,1:])
    a=averaged(x)
    old=x.clone()
    old[:,:,:4,0]*=-1
    b=averaged(old)
    recent=x.clone()
    recent[:,:,4:,0]*=-1
    c=averaged(recent)
    both=old.clone()
    both[:,:,4:,0]*=-1
    d=averaged(both)
    return torch.cat((.25*(a-b+c-d),.25*(a+b-c-d)),2)


@torch.inference_mode()
def state(raw):
    stream=parent.stream(raw)
    net=parent.Path(stream.shape[-1]).cuda().eval()
    net.load_state_dict(orientation.encoder())
    rows=[]
    for idx in np.array_split(np.arange(len(raw)),(len(raw)+95)//96):
        x=torch.from_numpy(parent.patches(stream,idx)).cuda()
        own=split_response(net,x)
        comparisons=[]
        for shift in (13,-13):
            other=x.roll(shift,1)
            other[:,:,:,0]=x[:,:,:,0]
            comparisons.append(split_response(net,other))
        typical=.5*(comparisons[0]+comparisons[1])
        linear=torch.stack((x[:,:,:4,0].float().mean(2),x[:,:,4:,0].float().mean(2)),2)
        difference=own-typical
        features=torch.cat((linear,typical,own,difference,difference.roll(17,1)),2)
        rows.append((features-features.mean(1,keepdim=True)).cpu().numpy())
    return np.concatenate(rows)


def design(z,p):
    return np.concatenate((trial.clock.center(p),z),2)


def record_with_state(z,p,y,d):
    t=np.arange(72+d%7,len(p)-60,7)
    q=trial.clock.center(p[t,:,2])
    obs=trial.clock.center(y[t,:,2])
    b=(q*obs).sum(dtype=float)/(q*q).sum(dtype=float).clip(1e-7)
    x=design(z[t],p[t]).reshape(len(t)*50,-1)
    return x.T@x/len(x),x.T@(obs/b.clip(.4,2.)-q).ravel()/len(x)


def warm_history():
    rows=[]
    for d in range(4,8):
        raw=trial.clock.book.rawday(d)
        rows.append(record_with_state(state(raw),parent.early.load(d,raw),trial.clock.labels(d).reshape(-1,50,3),d))
    return rows


def fit(hist):
    gs,b=[np.stack(v).astype(float) for v in zip(*hist)]
    g=gs.mean(0)
    def solve(cols,n):
        w=np.eye(len(g))[:,cols]
        w[:n]=-trial.curve.solve(g[:n,:n],g[:n,cols],.035)
        days=b@w
        mu=days.mean(0)
        mu*=abs(mu)/(abs(mu)+days.std(0)/len(days)**.5+1e-7)
        return w@trial.curve.solve(w.T@g@w,mu,.8)
    interaction=solve([13,14,15,16],9)
    weights=np.column_stack((solve([3,4],3),solve([5,6,7,8],5),solve([9,10,11,12],5),interaction,
                             solve([13,14],9),solve([15,16],9),solve([17,18,19,20],9),.5*interaction))
    return dict(w=(.40*weights).astype('float32'))


def changes_with_reference(z,m,p):
    return list(np.moveaxis(design(z,p)@m['w'],2,0))


def select(f,d):
    good=[]
    for i in (2,3,4,5,7):
        v=f[:,:,0,i]
        if ((v>0).sum(0).min()>=3 and v.mean(0).min()>0 and v[-1].min()>0 and (d[:,0,i]>0).sum()>=5
                and f[:,:,1,i].mean(0).min()>0 and all((v-f[:,:,0,k]).mean(0).min()>0 for k in (0,6))):
            good.append(i)
    return int(max(good,key=lambda i:f[:,0,0,i].mean())) if good else None


def correction():
    return history.correction(sys.modules[__name__])


def layer():
    return history.layer(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2)
    history.run(sys.modules[__name__])
