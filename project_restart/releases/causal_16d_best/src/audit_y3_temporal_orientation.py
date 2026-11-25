"""Decompose the selected encoder's odd response into order reversal and signed-price reversal on rolling dates."""
import json
from functools import lru_cache
import numpy as np
import torch
import experiment_y3_two_sided_path_encoder_v1 as model

trial=model.trial
OUT=model.OUT/'orientation'
ARMS=['original odd response','order component','price component']


@lru_cache(maxsize=1)
def encoder():
    # Only weights pretrained on dates 1-4 are loaded; the 16-date supervised readout is excluded.
    with np.load(model.OUT/'model.npz') as f:
        return {k.split('/',1)[1]:torch.from_numpy(f[k]) for k in f.files if k.startswith('matched/')}


@torch.inference_mode()
def parts(raw,weights=None):
    s=model.stream(raw)
    net=model.Path(s.shape[-1]).cuda().eval()
    net.load_state_dict(encoder() if weights is None else weights)
    blocks=[]
    for t in np.array_split(np.arange(len(raw)),(len(raw)+95)//96):
        x=torch.from_numpy(model.patches(s,t)).cuda()
        flip=x.clone()
        flip[:,:,:,0]*=-1
        a,b,c,d=[net(v)[:,:,1:] for v in (x,x.flip(2),flip,flip.flip(2))]
        order=.25*((a-b)+(c-d))
        price=.25*((a-c)+(b-d))
        torch.testing.assert_close(order+price,.5*(a-d),rtol=0,atol=5e-7)
        blocks.append(torch.cat((order,price),2).cpu().numpy())
    return np.concatenate(blocks)


def record(z,p,y,d):
    t=np.arange(72+d%7,len(p)-60,7)
    q=trial.clock.center(p[t])
    x=np.concatenate((q,z[t]),2).reshape(len(t)*50,-1)
    actual=trial.clock.center(y[t,:,2])
    f=q[:,:,2]
    beta=(actual*f).sum(dtype=float)/(f*f).sum(dtype=float).clip(1e-7)
    return x.T@x/len(x),x.T@(actual/beta.clip(.4,2.)-f).ravel()/len(x)


def readout(hist):
    gs,b=[np.stack(v).astype(float) for v in zip(*hist)]
    g=gs.mean(0)
    base=np.eye(7)[:,3:]
    base[:3]=-trial.curve.solve(g[:3,:3],g[:3,3:],.03)
    basis=base[:,:2]+base[:,2:]
    days=b@basis
    mu=days.mean(0)
    mu*=abs(mu)/(abs(mu)+days.std(0)/len(days)**.5+1e-7)
    coef=.44*trial.curve.solve(basis.T@g@basis,mu,.85)
    return np.stack((basis@coef,base[:,:2]@coef,base[:,2:]@coef),1).astype('float32')


def main():
    OUT.mkdir(exist_ok=True)
    hist,folds=[],[]
    expected=json.loads((model.OUT/'results.json').read_text())
    for d in range(4,16):
        raw=trial.clock.book.rawday(d)
        z=parts(raw)
        y=trial.clock.labels(d).reshape(-1,50,3)
        refs=[model.early.load(d,raw)] if d<8 else model.history.references(d,raw)
        records=[]
        if d>=8 and d%2==0:
            w=readout(hist)
            groups=[[],[]]
        for j,p in enumerate(refs):
            records.append(record(z,p,y,d))
            if d>=8:
                x=np.concatenate((trial.clock.center(p),z),2)
                delta=x@w
                np.testing.assert_allclose(delta[:,:,0],delta[:,:,1]+delta[:,:,2],atol=1e-6,rtol=1e-5)
                groups[j].append([v.sum(0) for v in trial.previous.statistics(p,y,list(np.moveaxis(delta,2,0)))])
        hist.append([np.mean(v,0) for v in zip(*records)])
        if d>=8 and d%2:
            found=np.array([[trial.scoring.audit.pooled(s) for s in np.sum(g,0)] for g in groups])
            wanted=np.asarray(expected['folds'][(d-9)//2])[:,:,3]
            np.testing.assert_allclose(found[:,:,0],wanted,rtol=0,atol=8e-7)
            folds.append(found.tolist())
    out=dict(arms=ARMS,dates=[[9,10],[11,12],[13,14],[15,16]],outcomes=['Y3','late','middle','terminal'],folds=folds)
    (OUT/'results.json').write_text(json.dumps(out,indent=2))
    print(np.array(folds)[:,0,0,:].tolist(),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
