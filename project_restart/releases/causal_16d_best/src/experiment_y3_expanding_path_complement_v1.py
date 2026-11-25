"""Estimate what an expanding, horizon-conditioned encoder adds after the recent own-book correction."""
import sys
from functools import lru_cache
import numpy as np
import torch

import audit_y3_temporal_orientation as orientation
import experiment_y3_dense_orientation_curve_v1 as dense
import y3_path_context_validation as history

context=history.incumbent
path=dense.parent
trial=history.trial
OUT=trial.clock.ART/'y3_expanding_path_complement_v1'
ARMS=['early path and book','expanded path','expanded beyond early path','expanded beyond own book',
      'expanded terminal residual','shifted expanded path','midpoint and terminal residual','half orthogonal addition']


@lru_cache(maxsize=4)
def weights(end):
    if end==16 and (dense.OUT/'model.npz').exists():
        with np.load(dense.OUT/'model.npz') as f:
            return {k:f[k] for k in f.files if k.startswith('dense/')}
    return dense.train(end,('dense',))


@torch.inference_mode()
def encode(raw,m,only_dense=False):
    s=path.stream(raw)
    if not only_dense:
        first=path.Path(s.shape[-1]).cuda().eval()
        first.load_state_dict(orientation.encoder())
    net=path.Path(s.shape[-1]).cuda().eval()
    net.load_state_dict({k.split('/',1)[1]:torch.from_numpy(v) for k,v in m.items() if k.startswith('dense/')})
    phi=torch.from_numpy(dense.PHI).cuda()
    rows=[]
    for t in np.array_split(np.arange(len(raw)),(len(raw)+95)//96):
        x=torch.from_numpy(path.patches(s,t)).cuda()
        reverse=path.reverse(x,True)
        response=.5*((net(x)-net(reverse))@phi)
        forecast=response[:,:,[4,8]]
        if only_dense:
            # Keep the original feature layout and matrix multiplication exactly; unused channels have zero readout weight.
            compact=forecast.new_zeros(len(x),50,9)
            compact[:,:,4:6]=forecast
            rows.append(compact.cpu().numpy())
            continue
        original=.5*(first(x)-first(reverse))[:,:,1:]
        own=context.split_response(first,x)[:,:,2:]
        for k in (-13,13):
            peer=x.roll(k,1)
            peer[:,:,:,0]=x[:,:,:,0]
            own=own-.5*context.split_response(first,peer)[:,:,2:]
        rows.append(torch.cat((original,own,forecast,forecast.roll(17,1),response[:,:,6:7]),2).cpu().numpy())
    return np.concatenate(rows)


def historical_state(raw,d):
    return encode(raw,weights(4*(d//4)))


def state(raw):
    return historical_state(raw,16)


record_with_state=path.record_with_state
changes_with_reference=path.changes_with_reference


def warm_history():
    rows=[]
    for d in range(4,8):
        raw=trial.clock.book.rawday(d)
        rows.append(record_with_state(historical_state(raw,d),path.early.load(d,raw),trial.clock.labels(d).reshape(-1,50,3),d))
    return rows


def fit(hist):
    gs,b=[np.stack(v).astype(float) for v in zip(*hist)]
    g=gs.mean(0)
    def solve(cols,n):
        s=np.eye(len(g))[:,cols]
        s[:n]=-trial.curve.solve(g[:n,:n],g[:n,cols],.04)
        cov=b@s
        mu=cov.mean(0)
        mu*=abs(mu)/(abs(mu)+cov.std(0)/len(cov)**.5+1e-7)
        return s@trial.curve.solve(s.T@g@s,mu,.85)
    orth=solve([7,8],7)
    w=np.column_stack((solve([3,4,5,6],3),solve([7,8],3),solve([7,8],5),orth,solve([8],7),
                       solve([9,10],7),solve([8,11],7),.5*orth))
    result=dict(w=(.40*w).astype('float32'))
    if len(hist)==12:result.update(weights(16))
    return result


def select(f,d):
    good=[]
    for i in (1,2,3,4,6,7):
        v=f[:,:,0,i]
        if ((v>0).sum(0).min()>=3 and v.mean(0).min()>0 and v[-1].min()>0 and (d[:,0,i]>0).sum()>=5
                and f[:,:,1,i].mean(0).min()>0 and (v-f[:,:,0,5]).mean(0).min()>0):
            good.append(i)
    return int(max(good,key=lambda i:f[:,0,0,i].mean())) if good else None


def correction():
    with np.load(OUT/'model.npz') as f:
        m={k:f[k] for k in f.files if k!='pick'}
        pick=int(f['pick'])
    keep=np.r_[0:3,7:9]
    only_dense=not np.any(np.delete(m['w'][:,pick],keep))
    return lambda raw,p:changes_with_reference(encode(raw,m,only_dense),m,p)[pick]


def layer():
    return history.layer(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2)
    history.run(sys.modules[__name__])
