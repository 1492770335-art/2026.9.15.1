"""Separate missed, reversed and overshooting one-tick price expectations in a frozen path encoder."""
import sys
import numpy as np
import torch

import y3_surprise_references as history

prev=history.incumbent
scale,path,trial=prev.scale,prev.path,prev.trial
OUT=trial.clock.ART/'y3_unfulfilled_price_moves_v1'
ARMS=('aggregate surprise','unfilled expectation','opposite direction','overshoot','deferred confirmation',
      'unrelated confirmation','persistent miss','three types','half persistent miss','unfilled beyond aggregate')


def components(raw,w):
    expected=prev.predicted(raw,w)
    actual=raw[:,:,0].astype('float32')-.00136
    same=actual*expected>0
    fulfilled=np.where(same,np.sign(expected)*np.minimum(abs(expected),abs(actual)),0)
    missed=fulfilled-expected
    opposite=np.where(same,0.,actual)
    extra=np.where(same,actual-fulfilled,0.)
    np.testing.assert_allclose(missed+opposite+extra,actual-expected,atol=2e-5,rtol=1e-6)
    # Release information about an unfilled expectation only after the four follow-up ticks have elapsed.
    lag=4
    confirmed=np.zeros_like(actual)
    persistent=np.zeros_like(actual)
    if len(raw)>lag:
        cs=np.vstack((np.zeros(50),actual.cumsum(0,dtype=float)))
        t=np.arange(lag,len(raw))
        subsequent=cs[t+1]-cs[t+1-lag]
        empty=abs(actual[t-lag])<.022
        issued=missed[t-lag]*empty
        paid=subsequent*expected[t-lag]>0
        confirmed[t]=issued*paid
        persistent[t]=issued*(~paid)
    return actual,(actual-expected,missed,opposite,extra,confirmed,np.roll(confirmed,17,axis=1),persistent)


@torch.inference_mode()
def encode(raw,m):
    actual,parts=components(raw,m['teacher'])
    series=[prev.make_price(actual-2*z) for z in parts]
    stream=path.stream(raw)
    net=scale.MagnitudePath(44).cuda().eval()
    net.load_state_dict({k[5:]:torch.from_numpy(v) for k,v in m.items() if k.startswith('fine/')})
    phi=torch.from_numpy(scale.base.dense.PHI[:,[4,8]].copy()).cuda()
    def head(x):
        a,va=net(x,phi)
        b,vb=net(path.reverse(x,True),phi)
        return .5*(a*va.exp()-b*vb.exp())
    rows=[]
    for t in np.array_split(np.arange(len(raw)),(len(raw)+79)//80):
        x=torch.from_numpy(path.patches(stream,t)).cuda()
        original=head(x)
        end=t[:,None]+1-8*np.arange(7,-1,-1)
        start=(end-8).clip(0);end=end.clip(0)
        pieces=[original]
        for s in series:
            patch=x.clone()
            v=(s[end]-s[start])/(end-start).clip(1)[:,:,None]
            patch[:,:,:,0]=torch.from_numpy(v.transpose(0,2,1).astype('float16')).cuda()
            pieces.append(.5*(original-head(patch)))
        out=torch.cat(pieces,2)
        rows.append((out-out.mean(1,keepdim=True)).cpu().numpy())
    return np.concatenate(rows)


def historical_state(raw,d):
    return encode(raw,prev.weights(4*(d//4)))


record_with_state=prev.record_with_state
changes_with_reference=prev.changes_with_reference


def warm_history():
    rows=[]
    for d in range(4,8):
        raw=trial.clock.book.rawday(d)
        y=trial.clock.labels(d).reshape(-1,50,3)
        rows.append(record_with_state(historical_state(raw,d),path.early.load(d,raw),y,d))
    return rows


def fit(rows):
    grams,moments=[np.stack(v).astype(float) for v in zip(*rows)]
    g=grams.mean(0)
    def solve(cols,n):
        basis=np.eye(len(g))[:,cols]
        basis[:n]=-trial.curve.solve(g[:n,:n],g[:n,cols],.05)
        daily=moments@basis
        mu=daily.mean(0)
        mu*=abs(mu)/(abs(mu)+daily.std(0)/np.sqrt(len(rows))+1e-7)
        return basis@trial.curve.solve(basis.T@g@basis,mu,.9)
    groups=[list(range(3+2*j,5+2*j)) for j in range(8)]
    _,total,miss,reverse,extra,confirmed,wrong,persistent=groups
    delay=solve(persistent,7)
    bank=[solve(total,5),solve(miss,5),solve(reverse,5),solve(extra,5),solve(confirmed,7),
          solve(wrong,7),delay,solve(miss+reverse+extra,7),.5*delay,solve(miss,7)]
    m=dict(w=(.40*np.column_stack(bank)).astype('float32'))
    if len(rows)==12:
        m.update(prev.weights(16))
    return m


def select(f,d):
    good=[]
    for j in (1,2,3,4,6,7,8,9):
        v=f[:,:,0,j]
        if ((v>0).sum(0).min()>=3 and v.mean(0).min()>0 and v[-1].min()>0 and
                (d[:,0,j]>0).sum()>=5 and f[:,:,1,j].mean(0).min()>0 and
                (v-f[:,:,0,5]).mean(0).min()>0):
            good.append(j)
    return max(good,key=lambda j:f[:,0,0,j].mean()) if good else None


def correction():
    with np.load(OUT/'model.npz') as f:
        m=dict(f)
    return lambda raw,p:changes_with_reference(encode(raw,m),m,p)[int(m['pick'])]


def layer():
    predict,update=prev.layer(),correction()
    def apply(raw,p):
        q=predict(raw,p)
        q[:,:,2]+=update(raw,q)
        return q
    return apply


if __name__=='__main__':
    torch.set_num_threads(2)
    history.run(sys.modules[__name__])
