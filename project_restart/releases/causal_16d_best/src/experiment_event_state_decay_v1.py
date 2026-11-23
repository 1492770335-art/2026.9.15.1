"""Compare persistence of a price move conditioned on the book before the move or at forecast time."""
import json
import numpy as np
from scipy.signal import lfilter
import experiment_pre_move_range_response_v1 as earlier

votes=earlier.votes
peer,base=votes.peer,votes.base
OUT=peer.ART/'event_state_decay_v1'
HALF=np.array((6,24,72),float)
HORIZONS=np.array((5,20,60),float)
UNITS=base.audit.SCALE


def inputs(raw,t):
    r=raw[:,:,0].astype('float32')-.00136
    r=votes.center(r)
    def f(j):
        return np.arcsinh(raw[:,:,j].astype('float32'))
    idx=(np.arange(len(r))-1).clip(0)
    a=np.stack((f(41)-f(42),f(54)-f(55),f(77)-f(78),f(4),f(3),f(5),f(8),f(39)),2)
    gates=np.tanh(a)[idx]
    abs_gate=np.abs(gates)
    proxy=np.concatenate((gates,abs_gate),2)
    pos=r.clip(0);neg=r.clip(None,0)
    out=[[],[]]
    for half in HALF:
        decay=2**(-1/half)
        def filt(x):
            return (lfilter([1./np.sqrt(half)],[1.,-decay],x,axis=0)[t]).astype('float32')
        left,right=filt(pos),filt(neg)
        # Both arms use a one-tick lag for the state. In the marked arm each return carries its own state.
        current=np.concatenate((left[:,:,None]*proxy[t],right[:,:,None]*proxy[t]),2)
        marked=np.concatenate((filt(pos[:,:,None]*proxy),filt(neg[:,:,None]*proxy)),2)
        for cols,state in zip(out,(current,marked)):
            cols.append(votes.center(np.concatenate(((left+right)[:,:,None],state),2)))
    return [np.stack(group,axis=2) for group in out]


def features(z,h):
    # Remaining decay over future h ticks is the difference of two exponential levels.
    exposure=UNITS[h]*(1-2**(-HORIZONS[h]/HALF))
    return (z*exposure[None,None,:,None]).reshape(*z.shape[:2],-1)


def stat(day):
    raw=base.paths.book.rawday(day)
    t=np.arange(day%7,len(raw)-60,7)
    y=votes.center(peer.labels(day).reshape(-1,50,3)[t])
    parts=[]
    for z in inputs(raw,t):
        g=0.;b=0.
        for h in range(3):
            x=features(z,h).reshape(y[:,:,h].size,-1)
            target=y[:,:,h].ravel().astype(float)
            g+=x.T@x/len(x);b+=x.T@target/len(x)
        parts.append((g,b))
    return parts


def fit(history):
    ws=[]
    for part in zip(*history):
        grams,rhs=(np.stack(a) for a in zip(*part))
        g=grams.mean(0);reg=.8*np.diag(g.diagonal().clip(1e-5))
        day=np.linalg.solve(grams+reg,rhs[:,:,None])[:,:,0]
        ws.append(.7*np.linalg.solve(g+reg,rhs.mean(0))+.3*np.median(day,axis=0))
    return ws


def predict_change(raw,t,ws):
    plain,marked=inputs(raw,t)
    return np.stack([.20*votes.center(features(marked,h)@ws[1]-features(plain,h)@ws[0]) for h in range(3)],2).astype('float32')


def screen():
    history,folds,days=[],[],[]
    for end in (8,10,12,14):
        history.extend(stat(d) for d in range(len(history),end))
        w=fit(history);rows=[[],[]]
        for d in (end,end+1):
            raw=base.paths.book.rawday(d);t=np.arange(d%4,len(raw),4)
            z=predict_change(raw,t,w);y=peer.labels(d).reshape(-1,50,3)[t]
            if d==end:
                st=t[t<219]
                np.testing.assert_allclose(z[:len(st)],predict_change(raw[:219],st,w),atol=1e-6,rtol=0)
            for group,ref in zip(rows,(peer.reference,earlier.early.baseline)):
                p=ref(d).reshape(-1,50,3)[t]
                group.append(tuple(a.reshape(-1,3) for a in (p,p+z,y)))
            days.append([peer.delta(r[-1:]) for r in rows])
        folds.append([peer.delta(r) for r in rows]);print(end+1,folds[-1],flush=True)
    f=np.array(folds)
    keep=(f[:,0].min(0)>0)&(f[:,1].mean(0)>0)&((np.array(days)[:,0]>0).sum(0)>=5)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=days,retained=keep.tolist()),indent=2))
    if keep.any():
        history.extend(stat(d) for d in (14,15))
        np.savez(OUT/'model.npz',w=fit(history),mask=keep)
    return keep.any()


def load():
    parent=base.load()
    with np.load(OUT/'model.npz') as st:w,mask=st['w'],st['mask']
    def predict(raw,p):
        q=parent(raw,p).copy()
        for k in range(4):
            t=np.arange(k,len(q),4)
            q[t]+=predict_change(raw,t,w)*mask
        return q
    return predict


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if screen():votes.follow.replay_candidate(OUT,load)
