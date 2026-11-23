"""Measure a price move's age by subsequent observed quote activity while preserving its original book state."""
import json
import numpy as np
import experiment_event_state_decay_v1 as fixed

base,peer=fixed.base,fixed.peer
OUT=peer.ART/'activity_clock_event_decay_v1'


def rates():
    return np.median([(abs(base.paths.book.rawday(d)[:,:,0].astype(float)-.00136)>.022).mean(0)
                      for d in range(8)],axis=0).clip(.04,.95)


def dynamic(raw,t,rate):
    raw_ret=raw[:,:,0].astype(float)-.00136
    active=(abs(raw_ret)>.022).astype(float)
    speed=.35+.65*active/rate
    n=len(raw)
    cs=np.vstack((np.zeros(50),active.cumsum(0)))
    left=(t+1-160).clip(0)
    intensity=(cs[t+1]-cs[left]+100*rate)/((t+1-left)[:,None]+100)
    expected=.35+.65*intensity/rate
    r=fixed.votes.center(raw_ret).astype('float32')
    def f(i):return np.arcsinh(raw[:,:,i].astype('float32'))
    marks=np.stack((f(41)-f(42),f(54)-f(55),f(77)-f(78),f(4),f(3),f(5),f(8),f(39)),axis=2)
    gate=np.tanh(marks)[(np.arange(n)-1).clip(0)]
    gate=np.concatenate((gate,abs(gate)),axis=2)
    stream=np.concatenate((r[:,:,None],r.clip(0)[:,:,None]*gate,r.clip(None,0)[:,:,None]*gate),axis=2)
    # Save only the requested origins during the causal recurrence.
    z=np.empty((len(t),50,len(fixed.HALF),stream.shape[2]),np.float32)
    for i,half in enumerate(fixed.HALF):
        retention=(2**(-speed/half)).astype('float32')
        state=np.zeros_like(stream[0])
        pos=0
        for k in range(n):
            state*=retention[k,:,None]
            state+=stream[k]
            if pos<len(t) and t[pos]==k:
                z[pos,:,i]=state/np.sqrt(half)
                pos+=1
        assert pos==len(t)
    return z,expected


def columns(z,expected,h):
    e=fixed.UNITS[h]*(1-2**(-fixed.HORIZONS[h]*expected[:,:,None]/fixed.HALF))
    return fixed.votes.center((z*e[:,:,:,None]).reshape(*z.shape[:2],-1))


def stats(d,rate):
    raw=base.paths.book.rawday(d);t=np.arange(d%7,len(raw)-60,7)
    z,speed=dynamic(raw,t,rate)
    y=fixed.votes.center(peer.labels(d).reshape(-1,50,3)[t])
    g=0.;r=0.
    for h in range(3):
        x=columns(z,speed,h).reshape(y[:,:,h].size,-1)
        g+=x.T@x/len(x);r+=x.T@y[:,:,h].ravel()/len(x)
    return [*fixed.stat(d),(g,r)]


def forecast(raw,t,rate,w):
    a,b=fixed.inputs(raw,t)
    z,speed=dynamic(raw,t,rate)
    result=[[],[]]
    for h in range(3):
        dyn=columns(z,speed,h)@w[2]
        for buf,x,wi in zip(result,(a,b),w[:2]):
            buf.append(.20*fixed.votes.center(dyn-fixed.features(x,h)@wi))
    return [np.stack(s,2).astype('float32') for s in result]


def screen():
    rate=rates()
    history=[];days=[];folds=[]
    for end in (8,10,12,14):
        history.extend(stats(d,rate) for d in range(len(history),end))
        w=fixed.fit(history)
        groups=[[[],[]] for _ in range(2)]
        for d in (end,end+1):
            raw=base.paths.book.rawday(d);t=np.arange(d%4,len(raw),4)
            y=peer.labels(d).reshape(-1,50,3)[t]
            adj=forecast(raw,t,rate,w)
            if d==end:
                st=t[t<193]
                for a,b in zip(adj,forecast(raw[:193],st,rate,w)):
                    np.testing.assert_allclose(a[:len(st)],b,rtol=0,atol=2e-6)
            for group,ref in zip(groups,(peer.reference,fixed.earlier.early.baseline)):
                p=ref(d).reshape(-1,50,3)[t]
                for rows,z in zip(group,adj):rows.append(tuple(x.reshape(-1,3) for x in (p,p+z,y)))
            days.append([[peer.delta(r[-1:]) for r in group] for group in groups])
        folds.append([[peer.delta(r) for r in group] for group in groups])
        print(end+1,folds[-1][0],flush=True)
    f=np.array(folds)
    keep=(f[:,0,0].min(0)>0)&(f[:,1,0].mean(0)>0)&(f[:,0,1].mean(0)>0)&((np.array(days)[:,0,0]>0).sum(0)>=5)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=days,retained=keep.tolist()),indent=2))
    if keep.any():
        history.extend(stats(d,rate) for d in (14,15))
        np.savez(OUT/'model.npz',rate=rate,w=fixed.fit(history),mask=keep)
    return keep.any()


def load():
    parent=base.load()
    with np.load(OUT/'model.npz') as st:rate,w,keep=st['rate'],st['w'],st['mask']
    def predict(raw,p):
        q=parent(raw,p).copy()
        for k in range(4):
            t=np.arange(k,len(q),4)
            q[t]+=forecast(raw,t,rate,w)[0]*keep
        return q
    return predict


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if screen():fixed.votes.follow.replay_candidate(OUT,load)
