"""Compare present and preceding turns in the 50-stock forecast direction at several lags."""
import json
import sys

import numpy as np

import y3_settled_rank_probability_references as history

parent,base = history.incumbent,history.base
OUT = parent.OUT.parent/'y3_angular_forecast_history_v1'
LAGS = (2,8,24,64)
ARMS = ('unchanged','current angular velocity','misassigned stock curvature','curvature',
        'velocity and curvature','persistent turns','late versus early rotation','directions at multiple speeds')


def training_references(day,raw):
    if day>=8:
        return history.references(day,raw)
    return history.previous.previous.previous.incumbent.parent.warm.read(day,raw)


def dot(a,b):
    return np.sum(a*b,axis=1,keepdims=True)


def unit(x):
    return x/np.sqrt(dot(x,x)).clip(1e-8)


def residual(x,*axes):
    z = x-x.mean(1,keepdims=True)
    for v in axes:
        z = z-dot(z,v)*v
    return z


def logmap(u,v):
    c = dot(u,v).clip(-.999999,1.)
    a = np.arccos(c)
    return (v-c*u)*(a/np.sin(a).clip(1e-6)).clip(.99,4.)


def motion(u,v,w):
    current = -logmap(u,v)
    old = -logmap(v,w)
    transported = old-dot(old,u)/(1+dot(u,v)).clip(.1)*(u+v)
    return current,current-transported,transported


def design(p):
    a = unit(residual(p[:,:,1].astype(float)))
    far = residual(p[:,:,2].astype(float),a)
    length = np.sqrt(dot(far,far))
    u = unit(far)
    parts = []
    for lo,hi in base.segments(len(p)):
        t = np.arange(lo,hi)
        h,z = a[t],u[t]
        vs,cs,wrong,persistent,early = [],[],[],[],[]
        for lag in LAGS:
            s,r = (t-lag).clip(lo),(t-2*lag).clip(lo)
            velocity,curvature,old = motion(z,u[s],u[r])
            _,nc,_ = motion(h,a[s],a[r])
            live = (t>=lo+2*lag)[:,None]
            def axis(v):
                return length[t]*residual(v,h,z)*live
            vs.append(axis(velocity))
            cs.append(axis(curvature))
            wrong.append(axis(np.roll(curvature,13,axis=1)))
            alignment = dot(unit(old),unit(velocity)).clip(-1,1)
            persistent.append(axis(velocity*alignment))
            early.append(axis(nc))
        parts.append(np.stack(vs+wrong+cs+persistent+early,2))
    return np.concatenate(parts),far,length


GROUPS = (range(4),range(4,8),range(8,12),[*range(4),*range(8,12)],range(12,16),
          [*range(8,12),*range(16,20)],[*range(4),*range(8,20)])


def observe(x,y):
    targets = np.stack((y[:,:,2],y[:,:,2]-base.RATIO*y[:,:,1]),2).astype(float)
    z = x.reshape(-1,x.shape[2])
    return z.T@z/len(z),z.T@targets.reshape(-1,2)/len(z)


def fit(rows):
    gc,bc = [np.array(v) for v in zip(*rows)]
    g = gc.mean(0)
    b = bc.mean(0)
    b *= abs(b)/(abs(b)+bc.std(0)/np.sqrt(len(bc))+1e-6)
    target = np.where(b[:,0]*b[:,1]>0,np.sign(b[:,0])*np.minimum(abs(b[:,0]),abs(b[:,1])),0.)
    weights = []
    for group in GROUPS:
        idx = list(group)
        w = np.zeros(len(g))
        gg = g[np.ix_(idx,idx)]
        w[idx] = np.linalg.solve(gg+1.8*np.diag(gg.diagonal().clip(.00001)),target[idx])
        weights.append(w)
    return np.stack(weights,1)


def apply(x,far,length,w):
    tangent = x@w
    norm = np.sqrt((tangent*tangent).sum(1,keepdims=True))
    amount = np.minimum(.32,.16*length[:,:,None]/norm.clip(1e-9))
    revised = far[:,:,None]+amount*tangent
    return (length[:,:,None]*revised/np.sqrt(np.square(revised).sum(1,keepdims=True)).clip(1e-8)-far[:,:,None]).astype('float32')


def train():
    OUT.mkdir(exist_ok=True)
    rows,folds,daily = [],[],[]
    for d in range(4,16):
        if d in (8,12):
            w = fit(rows)
            group = []
        raw = history.trial.clock.book.rawday(d)
        y = history.trial.clock.labels(d).reshape(-1,50,3)
        ps = training_references(d,raw)
        obs,pair = [],[]
        for p in ps:
            x,far,length = design(p)
            t = np.concatenate([np.arange(lo+d%5,hi-60,5) for lo,hi in base.segments(len(p))])
            obs.append(observe(x[t],y[t]))
            if d>=8:
                z = apply(x,far,length,w)
                if d==8:
                    for n in (101,2487):
                        np.testing.assert_allclose(z[:n],apply(*design(p[:n]),w),atol=1e-6,rtol=0)
                    assert np.abs(z.mean(1)).max()<1e-6
                    assert np.abs(((2*p[:,:,2,None]+z)*z).mean(1)).max()<2e-6
                    assert np.abs((z*p[:,:,1,None]).mean(1)).max()<1e-6
                pair.append(base.records(p,y,z))
        rows.append(tuple(np.mean(v,0) for v in zip(*obs)))
        if d>=8:
            group.append(pair)
            daily.append([base.score(m,np.ones(len(ARMS)-1))[0] for m in pair])
            if d in (11,15):
                folds.append([base.score(m,np.ones(len(ARMS)-1)) for m in np.sum(group,0)])
    f,ds = np.array(folds),np.array(daily)
    ok = [i for i in range(3,len(ARMS)) if f[:,:,:,i].min()>0 and (ds[:,:,i]>0).sum(0).min()>=5
          and (f[:,:,0,i].mean(0)>f[:,:,0,1:3].mean(0).max(1)).all()]
    pick = max(ok,key=lambda i:f[:,:,0,i].min()) if ok else 0
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,arms=ARMS,folds=f.tolist(),
                                                  daily=ds.tolist(),windows=[[9,12],[13,16]]),indent=2))
    print('selected',ARMS[pick],f[:,:,0,:].round(6).tolist(),flush=True)
    if pick:
        np.savez(OUT/'model.npz',w=fit(rows)[:,pick-1])
    return bool(pick)


def correction():
    with np.load(OUT/'model.npz') as f:
        w = f['w'][:,None]
    return lambda p:apply(*design(p),w)[:,:,0]


def layer():
    before,update = parent.layer(),correction()
    def predict(raw,initial):
        p = before(raw,initial).copy()
        p[:,:,2] += update(p)
        return p
    return predict


def replay_with():
    update = correction()
    def read(d,raw):
        p,h,n = history.read(d,raw)
        p[:,:,2] += update(p)
        h[:,:,2] += update(h)
        return p,h,n
    return history.stack.verifier,read


if __name__=='__main__' and (sys.argv[1:]==['replay'] or train()):
    base.replay.replay(sys.modules[__name__])
