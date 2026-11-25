"""Predict the payoff after a stock overtakes a peer in the cross-sectional forecasts."""
import json

import numpy as np
import experiment_opening_phase_slopes_v1 as parent
from replay_periodic_parent_v1 import verify

clock = parent.clock
OUT = clock.ART / 'peer_crossing_payoff_v1'
LAGS = (8, 32, 96)


def design(raw, prediction, ticks, permutation=None):
    price = np.cumsum(raw[:,:,0].astype(float) - .00136, axis=0)
    far = prediction[:,:,2] - clock.RATIO * prediction[:,:,1]
    fields = np.stack((prediction[:,:,1], far), axis=2)
    result = []
    for block in np.array_split(ticks, max(1, (len(ticks)+95)//96)):
        current = fields[block]
        scales = np.sqrt(.003 + current.var(1,keepdims=True))
        nuisance = [np.tanh(current/scales)]
        crossed = []
        for lag in LAGS:
            old = np.maximum(0, block-lag)
            prev = fields[old]
            returns = price[block]-price[old]
            returns = clock.center(returns)/np.sqrt(lag)
            pastscale = np.sqrt(.003+prev.var(1,keepdims=True))
            movement = np.tanh(current/scales-prev/pastscale)
            payoff = np.tanh(returns)
            nuisance.extend((movement, movement*payoff[:,:,None],payoff[:,:,None]))
            for k in range(fields.shape[2]):
                newgap = (current[:,:,None,k]-current[:,None,:,k])/scales[:,:,k,None]
                oldgap = (prev[:,:,None,k]-prev[:,None,:,k])/pastscale[:,:,k,None]
                priced = returns if permutation is None else returns[:,permutation]
                spread = priced[:,:,None]-priced[:,None,:]
                # Each ordered pair contributes only after it has crossed in forecast rank.
                flip = np.sign(newgap)*(newgap*oldgap < 0)
                confidence = np.tanh(abs(newgap))*np.tanh(abs(oldgap))
                paid = np.tanh(spread)*np.sign(newgap)
                crossed.extend(((flip*confidence).mean(2),
                                (flip*confidence*paid).mean(2),
                                (flip*confidence*(paid**2)).mean(2)))
        result.append((clock.center(np.concatenate(nuisance,axis=2)),
                       clock.center(np.stack(crossed,axis=2))))
    return tuple(np.concatenate(cols).astype('float32') for cols in zip(*result))


def record(x, z, p, y):
    # Calibrate on errors of a historical out-of-fold predictor.
    target = clock.center((y-p).astype(float)).reshape(-1,3)
    a = np.concatenate((x,z),axis=2).reshape(len(target),-1).astype(float)
    return a.T@a/len(a), a.T@target/len(a)


def weights(history,nuisance):
    grams,rhs = (np.stack(col) for col in zip(*history))
    gram = grams.mean(0)
    penalty = .35*np.diag(gram.diagonal().clip(.0001))+.00003*np.eye(len(gram))
    def contrast(g,b):
        w = np.linalg.solve(g+penalty,b)
        w[:nuisance] -= np.linalg.solve((g+penalty)[:nuisance,:nuisance],b[:nuisance])
        return w
    pooled = contrast(gram,rhs.mean(0))
    per_day = np.stack([contrast(g,b) for g,b in zip(grams,rhs)])
    return .55*(.75*pooled+.25*np.median(per_day,axis=0))


def apply(x,z,w):
    return clock.center(np.concatenate((x,z),axis=2)@w)


def screen():
    perm = np.random.default_rng(8141).permutation(50)
    history = [[],[]]
    folds,days,placebos,far_ic = [],[],[],[]
    for start in (8,10,12,14):
        counts = len(history[0])
        pars = [weights(h,nuisance) for h in history] if counts else None
        groups = [[],[]]
        controls,far_rows = [],[]
        for d in (start,start+1):
            raw = clock.book.rawday(d)
            t = np.arange(d%4,len(raw),4)
            train = t<len(raw)-60
            y = clock.labels(d).reshape(-1,50,3)[t]
            for j,ref in enumerate((clock.incumbent.fi.peer.reference,clock.incumbent.parent.fixed.earlier.early.baseline)):
                forecasts = ref(d).reshape(-1,50,3)
                p = forecasts[t]
                x,z = design(raw,forecasts,t)
                nuisance = x.shape[2]
                if not j:
                    _,shuffled = design(raw,forecasts,t,perm)
                    for h,features in zip(history,(z,shuffled)):
                        h.append(record(x[train],features[train],p[train],y[train]))
                    if d == start:
                        subset = t < 277
                        for arr,check in zip((x,z),design(raw[:277],forecasts[:277],t[subset])):
                            np.testing.assert_allclose(arr[subset],check,rtol=0,atol=2e-7)
                if pars is None:continue
                adjustment = apply(x,z,pars[0])
                groups[j].append((p,y,adjustment))
                if not j:
                    controls.append((p,y,apply(x,shuffled,pars[1])))
                    actual = clock.center(y[:,:,2]-clock.RATIO*y[:,:,1])
                    distant = clock.center(p[:,:,2]-clock.RATIO*p[:,:,1])
                    far_rows.append((distant.ravel(),(distant+adjustment[:,:,2]).ravel(),actual.ravel()))
            if pars is not None:days.append(clock.deltas(groups[0][-1:]))
        if pars is not None:
            folds.append([clock.deltas(g) for g in groups])
            placebos.append(clock.deltas(controls))
            a,b,y = (np.concatenate(col) for col in zip(*far_rows))
            far_ic.append(float(np.corrcoef(b,y)[0,1]-np.corrcoef(a,y)[0,1]))
            print('rolling dates',start+1,start+2,folds[-1][0],flush=True)
    f = np.asarray(folds)
    keep = (f.min((0,1))>0)&((np.array(days)>0).sum(0)>=4)
    keep &= (f[:,0]-np.asarray(placebos)).mean(0)>0
    keep[2] &= np.mean(far_ic)>0
    report = dict(folds=folds,daily=days,shuffled=placebos,far_ic=far_ic,keep=keep.tolist())
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    if keep.any():np.save(OUT/'weights.npy',weights(history[0],nuisance)*keep)
    return keep.any()


def layer():
    base = parent.layer()
    w = np.load(OUT/'weights.npy')
    def predict(raw,p):
        q = base(raw,p)
        x,z = design(raw,q,np.arange(len(q)))
        return (q+apply(x,z,w)).astype(q.dtype)
    return predict


def load():
    import experiment_periodic_near_carry_v1 as upstream
    baseline = upstream.load()
    correction = layer()
    return lambda raw,p: correction(raw,baseline(raw,p))


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if screen():verify(OUT,layer,parent.OUT)
