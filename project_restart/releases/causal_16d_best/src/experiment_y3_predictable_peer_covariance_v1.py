"""Estimate the part of cross-stock return covariance transmitted by contemporaneous book forecasts."""
import json

import numpy as np
import torch

import experiment_y3_disjoint_book_curve_v1 as source

curve, clock, incumbent = source.curve, source.clock, source.incumbent
OUT = clock.ART / 'y3_predictable_peer_covariance_v1'


def design(raw, enc):
    x = source.features(raw, enc)
    # This shared readout operates on each stock individually and cannot see a peer.
    return np.concatenate((x, np.tanh(2*x), x[:,:,:12]*np.tanh(x[:,:,24:])),2)


def actual(y,t):
    blocks,_ = curve.target(y,t)
    return np.stack((blocks[:,:,1:3].sum(2),blocks.sum(2)),axis=2).astype('float32')


def readout(x,y,d):
    t = np.arange(d%5,len(x)-60,5)
    return curve.gram(x[t], actual(y,t).clip(-3.5,3.5))


def shared(history):
    return curve.solve(*[np.mean(v,axis=0) for v in zip(*history)],.48).astype('float32')


def stats(x, y):
    rows = []
    for parity in (0,1):
        targets = y[:,parity::2]
        r = targets-targets.mean(1,keepdims=True)
        # Both sender versions refer to the same 31-50 or 21-60 return interval.
        pair = (x[:,1-parity::2], y[:,1-parity::2])
        record = []
        for sender in pair:
            a = sender-sender.mean(1,keepdims=True)
            record.append([curve.gram(a[:,:,k],r[:,:,k]) for k in range(2)])
        rows.append(record)
    return rows


def maps(history):
    # Same shrinkage in correlation units for predictable- and total-return covariance.
    w = np.empty((2,2,2,25,25))
    for side in (0,1):
        for part in (0,1):
            for k in range(2):
                g,r = [np.mean(v,0) for v in zip(*[h[side][part][k] for h in history])]
                w[side,part,k] = curve.solve(g,r,1.7)
    return w.astype('float32')


def moves(x, w):
    pred = np.empty((len(x),50,2,2),dtype='float32')
    for side in (0,1):
        f = x[:,1-side::2]
        f = f-f.mean(1,keepdims=True)
        for estimator in (0,1):
            for k in (0,1):
                z = f[:,:,k]@w[side,estimator,k]
                pred[:,side::2,estimator,k] = z-z.mean(1,keepdims=True)
    return pred


def corrections(z):
    informed, total = z[:,:,0], z[:,:,1]
    # Exclude realized-return covariances as deployable forecasts on their own.
    return [.12*informed[:,:,0], .12*(informed[:,:,0]-total[:,:,0]),
            .12*(informed[:,:,1]-total[:,:,1])]


def run():
    enc = source.encoder(4)
    training, history, folds, daily, details = [], [], [], [], {}
    for day in range(16):
        raw = clock.book.rawday(day)
        x = design(raw,enc)
        y = clock.labels(day).reshape(-1,50,3)
        if day >= 4:
            if day%2 == 0:
                w = shared(training)
                if day >= 8:
                    projection = maps(history)
                    groups = [[[],[],[]],[[],[],[]]]
            local = x@w
            t = np.arange(len(y)-60)
            outcomes = actual(y,t)
            if day >= 8:
                z = moves(local,projection)
                if day%2 == 0:
                    short = moves(design(raw[:383],enc)@w,projection)
                    np.testing.assert_allclose(short, z[:383],rtol=0,atol=4e-6)
                for j,ref in enumerate(curve.REFS):
                    p = ref(day).reshape(y.shape)
                    if not j:
                        arrays = [outcomes[:,:,0],local[t,:,0],z[t,:,0,0],z[t,:,1,0]]
                        obs = np.column_stack([clock.center(a).ravel() for a in arrays]).astype(float)
                        conf = clock.center(p[t]).reshape(-1,3).astype(float)
                        obs -= conf@np.linalg.lstsq(conf,obs,rcond=None)[0]
                        details[day+1] = np.corrcoef(obs.T)[0,1:].tolist()
                    for g, change in zip(groups[j],corrections(z)):
                        g.append((p,y,change))
                daily.append([curve.metric.gain(g[-1:])['y3'] for g in groups[0]])
                if day%2:
                    folds.append([[curve.metric.gain(g)['y3'] for g in s] for s in groups])
                    print('ending day',day+1,folds[-1][0],flush=True)
            # All dates that supply supervised covariance use predictions from earlier labels.
            history.append(stats(local[t],outcomes))
        training.append(readout(x,y,day))
    a = np.array(folds)
    keep = ((a>0).sum(0).min(0)>=3)&(a.mean(0).min(0)>0)&(a[-1].min(0)>0)&((np.array(daily)>0).sum(0)>=5)
    choices = np.flatnonzero(keep)
    pick = int(max(choices,key=lambda i:a[:,0,i].mean())) if len(choices) else None
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,folds=folds,daily=daily,
        partial_middle_ic=details, partial_columns=['local','forecast-based peers','return-based peers'],
        arms=['forecast-based middle','excess middle','excess far']),indent=2))
    if pick is not None:
        np.savez(OUT/'model.npz',mean=enc[0],scale=enc[1],projection=enc[2],read=shared(training),
                 maps=maps(history),pick=pick)
    return pick is not None


def layer():
    parent = incumbent.layer()
    with np.load(OUT/'model.npz') as s:
        enc = tuple(s[k] for k in ('mean','scale','projection'))
        w,m,pick = s['read'],s['maps'],int(s['pick'])
    def apply(raw,p):
        q = parent(raw,p)
        z = moves(design(raw,enc)@w,m)
        q[:,:,2] += corrections(z)[pick]
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as ancestor
    parent, update = ancestor.load(), layer()
    return lambda raw,p:update(raw,parent(raw,p))


if __name__ == '__main__':
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    if run():
        source.replay.verify(OUT,layer,incumbent.OUT)
