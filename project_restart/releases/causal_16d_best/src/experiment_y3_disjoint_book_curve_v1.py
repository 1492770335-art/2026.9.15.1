"""Predict the central 20 ticks of Y3 from other names, with disjoint sender and receiver panels."""
import json
import numpy as np
import torch

import experiment_y3_duration_payoff_calibration_v1 as incumbent
import experiment_y3_partial_forward_curve_v1 as curve
import experiment_y3_sector_anchored_factors_v1 as sector
import replay_periodic_parent_v1 as replay

clock = curve.clock
OUT = clock.ART / 'y3_disjoint_book_curve_v1'
NMODES = 12


def encoder(end=8):
    samples = [np.arcsinh(clock.book.rawday(d)[d::26].astype('float32')).reshape(-1,100) for d in range(end)]
    observations = np.concatenate(samples)
    mean = observations.mean(0, dtype=float)
    scale = observations.std(0, dtype=float).clip(.07)
    x = ((observations-mean)/scale).clip(-4,4)
    powers, vectors = np.linalg.eigh(x.T@x/len(x))
    matrix = vectors[:,-NMODES:] / np.sqrt(powers[-NMODES:]).clip(.25)
    return mean.astype('float32'), scale.astype('float32'), matrix.astype('float32')


def features(raw, encoding):
    mean, scale, matrix = encoding
    # Each stock is projected before stocks are split. No cross-sectional preprocessing.
    states = np.empty((*raw.shape[:2], NMODES), dtype='float32')
    for t in np.array_split(np.arange(len(raw)), max(1,(len(raw)+399)//400)):
        x = np.arcsinh(raw[t].astype('float32'))
        states[t] = ((x-mean)/scale).clip(-4,4)@matrix
    end = np.arange(1,len(raw)+1)
    csum = np.concatenate((np.zeros_like(states[:1],dtype=float),states.cumsum(0,dtype=float)))
    def average(width):
        left = (end-width).clip(0)
        return ((csum[end]-csum[left])/(end-left)[:,None,None]).astype('float32')
    fast, slow = average(7), average(48)
    return np.concatenate((states-fast, fast-slow, slow),axis=2)


def observe(x, y, day):
    t = np.arange(day%4, len(x)-60, 4)
    path,_ = curve.target(y,t)
    intervals = np.stack((path[:,:,1:3].sum(2), path[:,:,::3].sum(2)), axis=2)
    records = []
    for parity in range(2):
        # Subtract receiver-panel mean so a donor's own label cannot enter its targets.
        targets = intervals[:,parity::2]
        targets = (targets-targets.mean(1,keepdims=True)).clip(-3.5,3.5)
        sender = x[t,1-parity::2].reshape(len(t),-1)
        mean = sender.mean(0)
        f = sender-mean
        r = targets.reshape(len(t),-1).astype('float32')
        records.append((mean, f.T@f/len(t), f.T@r/len(t)))
    return records


def fit(history):
    means, weights, ranks = [], [], []
    for side in zip(*history):
        mu, gram, rhs = [np.stack(v) for v in zip(*side)]
        avg = gram.mean(0,dtype=float)
        penalty = np.diag(2.*avg.diagonal().clip(.012))+.002*np.eye(len(avg))
        # Discover portfolios on alternate dates and cross-check directions on the other dates.
        half = [rhs[k::2].mean(0,dtype=float) for k in range(2)]
        std = avg.diagonal().clip(.012)**-.5
        z = [std[:,None]*v for v in half]
        dependence = (z[0].T@z[1]+z[1].T@z[0])*.5
        noise = (z[0]-z[1]).T@(z[0]-z[1])*.25
        strength, directions = np.linalg.eigh(dependence)
        uncertainty = np.einsum('ij,ij->j', directions, noise@directions)
        keep = strength.clip(0)/(strength.clip(0)+uncertainty+.00001)
        smoother = (directions*keep)@directions.T
        weights.append(np.linalg.solve(avg+penalty, rhs.mean(0,dtype=float)@smoother).astype('float32'))
        means.append(mu.mean(0))
        ranks.append(float(keep.sum()))
    return np.array(means), np.array(weights), ranks


def forecasts(x, model):
    means, weights = model[:2]
    out = np.empty((*x.shape[:2],2),dtype='float32')
    for parity in range(2):
        sender = x[:,1-parity::2].reshape(len(x),-1)
        q = ((sender-means[parity])@weights[parity]).reshape(len(x),25,2)
        out[:,parity::2] = q-q.mean(1,keepdims=True)
    return out


def matched_control(groups):
    order = np.arange(50)
    for parity in range(2):
        for group in np.unique(groups):
            ids = np.flatnonzero((np.arange(50)%2==parity)&(groups==group))
            order[ids] = np.roll(ids,1)
    return order


def corrections(f, order):
    mid = f[:,:,0]
    other = mid[:,order]
    return [.15*mid, .15*(mid-other), .15*(mid-f[:,:,1])]


def correlation(x,y):
    return float(np.corrcoef(x.ravel(),y.ravel())[0,1])


def run():
    encoding = encoder()
    history, folds, daily, diagnosis = [], [], [], {}
    for day in range(16):
        raw = clock.book.rawday(day)
        x = features(raw,encoding)
        y = clock.labels(day).reshape(-1,50,3)
        if day >= 8:
            if day%2 == 0:
                model = fit(history)
                ids = sector.grouping(day)
                order = matched_control(ids)
                groups = [[[],[],[]],[[],[],[]]]
            f = forecasts(x,model)
            z = corrections(f,order)
            if day%2 == 0:
                tiny = forecasts(features(raw[:389],encoding),model)
                np.testing.assert_allclose(tiny, f[:389], atol=3e-6,rtol=0)
                # Verify that changing every receiver's book cannot change its panel's output.
                modified = raw[:389].copy()
                modified[:,::2] = 0
                independent = forecasts(features(modified,encoding),model)
                np.testing.assert_array_equal(tiny[:,::2],independent[:,::2])
            t = np.arange(len(raw)-60)
            path,_ = curve.target(y,t)
            middle = path[:,:,1:3].sum(2)
            for j, ref in enumerate(curve.REFS):
                p = ref(day).reshape(y.shape)
                if j == 0:
                    confounders = clock.center(p[t]).reshape(-1,3).astype(float)
                    targets = np.column_stack([v.ravel() for v in (middle,f[t,:,0],f[t][:,order,0])])
                    resid = targets-confounders@np.linalg.lstsq(confounders,targets,rcond=None)[0]
                    diagnosis[day+1] = dict(middle_ic=correlation(middle,f[t,:,0]),
                            partial_matched=correlation(resid[:,0],resid[:,1]),
                            partial_reassigned=correlation(resid[:,0],resid[:,2]), rank=model[2])
                for group, change in zip(groups[j], z):
                    group.append((p,y,change))
            daily.append([curve.metric.gain(g[-1:])['y3'] for g in groups[0]])
            if day%2:
                folds.append([[curve.metric.gain(g)['y3'] for g in side] for side in groups])
                print('fold ending',day+1,folds[-1][0],flush=True)
        history.append(observe(x,y,day))
    a = np.array(folds)
    ok = ((a>0).sum(0).min(0)>=3)&(a.mean(0).min(0)>0)&(a[-1].min(0)>0)&((np.array(daily)>0).sum(0)>=5)
    has_signal = np.mean([d['partial_matched']-d['partial_reassigned'] for d in diagnosis.values()])>0
    choices = np.flatnonzero(ok) if has_signal else []
    pick = int(max(choices,key=lambda i:a[:,0,i].mean())) if len(choices) else None
    report = dict(selected=pick,folds=folds,daily=daily,diagnostics=diagnosis,
                  arms=['middle','middle minus sector-matched reassignment','middle minus edges'])
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    if pick is not None:
        mu,w,_ = fit(history)
        np.savez(OUT/'model.npz',mean=encoding[0],scale=encoding[1],projection=encoding[2],mu=mu,w=w,
                 order=matched_control(sector.grouping(16)),pick=pick)
    return pick is not None


def layer():
    parent = incumbent.layer()
    with np.load(OUT/'model.npz') as s:
        enc = tuple(s[k] for k in ('mean','scale','projection'))
        model,order,pick = (s['mu'],s['w']),s['order'],int(s['pick'])
    def apply(raw,p):
        q = parent(raw,p)
        q[:,:,2] += corrections(forecasts(features(raw,enc),model),order)[pick]
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as ancestor
    parent,update = ancestor.load(),layer()
    return lambda raw,p:update(raw,parent(raw,p))


if __name__ == '__main__':
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    if run():
        replay.verify(OUT,layer,incumbent.OUT)
