"""Recalibrate each stock's future rank probabilities using only settled earlier forecast errors."""
import json
import sys

import numpy as np
from scipy.stats import rankdata
import torch

import experiment_y3_settled_factor_confidence_v1 as proxy
import y3_rank_capacity_references as history

parent,base,rank = history.incumbent,history.base,history.rank
OUT = parent.OUT.parent/'y3_settled_rank_probabilities_v1'
ARMS = ('unchanged','displaced stock calibration','blind historical frequencies','recent frequency ratio',
        'intraday frequency ratio','conditional log-odds','same-sign agreement at two speeds',
        'recent relative probability update','intraday relative probability update','past excess tail odds')


@torch.inference_mode()
def panel(raw,nets):
    x,ph = rank.inputs.features(raw,np.arange(len(raw)))
    probabilities,payouts = [],[]
    for i in range(0,len(raw),96):
        v,s = torch.from_numpy(x[i:i+96]).cuda(),torch.from_numpy(ph[i:i+96]).cuda()
        _,a,p = nets[0](v,s)
        q = nets[1](v,s)[2]
        probabilities.append(a[:,:,2].cpu().numpy())
        payouts.append((.5*(p[:,2]+q[:,2])).cpu().numpy())
    return np.concatenate(probabilities),np.concatenate(payouts)


def balanced(p):
    x = p.copy()
    for _ in range(4):
        x *= 5/x.sum(1,keepdims=True).clip(1e-9)
        x /= x.sum(2,keepdims=True).clip(1e-9)
    return x


def outcomes(raw):
    k,coef,missing = proxy.historical_geometry()
    path = proxy.compound.compound(raw,k)
    bad = np.vstack((np.zeros((1,50)),(~proxy.compound.present(raw,missing)).cumsum(0)))
    t = np.arange(max(0,len(raw)-60))
    y = (coef[2,0]+coef[2,1]*proxy.compound.span(path,t,60,k) -
         base.RATIO*(coef[1,0]+coef[1,1]*proxy.compound.span(path,t,20,k)))
    valid = (bad[t+61]==bad[t])
    y = np.where(valid,y,np.nan)
    # Percentile ranks among the visible names handle missing stocks and split ties across bin boundaries.
    n = valid.sum(1,keepdims=True).clip(1)
    left = (rankdata(y,axis=1,method='min',nan_policy='omit')-1)*10/n
    right = rankdata(y,axis=1,method='max',nan_policy='omit')*10/n
    edges = np.arange(10)
    mass = (np.minimum(right[:,:,None],edges+1)-np.maximum(left[:,:,None],edges)).clip(0)
    probs = np.nan_to_num(mass/(right-left)[:,:,None].clip(1e-10))
    return probs,valid&(n>=35)


def signals(raw,nets):
    prior,pay = panel(raw,nets)
    baseline = balanced(prior)
    true,valid = outcomes(raw)
    output = []
    for lo,hi in base.segments(len(raw)):
        marks = np.arange(lo,max(lo,hi-60),4)
        ok = valid[marks,:,None]
        obs = true[marks]*ok
        forecast = baseline[marks]*ok
        def prefix(v):
            return np.concatenate((np.zeros_like(v[:1],shape=(1,50,10)),v.cumsum(0,dtype=float)))
        c1,c2 = prefix(obs),prefix(forecast)
        for start in range(lo,hi,40):
            end = min(start+40,hi)
            released = np.searchsorted(marks,start-60,side='right')
            def evidence(width):
                left = max(0,released-width//4)
                truth = c1[released]-c1[left]
                expected = c2[released]-c2[left]
                # Forty overlapping ticks form one horizon; use a substantial prior per name and bin.
                return np.log((truth+12)/(expected+12)).clip(-1.2,1.2)
            a,b = evidence(360),evidence(1440)
            left = max(0,released-90)
            count = c1[released]-c1[left]
            blind = np.log((count+12)/(count.mean(1,keepdims=True)+12))
            p = prior[start:end]; price = pay[start:end,None]
            def adjustment(logratio):
                shift = balanced(p*np.exp(.5*logratio)[None])-baseline[start:end]
                return (shift*price).sum(-1)
            short,long = adjustment(a),adjustment(b)
            consensus = np.where(a*b>0,np.sign(b)*np.minimum(abs(a),abs(b)),0.)
            tail = a.copy()
            tail[:,2:-2] = 0
            extra = a-b
            cols = (adjustment(np.roll(a,13,0)),adjustment(blind),short,long,
                    adjustment(extra),adjustment(consensus),short-short.mean(1,keepdims=True),
                    long-long.mean(1,keepdims=True),adjustment(tail))
            output.append(np.stack(cols,2).astype('float32'))
    return np.concatenate(output)


def train():
    OUT.mkdir(exist_ok=True)
    rows,cal,folds,daily = [],[],[],[]
    for day in range(16):
        if day in (4,8,12):
            nets = rank.fit(rows)[1:]
            if day>=8:
                w = base.calibrate(cal)
                fold = []
        if day>=4:
            raw = history.trial.clock.book.rawday(day)
            y = history.trial.clock.labels(day).reshape(-1,50,3)
            z = signals(raw,nets)
            if day==4:
                for n in (51,181,2449):
                    np.testing.assert_allclose(z[:n],signals(raw[:n],nets),atol=2e-6,rtol=0)
            ps = history.references(day,raw) if day>=8 else history.previous.previous.incumbent.parent.warm.read(day,raw)
            pair = [base.records(p,y,z) for p in ps]
            cal.append(pair)
            if day>=8:
                fold.append(pair)
                daily.append([base.score(m,w) for m in pair])
                if day in (11,15):
                    folds.append([base.score(m,w) for m in np.sum(fold,0)])
        rows.append(rank.samples(day))
    f,ds = np.asarray(folds)[:,:,0],np.asarray(daily)[:,:,0]
    ok = [j for j in range(3,len(ARMS)) if f[:,:,j].min()>0 and (ds[:,:,j]>0).sum(0).min()>=5 and
          (f[:,:,j].mean(0)>f[:,:,1:3].mean(0).max(1)).all()]
    pick = max(ok,key=lambda j:f[:,:,j].min()) if ok else 0
    daily_screen = pick
    trend = ARMS.index('conditional log-odds')
    if pick==0 and f[:,:,trend].min()>0 and (f[:,:,trend].mean(0)>f[:,:,1:3].mean(0).max(1)).all():
        pick = trend
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,daily_screen=daily_screen,arms=ARMS,folds=np.asarray(folds).tolist(),
                                    daily=np.asarray(daily).tolist()),indent=2))
    print('selected',ARMS[pick],flush=True)
    if pick:
        np.savez(OUT/'model.npz',pick=pick-1,w=base.calibrate(cal)[pick-1])
    return bool(pick)


def correction():
    nets,_ = parent.load_models()
    with np.load(OUT/'model.npz') as f:
        w,pick = float(f['w']),int(f['pick'])
    return lambda raw:(signals(raw,nets)[:,:,pick]*w).astype('float32')


def layer():
    before,update = parent.layer(),correction()
    def predict(raw,initial):
        p = before(raw,initial).copy()
        p[:,:,2] += update(raw)
        return p
    return predict


def replay_with():
    update = correction()
    def read(day,raw):
        p,h,n = history.read(day,raw)
        p[:,:,2] += update(raw)
        h[:,:,2] += update(raw[:n])
        return p,h,n
    return history.stack.verifier,read


if __name__=='__main__' and (sys.argv[1:]==['replay'] or train()):
    base.replay.replay(sys.modules[__name__])
