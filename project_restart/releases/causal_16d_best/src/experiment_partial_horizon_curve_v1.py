"""Walk-forward calibration of the distinct information carried by shorter horizon forecasts."""
import json
import numpy as np
from experiment_causal_cross_horizon_ridge_v1 import ART, labels, corr, measure
import experiment_eight_date_expert_contrasts_v1 as early
import experiment_execution_volume_clock_v1 as clock
import replay_verified_forecast_parent_v1 as replay

OUT = ART / 'partial_horizon_curve_v1'
PARENT = ART / 'legacy_horizon_completion_v1'


def mean_past(a, width):
    s = np.concatenate((np.zeros_like(a[:1], dtype=float), a.cumsum(axis=0, dtype=float)))
    end = np.arange(1, len(a) + 1)
    start = (end - width).clip(0)
    return (s[end] - s[start]) / (end - start).reshape((-1,) + (1,) * (a.ndim - 1))


def features(raw, pred):
    m = pred.mean(1, keepdims=True, dtype=float)
    r = pred - m
    price = raw[:, :, 0].astype(float) - .00136
    risk = np.log((.25 + mean_past(price**2, 80))**.5).clip(-1.5, 1.5)
    age = np.exp(-np.arange(len(pred)) / 400)[:, None]
    result = []
    for ix, p in enumerate((m, r)):
        if ix == 0:
            cross = p.transpose(0, 2, 1) @ p
            gram = mean_past(cross, 160) + np.eye(3)[None] * .004
        else:
            gram = p.transpose(0, 2, 1) @ p / 50
            gram += .35 * mean_past(gram, 48) + np.eye(3)[None] * .001
        short = []
        for j in (0, 1):
            longer = slice(j+1, 3)
            sub = gram[:, longer, longer]
            reg = np.eye(2-j)[None] * (.06*sub.diagonal(axis1=1,axis2=2) + .0008)[:,None,:]
            coef = np.linalg.solve(sub + reg, gram[:, longer, j, None])
            q = p[:,:,j] - (p[:,:,longer] @ coef)[:,:,0]
            amp = np.sqrt(gram[:,j,j]).clip(.05, None)[:,None]
            short.extend((q, q * np.tanh(abs(q)/amp-1), q*age, q*risk.mean(1, keepdims=True) if ix==0 else q*risk))
            for width in (5, 20, 60):
                avg = mean_past(q,width)
                short.extend((avg, (q-avg)*np.tanh(q*avg/amp**2)))
        f=np.stack(short,axis=2)
        result.append((f[:,0] if ix==0 else f-f.mean(1,keepdims=True)).astype('float32'))
    return result


def add(p, z, w):
    terms=((z[0]@w[0])[:,None],z[1]@w[1])
    return (p+sum(terms)).astype('float32'),terms


def train():
    OUT.mkdir(exist_ok=True)
    hist=[]; daily=[]; rows=[]; report={}
    for d in range(8,16):
        if d in (10,12,14):
            w=clock.flow.fit(hist);rows=[]
        p=np.load(ART/'opening_oof_cache_v1'/f'day_{d:02d}.npy')
        z=features(clock.pair.rawday(d),p);y=labels(d).reshape(p.shape)
        if d>=10:
            q,terms=add(p,z,w)
            row=tuple(v.reshape(-1,3) for v in (p,q,y));rows.append(row)
            old=corr(row[0],row[2])
            daily.append([corr((p+c).reshape(-1,3),row[2])-old for c in terms])
        hist.append(early.moments(p,y,z))
        if d in (11,13,15):
            report[str(d+1)]=measure(rows);print(d+1,report[str(d+1)],flush=True)
    gains=np.stack(daily)
    mask=((gains>0).sum(0)>=4)&(gains[-2:].mean(0)>0)&(gains.mean(0)>0)
    mask[:,0]=False
    report['eligible']=mask.tolist();report['early_daily_delta']=gains.tolist()
    w=[v*keep for v,keep in zip(clock.flow.fit(hist),mask)]
    np.savez(OUT/'model.npz',market=w[0],relative=w[1])
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    if not mask.any():return
    rows=[]
    for d in range(16,20):
        p=np.load(PARENT/f'pred_{d}.npy');raw=clock.pair.rawday(d)
        q,_=add(p,features(raw,p),w);np.save(OUT/f'pred_{d}.npy',q)
        rows.append(tuple(v.reshape(-1,3) for v in (p,q,labels(d))))
    result=measure(rows);report['20']=result;print(result,flush=True)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))


def verify():
    with np.load(OUT/'model.npz') as st:w=[st['market'],st['relative']]
    replay.main(PARENT,OUT,lambda raw,p:add(p,features(raw,p),w)[0])


if __name__=='__main__':
    import sys
    if '--replay' in sys.argv:verify()
    else:train()
