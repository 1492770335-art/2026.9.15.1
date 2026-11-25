"""Learn the signed return value of duration-sensitive occupancy using completed training holdout dates."""
import json
import numpy as np
import torch
import experiment_y3_pressure_duration_curve_v1 as duration

curve=duration.curve
incumbent=duration.incumbent
clock=duration.clock
OUT=clock.ART/'y3_duration_payoff_calibration_v1'


def shuffled(m,h):
    # Swap the duration profiles between states with nearby mean life. Preserve each
    # destination state's unconditional baseline exit probability.
    marginal=1-m[-2].diagonal()
    order=np.argsort(marginal);permuted=order.copy()
    permuted[:4]=order[:4].reshape(2,2)[:,::-1].ravel()
    permuted[[3,4]]=permuted[[4,3]]
    ratio=h/marginal[:,None]
    out=np.empty_like(h)
    out[order]=marginal[order,None]*ratio[permuted]
    return out.clip(.01,.95)


def fit_models(end):
    m=duration.markov.fit_states(end);h=duration.duration(end,m)
    hs=(h,shuffled(m,h));history=[[],[],[]]
    for d in range(end):
        raw=clock.book.rawday(d);t=np.arange(d%7,len(raw)-61,7)
        y=clock.labels(d).reshape(-1,50,3)
        band,_=curve.target(y,t)
        truth=band.sum(2,keepdims=True).clip(-4,4).astype('float32')
        ctrl,correct=duration.features(raw,t,m,hs[0])
        wrong=duration.features(raw,t,m,hs[1])[1]
        for hist,s in zip(history,(ctrl,correct,wrong)):hist.append(curve.gram(s,truth))
    w=[curve.solve(*[np.mean(v,0) for v in zip(*hist)],.45).ravel() for hist in history]
    return m,hs,w


def raw_change(raw,t,m,hs,w):
    ctrl,correct=duration.features(raw,t,m,hs[0])
    wrong=duration.features(raw,t,m,hs[1])[1]
    old=ctrl@w[0]
    return np.stack((correct@w[1]-old,wrong@w[2]-old),2)


def stats(p,y,z):
    a=clock.center(p[:,:,2]);b=clock.center(y[:,:,2])
    beta=((a*b).mean()/(a*a).mean().clip(.00001)).clip(.45,2)
    error=(b/beta-a).clip(-4,4)
    return np.stack(((z*error[:,:,None]).mean((0,1)),(z*z).mean((0,1))))


def weigh(history):
    records=np.stack(history);cross,risk=records.mean(0)
    uncertainty=records[:,0].std(0)/np.sqrt(len(records))
    weight=(cross/(risk+.0002)).clip(-.45,.45)
    return weight*abs(cross)/(abs(cross)+uncertainty+.000015)


def run():
    hist=[];folds=[];days=[];wlog=[]
    for end in (8,10,12,14):
        m,hs,w=fit_models(end);weight=weigh(hist) if hist else None
        groups=[[[],[],[]],[[],[],[]]]
        for d in (end,end+1):
            raw=clock.book.rawday(d);t=np.arange(d%4,len(raw),4)
            z=raw_change(raw,t,m,hs,w)
            if d==end:
                sub=t[t<431];np.testing.assert_allclose(raw_change(raw[:431],sub,m,hs,w),z[:len(sub)],rtol=0,atol=2e-6)
            truth=clock.labels(d).reshape(-1,50,3)[t]
            for j,ref in enumerate(curve.REFS):
                p=ref(d).reshape(-1,50,3)[t];valid=t<len(raw)-61
                if not j:hist.append(stats(p[valid],truth[valid],z[valid]))
                if weight is None:continue
                correct,wrong=np.moveaxis(z*weight,2,0)
                for k,v in enumerate((correct,wrong,correct-wrong)):groups[j][k].append((p,truth,v))
            if weight is not None:days.append([curve.metric.gain(g[-1:]) for g in groups[0]])
        if weight is not None:
            wlog.append(weight.tolist());folds.append([[curve.metric.gain(g) for g in side] for side in groups])
            print(end+1,weight,folds[-1][0],flush=True)
    def passes(k):
        return all(sum(f[j][k]['y3']>0 for f in folds)>=2 and np.mean([f[j][k]['y3'] for f in folds])>0
                   and folds[-1][j][k]['y3']>0 for j in range(2)) and sum(d[k]['y3']>0 for d in days)>=4
    good=[k for k in (0,2) if passes(k)]
    pick=max(good,key=lambda a:np.mean([f[0][a]['y3'] for f in folds])) if good else None
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,folds=folds,daily=days,coefficients=wlog,
            arms=['matched duration','permuted duration','difference']),indent=2))
    if pick is not None:
        m,hs,w=fit_models(16)
        np.savez(OUT/'model.npz',**{f'm{i}':v for i,v in enumerate(m)},hs=hs,w=w,coef=weigh(hist),pick=pick)
    return pick is not None


def layer():
    parent=incumbent.layer()
    with np.load(OUT/'model.npz') as s:
        m=[s[f'm{i}'] for i in range(6)];hs,w,coef,pick=s['hs'],s['w'],s['coef'],int(s['pick'])
    def apply(raw,p):
        q=parent(raw,p);t=np.arange(len(q))
        correct,wrong=np.moveaxis(raw_change(raw,t,m,hs,w)*coef,2,0)
        q[:,:,2]+=correct if pick==0 else correct-wrong
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as ancestor
    parent,act=ancestor.load(),layer()
    return lambda raw,p:act(raw,parent(raw,p))


if __name__=='__main__':
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    if run():incumbent.incumbent.base.chain.replay.verify(OUT,layer,incumbent.OUT)
