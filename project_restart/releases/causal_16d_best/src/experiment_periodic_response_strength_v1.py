"""Estimate the clock correction strength on preceding out-of-fold days, separately by target."""
import json
import numpy as np

import experiment_periodic_book_response_v1 as trial

OUT=trial.ART/'periodic_response_strength_v1'


def corrections(terms):
    m, r = terms
    z = m + r
    z[:,:,2] = trial.RATIO*z[:,:,1]
    return z


def coefficients(history):
    # Estimate the correlation-maximizing incremental slope after projecting out the parent forecast.
    xy,xx=(np.stack(col) for col in zip(*history))
    pooled=xy.mean(0)/xx.mean(0).clip(1.e-12)
    daily=np.median(xy/xx.clip(1.e-12),axis=0)
    return (1+.65*(.6*pooled+.4*daily-1)).clip(0,6)


def moments(p,y,z):
    a,b,x=[v.reshape(-1,3).astype(float) for v in (p,y,z)]
    a-=a.mean(0); b-=b.mean(0); x-=x.mean(0)
    scale=(a*b).mean(0)/(a*a).mean(0).clip(1.e-6)
    x-=a*((a*x).mean(0)/(a*a).mean(0).clip(1.e-6))
    return (x*b).mean(0)/scale.clip(.5,2),(x*x).mean(0)


def screen():
    history=[];cal=[];windows=[];amps=[];daily=[]
    shift,_=trial.clock_geometry()
    for end in (8,10,12,14):
        history.extend(trial.statistics(d,shift) for d in range(len(history),end))
        w=trial.weights(history)
        strength=coefficients(cal) if cal else None
        if strength is not None:amps.append(strength.tolist())
        groups=[[],[]]
        for d in (end,end+1):
            raw=trial.book.rawday(d);t=np.arange(d%3,len(raw),3)
            z=corrections(trial.changes(raw,t,shift,w))
            y=trial.labels(d).reshape(-1,50,3)[t]
            for j,ref in enumerate((trial.incumbent.fi.peer.reference,trial.incumbent.parent.fixed.earlier.early.baseline)):
                p=ref(d).reshape(-1,50,3)[t]
                if j==0:
                    train=t<len(raw)-60
                    cal.append(moments(p[train],y[train],z[train]))
                if strength is not None:groups[j].append((p+z,y,z*(strength-1)))
            if strength is not None:daily.append(trial.deltas(groups[0][-1:]))
        if strength is not None:
            windows.append([trial.deltas(rows) for rows in groups])
            print(end+1,'multiplier',strength,'gain',windows[-1][0],flush=True)
    passed=(np.array(windows).min(axis=(0,1))>0)&((np.asarray(daily)>0).sum(0)>=4)
    final=np.where(passed,coefficients(cal),1.)
    (OUT/'results.json').write_text(json.dumps(dict(windows=windows,daily=daily,
              past_multipliers=amps,eligible=passed.tolist(),final_multiplier=final.tolist()),indent=2))
    return passed,final


def layer():
    with np.load(trial.OUT/'model.npz') as f:
        w = f['m'], f['r']
        shift = int(f['shift'])
    strength = np.load(OUT/'multipliers.npy')
    def apply(raw,p):
        z = corrections(trial.changes(raw,np.arange(len(raw)),shift,w))
        return (p + z*(strength-1)).astype(p.dtype)
    return apply


def load():
    import experiment_periodic_near_carry_v1 as near
    parent = near.load()
    update = layer()
    return lambda raw,p: update(raw,parent(raw,p))


if __name__=='__main__':
    import replay_periodic_parent_v1 as replay
    OUT.mkdir(exist_ok=True)
    keep,final=screen()
    if keep.any():
        np.save(OUT/'multipliers.npy',final)
        replay.prepare()
        first=replay.evaluate(layer())
        (OUT/'raw_replay.json').write_text(json.dumps(first,indent=2))
        second=replay.evaluate(layer())
        np.testing.assert_allclose(first['pooled']['candidate'],second['pooled']['candidate'],atol=1.e-10,rtol=0)
        (OUT/'reload_verification.json').write_text(json.dumps(second,indent=2))
        print(second['pooled'],flush=True)
