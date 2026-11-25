"""Evaluate two payoff-specific corrections at every internal tick with a walk-forward readout."""
import json
import numpy as np
from scipy.optimize import minimize

import experiment_y3_band_specific_factors_v1 as trial

clock=trial.clock
OUT=clock.ART/'y3_factor_band_readout_v1'


def moments(p,y,terms):
    a,b=[v[:,:,2].ravel().astype(float) for v in (p,y)]
    x=np.column_stack([z.ravel() for z in terms]).astype(float)
    a-=a.mean();b-=b.mean();x-=x.mean(0)
    g=(a*a).mean().clip(1e-9)
    beta=((a*b).mean()/g).clip(.4,2.5)
    x-=a[:,None]*((a@x)/(len(a)*g))
    return x.T@x/len(x), x.T@b/len(x)/beta


def coefficients(hist):
    g,b=(np.mean(c,0) for c in zip(*hist))
    penalty=3*np.diag(g.diagonal().clip(1e-8))
    h=g+penalty;rhs=b+penalty@np.ones(2)
    unit=g.diagonal().mean().clip(1e-8)
    cost=lambda u:(.5*u@h@u-rhs@u)/unit
    fit=minimize(cost,np.ones(2),jac=lambda u:(h@u-rhs)/unit,bounds=[(0,2),(0,2)],method='L-BFGS-B')
    if not fit.success:raise ValueError(fit.message)
    return fit.x


def screen():
    factor=trial.factors
    prior=factor.training_geometry()
    geos=[factor.basis_for(prior,slice(0,2),2),factor.basis_for(prior,slice(2,None),3)]
    hist=[];cal=[];folds=[[],[]];daily=[[],[]];multipliers=[]
    for end in (8,10,12,14):
        hist.extend(trial.record(d,geos) for d in range(len(hist),end))
        ws=[factor.transport.fit(h) for h in zip(*hist)]
        w=coefficients(cal) if cal else None
        groups=[[[],[]],[[],[]]]
        for d in (end,end+1):
            raw=clock.book.rawday(d);t=np.arange(len(raw))
            y=clock.labels(d).reshape(-1,50,3)
            pieces=[trial.predict(raw,tt,geos,ws)[0] for tt in np.array_split(t,4)]
            z=[np.concatenate(v) for v in zip(*pieces)]
            if d==end:
                for a,b in zip(z,trial.predict(raw[:521],t[:521],geos,ws)[0]):
                    np.testing.assert_allclose(a[:521],b,atol=6e-7,rtol=0)
            for k,ref in enumerate((clock.incumbent.fi.peer.reference,clock.incumbent.parent.fixed.earlier.early.baseline)):
                p=ref(d).reshape(y.shape)
                if not k:cal.append(moments(p[:-60],y[:-60],[v[:-60] for v in z]))
                for j,c in enumerate((np.ones(2),w)):
                    if c is None:continue
                    groups[j][k].append((p,y,c[0]*z[0],c[1]*z[1]))
                    if not k:daily[j].append(trial.evaluate(groups[j][k][-1:]))
        for j in range(2):
            if groups[j][0]:
                folds[j].append([trial.evaluate(g) for g in groups[j]])
        multipliers.append(w.tolist() if w is not None else None)
        print('window',end+1,'fixed',folds[0][-1],'fitted',folds[1][-1] if w is not None else None,flush=True)
    def passed(i):
        return (min(v['y3'] for row in folds[i] for v in row)>0 and
                np.mean([row[0]['far'] for row in folds[i]])>0 and
                sum(v['y3']>0 for v in daily[i])>=len(daily[i])//2+1)
    selected='fixed' if passed(0) else 'fitted' if passed(1) else None
    final=np.ones(2) if selected=='fixed' else coefficients(cal)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=daily,selected=selected,
                                       weights=multipliers,final=final.tolist(),sampling_stride=1),indent=2))
    if selected:
        hist.extend(trial.record(d,geos) for d in (14,15))
        state={'alpha':final}
        for j,(geo,h) in enumerate(zip(geos,zip(*hist))):
            for k,v in zip(('basis','cov','eigen','w'),(*geo,factor.transport.fit(h))):state[f'{k}{j}']=v
        np.savez(OUT/'model.npz',**state)
    return selected


def layer():
    parent=trial.factors.incumbent.layer()
    with np.load(OUT/'model.npz') as s:
        geos=[[s[f'{k}{j}'] for k in ('basis','cov','eigen')] for j in (0,1)]
        ws=[s[f'w{j}'] for j in (0,1)];alpha=s['alpha']
    def apply(raw,p):
        q=parent(raw,p)
        for t in np.array_split(np.arange(len(q)),min(4,len(q))):
            z,_=trial.predict(raw,t,geos,ws)
            q[t,:,2]+=alpha[0]*z[0]+alpha[1]*z[1]
        return q
    return apply


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if screen():trial.factors.replay.verify(OUT,layer,trial.factors.incumbent.OUT)
