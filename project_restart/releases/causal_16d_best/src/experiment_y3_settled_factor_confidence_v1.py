"""Estimate the current worth of two Y3 factor revisions from completed returns on disjoint stocks."""
import json
import numpy as np
from scipy.special import ndtr

import experiment_compounded_return_audit_v1 as compound
import experiment_y3_factor_band_readout_v1 as previous

trial=previous.trial
clock=trial.clock
OUT=clock.ART/'y3_settled_factor_confidence_v1'
REFS=(clock.incumbent.fi.peer.reference,clock.incumbent.parent.fixed.earlier.early.baseline)


def traces():
    prior=trial.factors.training_geometry()
    geos=[trial.factors.basis_for(prior,slice(0,2),2),trial.factors.basis_for(prior,slice(2,None),3)]
    history=[]
    for end in (8,10,12,14):
        history.extend(trial.record(d,geos) for d in range(len(history),end))
        weights=[trial.factors.transport.fit(h) for h in zip(*history)]
        for d in (end,end+1):
            raw=clock.book.rawday(d)
            pieces=[trial.predict(raw,t,geos,weights)[0] for t in np.array_split(np.arange(len(raw)),4)]
            yield d,raw,np.stack([np.concatenate(v) for v in zip(*pieces)],axis=2)


def historical_geometry():
    with np.load(compound.OUT/'model.npz') as s:return float(s['k']),s['coeff'],s['row']


def proxy(raw,geometry):
    k,coeff,row=geometry
    price=compound.compound(raw,k);n=len(raw)
    missing=np.vstack((np.zeros((1,50),int),(~compound.present(raw,row)).cumsum(0)))
    t=np.arange(max(0,n-60))
    y=coeff[2,0]+coeff[2,1]*compound.span(price,t,60,k)
    return y,missing[t+61]==missing[t]


def adjustment(raw,p,z,geo,contexts=None):
    y,valid=proxy(raw,geo)
    w=np.full((len(p),2,2),.5)
    for side in (0,1):
        cols=np.arange(side,50,2)
        rows=[];states=[]
        # Only release a block once its last 60-tick outcome has settled.
        for start in range(0,len(y)-119,120):
            t=np.arange(start,start+120,4)
            observed=valid[t][:,cols].astype(float)
            def center(v):
                avg=(v*observed).sum(1,keepdims=True)/observed.sum(1,keepdims=True).clip(1)
                return (v-avg)*observed
            a=center(p[t][:,cols,2])
            b=center(y[t][:,cols])
            x=np.stack([center(z[t][:,cols,j]) for j in range(2)],axis=2)
            rows.append(np.r_[(a*a).mean(),(a*b).mean(),(x*a[:,:,None]).mean((0,1)),
                              (x*b[:,:,None]).mean((0,1)),(x*x).mean((0,1))])
            pos=start+119+60
            blocks=np.array(rows[-16:])
            if contexts is None:
                moments=blocks.mean(0)
            else:
                states.append(contexts[t].mean(0))
                distance=np.mean((np.array(states[-16:])-contexts[pos])**2,axis=1)
                similarity=.15+.85*np.exp(-.8*distance)
                similarity/=similarity.sum()
                moments=similarity@blocks
            slope=(moments[1]/max(moments[0],1e-6)).clip(.4,2.)
            gain=blocks[:,4:6]/slope-blocks[:,2:4]-.5*blocks[:,6:8]
            # Uncertainty is estimated across blocks, since adjacent returns overlap.
            if contexts is None:
                count=len(blocks);mean=gain.mean(0);var=gain.var(0)
            else:
                count=1/np.sum(similarity**2)
                mean=similarity@gain
                var=similarity@((gain-mean)**2)
            se=np.sqrt((var+.00016**2)/count)
            evidence=(mean/se).clip(-3,3)
            weight=.5+count/(6+count)*(ndtr(evidence)-.5)
            w[pos:pos+120,1-side]=weight
    near=z[:,:,0]*w[:,np.arange(50)%2,0]
    far=z[:,:,1]*w[:,np.arange(50)%2,1]
    return clock.center(near),clock.center(far),w


def screen():
    geo=historical_geometry();folds=[];daily=[];means=[]
    for d,raw,z in traces():
        if d%2==0:groups=[[[],[]],[[],[]]]
        y=clock.labels(d).reshape(-1,50,3)
        for k,ref in enumerate(REFS):
            p=ref(d).reshape(y.shape)
            near,far,w=adjustment(raw,p,z,geo)
            if k==0:
                means.append(w.mean((0,1)).tolist())
                if d%2==0:
                    for n in (119,179,180,181,541,1401):
                        a,b,_=adjustment(raw[:n],p[:n],z[:n],geo)
                        np.testing.assert_allclose(a,near[:n],rtol=0,atol=1e-12)
                        np.testing.assert_allclose(b,far[:n],rtol=0,atol=1e-12)
            groups[0][k].append((p,y,z[:,:,0]*.5,z[:,:,1]*.5))
            groups[1][k].append((p,y,near,far))
        daily.append([trial.evaluate(g[0][-1:]) for g in groups])
        if d%2:
            folds.append([[trial.evaluate(g) for g in pair] for pair in groups])
            print('window',d,'fixed / live',folds[-1][0][0],folds[-1][1][0],flush=True)
    v=np.array([[[p['y3'] for p in row] for row in fold] for fold in folds])
    keep=bool(np.min(v[:,1])>0 and np.mean(v[:,1]-v[:,0])>0 and
              sum(d[1]['y3']>0 for d in daily)>=5 and np.mean([r[1][0]['far'] for r in folds])>0)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=daily,weights=means,keep=keep),indent=2))
    return keep


def layer():
    base=trial.factors.incumbent.layer();geo=historical_geometry()
    with np.load(previous.OUT/'model.npz') as s:
        geos=[[s[f'{k}{j}'] for k in ('basis','cov','eigen')] for j in (0,1)]
        ws=[s[f'w{j}'] for j in (0,1)]
    def apply(raw,p):
        q=base(raw,p)
        parts=[trial.predict(raw,t,geos,ws)[0] for t in np.array_split(np.arange(len(q)),min(4,len(q)))]
        z=np.stack([np.concatenate(v) for v in zip(*parts)],axis=2)
        a,b,_=adjustment(raw,q,z,geo)
        q[:,:,2]+=a+b
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as base
    upstream=base.load();adjust=layer()
    return lambda raw,p:adjust(raw,upstream(raw,p))


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if screen():trial.factors.replay.verify(OUT,layer,trial.factors.incumbent.OUT)
