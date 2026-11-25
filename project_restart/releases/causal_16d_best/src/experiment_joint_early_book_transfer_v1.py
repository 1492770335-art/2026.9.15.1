"""Give each horizon a readout of the original stock-transition and ordered-pressure expert outputs."""
import json
import sys

import numpy as np
import torch
import experiment_y3_pressure_horizon_controls_v1 as pressure
import joint_horizon_stage_references as history

transition=pressure.parent
parent,stack,trial=history.incumbent,history.stack,history.trial
OUT=trial.clock.ART/'joint_early_book_transfer_v1'
ARMS=('unchanged','fixed positive transition mixture','stock transition terms','ordered pressure',
      'unconditional pressure','transitions and ordered pressure','full early library',
      'separate near and terminal pressure','orthogonal interaction only')
FIELDS=11


def estimates(past):
    transitions,pressures=zip(*past)
    mu,bank=transition.geometry(transitions)
    return dict(mu=mu,bank=bank,signature=pressure.fit(pressures))


def signals(raw,prefix,st):
    values=transition.terms(transition.incumbent.state(raw)[1],st['mu'],st['bank'])
    order=pressure.signature.predict(prefix,st['signature'])
    # The head is calibrated in regularized units, matching the original .10 pressure readout.
    return np.concatenate((np.stack(values,2),.10*order),2).astype('float32')


def observations(day,x,raw):
    y=trial.clock.labels(day).reshape(-1,50,3)
    ref=history.references(day,raw)
    grams=[]
    for p in ref:
        a=np.concatenate((np.ones_like(p[:,:,:1]),p,x,y),2).reshape(len(y)*50,-1).astype(float)
        grams.append(a.T@a)
    return np.stack(grams)


def fit(records,target):
    days=np.asarray(records).mean(1)
    ids=np.arange(3,3+FIELDS)
    def terms(g):
        c=stack.covariance(g)
        h=c[:3,:3]
        cross=c[np.ix_(ids,np.arange(3))]
        projection=np.linalg.solve(h+.05*np.diag(h.diagonal()),cross.T)
        rhs=c[:,3+FIELDS+target]*c[target,target]/c[target,3+FIELDS+target]-c[:,target]
        return c[np.ix_(ids,ids)]-cross@projection,rhs[ids]-projection.T@rhs[:3]
    gram,rhs=terms(days.sum(0))
    gradients=np.stack([terms(d)[1] for d in days])
    noise=gradients.std(0)*np.sqrt(len(days))
    rhs*=abs(rhs)/(abs(rhs)+noise+1e-8)
    w=np.zeros((len(ARMS),3+FIELDS))
    w[:,target]=1
    w[1,3:6]=.25
    for i,cols in enumerate(((0,1,2,3,4),(6,7,9,10),(5,8),(0,1,2,3,4,6,7,9,10),
                             tuple(range(FIELDS)),(7,10),(2,3,4)),start=2):
        idx=np.array(cols)
        g=gram[np.ix_(idx,idx)]
        w[i,3+idx]=np.linalg.solve(g+1.5*np.diag(g.diagonal().clip(.0001)),rhs[idx]).clip(-1.,1.)
    return w


def gains(g,w,target):
    c=stack.covariance(g)
    s=(w@c[:-3,-3+target])/np.sqrt(((w@c[:-3,:-3])*w).sum(1)*c[-3+target,-3+target])
    return s-s[0]


def train():
    OUT.mkdir(exist_ok=True)
    past=[]
    stats=[]
    for day in range(16):
        raw=trial.clock.book.rawday(day)
        y=trial.clock.labels(day).reshape(-1,50,3)
        prefix=pressure.signature.observations(raw)
        if day>=8:
            if day%2==0:
                st=estimates(past)
            x=signals(raw,prefix,st)
            if day==8:
                n=311
                np.testing.assert_allclose(x[:n],signals(raw[:n],pressure.signature.observations(raw[:n]),st),atol=3e-6,rtol=0)
            stats.append(observations(day,x,raw))
        t=np.arange(day%5,len(y)-60,5)
        past.append((transition.incumbent.stats(raw,y,day),pressure.record(prefix,t,y)))
    picks,weights,reports=[],[],[]
    for target in range(3):
        folds,daily=[],[]
        for end in (2,4,6):
            w=fit(stats[:end],target)
            folds.append([gains(g,w,target) for g in sum(stats[end:end+2])])
            daily.extend([[gains(g,w,target) for g in pair] for pair in stats[end:end+2]])
        f,d=np.array(folds),np.array(daily)
        avg=f.mean(0)
        ok=[i for i in range(1,len(ARMS)) if (avg[:,i]>0).all() and f[-1,:,i].min()>0
            and (f[:,:,i]>0).sum(0).min()>=2 and (d[:,:,i]>0).sum(0).min()>=4]
        pick=max(ok,key=lambda i:avg[:,i].min()) if ok else 0
        picks.append(pick)
        weights.append(fit(stats,target)[pick,3:])
        reports.append(dict(mean=avg.tolist(),folds=f.tolist(),daily=d.tolist()))
        print(target+1,ARMS[pick],np.round(avg,5),flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(selected=picks,arms=ARMS,targets=reports,rolling_days=list(range(11,17))),indent=2))
    if any(picks):
        final=estimates(past)
        # Re-estimation through day 16 must reproduce the frozen upstream experts.
        with np.load(transition.OUT/'model.npz') as old:
            for k in ('mu','bank'):
                np.testing.assert_allclose(final[k],old[k],atol=2e-5,rtol=1e-5)
        with np.load(pressure.OUT/'model.npz') as old:
            np.testing.assert_allclose(final['signature'],old['w'],atol=2e-5,rtol=1e-5)
        np.savez(OUT/'model.npz',w=np.array(weights).T,pick=picks,**final)
    return any(picks)


def correction():
    with np.load(OUT/'model.npz') as archive:
        model=dict(archive)
    def apply(raw,p):
        x=signals(raw,pressure.signature.observations(raw),model)
        return p+(x@model['w']).astype('float32')
    return apply


def layer():
    start=parent.layer()
    update=correction()
    return lambda raw,p:update(raw,start(raw,p))


def replay():
    previous=None
    for name in ('raw_replay','reload_verification'):
        update=correction()
        verifier,load=history.replay_with()
        trace=verifier.trace
        rows,daily=[],{}
        old=json.loads((parent.OUT/'reload_verification.json').read_text())
        for d in range(16,20):
            raw=trace.replay.original.features(trace.replay.original.DATA,d)
            p,h,n=load(d,raw)
            q=update(raw,p)
            head=update(raw[:n],h)
            err=float(abs(head-q[:n]).max())
            assert err<1e-4 and np.isfinite(q).all()
            row=p.reshape(-1,3),q.reshape(-1,3),trace.replay.labels(d)
            scored=trace.older.measure([row])
            assert abs(scored['baseline_score']-old['daily'][str(d+1)]['score'])<1e-10
            daily[str(d+1)]=dict(score=scored['score'],gain=scored['gain'],prefix_error=err,corr=scored['candidate'])
            rows.append(row)
        pooled=trace.older.measure(rows)
        assert pooled['rows']==936250 and abs(pooled['baseline_score']-old['pooled']['score'])<1e-10
        report=dict(parent=parent.OUT.name,pooled=pooled,daily=daily)
        if previous is not None:
            assert report==previous
        (OUT/(name+'.json')).write_text(json.dumps(report,indent=2))
        print(name,pooled,flush=True)
        previous=report


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['replay'] or train():
        replay()
