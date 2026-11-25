"""Calibrate three horizon-specific heads on chronological order-clock forecasts."""
import json
import sys
import numpy as np
import torch
import y3_order_clock_references as history

clocks=history.incumbent
parent=clocks
stack,trial=history.stack,history.trial
OUT=trial.clock.ART/'joint_order_clock_calibration_v1'
ARMS=('unchanged','half increment','one increment','two increments','date-median amplitude','separate windows',
      'projected windows and clock increment','replicated windows')


def basis(delta,bands):
    return np.concatenate((delta[:,:,None],bands),2).astype('float32')


def record(day):
    raw=trial.clock.book.rawday(day)
    base=history.previous.references(day,raw)
    preds=history.references(day,raw)
    state=clocks.parameters(day-day%2)
    bands=clocks.design_bank(raw,np.arange(len(raw)))@state[:,-3:]
    y=trial.clock.labels(day).reshape(-1,50,3)
    grams=[]
    for old,p in zip(base,preds):
        z=basis(p[:,:,2]-old[:,:,2],bands)
        a=np.concatenate((np.ones_like(p[:,:,:1]),p,z,y),2).reshape(-1,11).astype(float)
        grams.append(a.T@a)
    return grams


def fitted(days,k):
    dates=np.asarray(days).mean(1)
    def eq(g):
        c=stack.covariance(g)
        existing=c[:3,:3]
        cross=c[3:7,:3]
        projection=np.linalg.solve(existing+.025*np.diag(existing.diagonal()),cross.T)
        target=c[:,-3+k]*(c[k,k]/c[k,-3+k])-c[:,k]
        return c[3:7,3:7]-cross@projection,target[3:7]-projection.T@target[:3]
    gram,b=eq(dates.sum(0))
    local=[eq(d) for d in dates]
    gradient=np.stack([v for _,v in local])
    noise=gradient.std(0)*np.sqrt(len(local))
    b*=abs(b)/(abs(b)+noise+1e-9)
    ws=np.zeros((len(ARMS),4))
    ws[1:4,0]=(.5,1.,2.)
    opt=[v[0]/g[0,0].clip(1e-8) for g,v in local]
    ws[4,0]=np.clip(np.median(opt)*.65,-2.,3.)
    for i,ids in ((5,np.arange(1,4)),(6,np.arange(4)),(7,np.arange(1,4))):
        g=gram[np.ix_(ids,ids)]
        rhs=b[ids].copy()
        if i==7:
            rhs*=abs(np.sign(gradient[:,ids]).mean(0))
        w=np.linalg.solve(g+1.2*np.diag(g.diagonal().clip(1e-7)),rhs)
        ws[i,ids]=w.clip(-2.5,2.5)
    return ws


def gain(g,w,k):
    c=stack.covariance(g)
    coef=np.zeros((len(w),7))
    coef[:,k]=1
    coef[:,3:]=w
    values=coef@c[:7,-3+k]/np.sqrt(((coef@c[:7,:7])*coef).sum(1)*c[-3+k,-3+k])
    return values-values[0]


def train():
    OUT.mkdir(exist_ok=True)
    stats=[record(day) for day in range(8,16)]
    selected,weights,report=[],[],[]
    for k in range(3):
        folds,daily=[],[]
        for n in (2,4,6):
            ws=fitted(stats[:n],k)
            folds.append([gain(g,ws,k) for g in np.sum(stats[n:n+2],0)])
            daily.extend([[gain(g,ws,k) for g in pair] for pair in stats[n:n+2]])
        f,d=np.asarray(folds),np.asarray(daily)
        mean=f.mean(0)
        eligible=[a for a in range(1,len(ARMS)) if mean[:,a].min()>0 and f[-1,:,a].min()>0
                  and (f[:,:,a]>0).sum(0).min()>=2 and (d[:,:,a]>0).sum(0).min()>=4]
        pick=max(eligible,key=lambda a:mean[:,a].min()) if eligible else 0
        weights.append(fitted(stats,k)[pick])
        selected.append(pick)
        report.append(dict(mean=mean.tolist(),folds=f.tolist(),daily=d.tolist()))
        print('target',k+1,ARMS[pick],'mean',np.round(mean,6),flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(selected=selected,arms=ARMS,targets=report,training_days=list(range(9,17))),indent=2))
    if any(selected):
        np.savez(OUT/'model.npz',w=np.array(weights).T,selected=selected)
    return any(selected)


def readout():
    with np.load(clocks.OUT/'model.npz') as archive:
        st=dict(archive)
    with np.load(OUT/'model.npz') as archive:
        w=archive['w']
    def apply(raw,early):
        s=clocks.design_bank(raw,np.arange(len(raw)))@st['clocks']
        design=np.concatenate((trial.clock.center(early),s),2)
        delta=clocks.changes(design,st)[int(st['pick'])]
        q=early.copy()
        q[:,:,2]+=delta
        z=basis(q[:,:,2]-early[:,:,2],s[:,:,-3:])
        return q,q+(z@w).astype('float32')
    return apply


def layer():
    older=clocks.parent.layer()
    apply=readout()
    return lambda raw,p:apply(raw,older(raw,p))[1]


def replay_with():
    ver,early=history.previous.replay_with()
    apply=readout()
    def read(d,raw):
        p,h,n=early(d,raw)
        return apply(raw,p)[1],apply(raw[:n],h)[1],n
    return ver,read


def replay():
    expected=json.loads((clocks.OUT/'reload_verification.json').read_text())
    previous=None
    for name in ('raw_replay','reload_verification'):
        ver,early=history.previous.replay_with()
        apply=readout()
        rows=[]
        daily={}
        for d in range(16,20):
            raw=ver.trace.replay.original.features(ver.trace.replay.original.DATA,d)
            p,h,n=early(d,raw)
            baseline,full=apply(raw,p)
            _,head=apply(raw[:n],h)
            err=float(abs(full[:n]-head).max())
            assert err<1e-4 and np.isfinite(full).all()
            row=baseline.reshape(-1,3),full.reshape(-1,3),ver.trace.replay.labels(d)
            rows.append(row)
            s=ver.trace.older.measure([row])
            assert abs(s['baseline_score']-expected['daily'][str(d+1)]['score'])<1e-10
            daily[str(d+1)]=dict(score=s['score'],gain=s['gain'],prefix_error=err,corr=s['candidate'])
        s=ver.trace.older.measure(rows)
        assert s['rows']==936250
        np.testing.assert_allclose(s['base'],expected['pooled']['candidate'],atol=1e-10,rtol=0)
        result=dict(parent=clocks.OUT.name,pooled=s,daily=daily)
        if previous is not None:
            assert previous==result
        (OUT/(name+'.json')).write_text(json.dumps(result,indent=2))
        print(name,s,flush=True)
        previous=result


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['replay'] or train():
        replay()
