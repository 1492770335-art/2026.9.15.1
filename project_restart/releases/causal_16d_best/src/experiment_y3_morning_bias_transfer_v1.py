"""Estimate morning stock-level errors from completed price paths and carry persistent offsets into the afternoon."""
import json
import sys

import numpy as np

import experiment_compounded_return_audit_v1 as prices
import y3_intraday_replication_references as history

parent,base = history.incumbent,history.base
OUT = parent.OUT.parent/'y3_morning_bias_transfer_v1'
ARMS = ('unchanged','reassigned morning error','morning forecasts only','historical stock offset',
        'completed Y3 error','distant interval error','agreement across morning thirds',
        'morning beyond historical offset','morning and historical offset','consistent full and distant errors',
        'observed morning prices only','afternoon-decaying error')


def signals_at_noon(raw,p,geo):
    k,coef,row = geo
    x = raw[:2401]
    close = prices.compound(x,k)
    missing = np.vstack((np.zeros((1,50),int),(~prices.present(x,row)).cumsum(0)))
    t = np.concatenate([np.arange(start,start+120,4) for start in range(0,2401-180+1,180)])
    good = (missing[t+61]==missing[t])
    def relative(v):
        return good*(v-(v*good).sum(1,keepdims=True)/good.sum(1,keepdims=True).clip(1))
    truth = [coef[j,0]+coef[j,1]*prices.span(close,t,h,k) for j,h in ((1,20),(2,60))]
    observed = np.stack((relative(truth[1]),relative(truth[1]-base.RATIO*truth[0])),2)
    forecast = np.stack((relative(p[t,:,2]),relative(p[t,:,2]-base.RATIO*p[t,:,1])),2)
    # The slope comes from morning outcomes already available when the afternoon opens.
    slope = ((observed*forecast).sum((0,1))/(forecast**2).sum((0,1)).clip(1e-6)).clip(.35,2.8)
    e = observed/slope-forecast
    def avg(arr,mask):
        n=good[mask].sum(0)
        v=arr[mask].sum(0)/(n+12)[:,None]
        return v-v.mean(0)
    ids=np.arange(len(t))
    whole = avg(e,ids)
    parts = np.stack([avg(e,ix) for ix in np.array_split(ids,3)])
    robust = np.median(parts,0)
    robust *= (np.sign(parts)==np.sign(robust)).mean(0)
    return np.column_stack((whole,robust,avg(forecast,ids)[:,0],avg(observed/slope,ids)[:,0]))


def design(m,x_prior,n):
    a,b,robust,_,forecast,obs=m.T
    stable=np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0)
    features=np.stack((np.roll(a,13),forecast,x_prior,a,b,robust,a-x_prior,.6*a+.4*x_prior,stable,obs,a),1)
    features-=features.mean(0)
    x=np.zeros((n,50,features.shape[1]),np.float32)
    if n>2401:
        x[2401:]=features
        x[2401:,:,-1]*=np.exp(-np.arange(n-2401)/1100.)[:,None]
    return x


def historical(estimates):
    return np.mean([m[:,0] for m in estimates],0) if estimates else np.zeros(50)


def fit(rows):
    # Pearson derivative and curvature with respect to each scalar correction, fit across dates and references.
    derivatives,curvatures = [],[]
    for pair in rows:
        vals,curves=[],[]
        for m in pair:
            c=base.covariance(m)
            gradients,hs=[],[]
            for k in (0,1):
                a,b=k,k-2
                baseline=c[a,b]/np.sqrt(c[a,a]*c[b,b])
                scale=c[2:-2,a]/c[a,a]
                gradients.append(c[2:-2,b]/np.sqrt(c[a,a]*c[b,b])-baseline*scale)
                hs.append((c.diagonal()[2:-2]/c[a,a]-scale**2).clip(0))
            vals.append(gradients)
            curves.append(hs)
        derivatives.append(np.mean(vals,0))
        curvatures.append(np.mean(curves,0))
    g=np.array(derivatives)
    avg=g.mean(0)
    rel=abs(avg)/(abs(avg)+g.std(0)/np.sqrt(len(g))+.00002)
    full,far=avg*rel
    agreement=np.where(full*far>0,np.sign(full)*np.minimum(abs(far),abs(full)),0)
    return (.35*agreement/(.04+.5*np.mean(curvatures,(0,1)))).clip(-.55,.55)


def records(day):
    raw = history.trial.clock.book.rawday(day)
    y = history.trial.clock.labels(day).reshape(-1,50,3)
    ps = history.references(day,raw) if day>=8 else parent.parent.parent.training_references(day,raw)
    return raw,y,ps


def train():
    OUT.mkdir(exist_ok=True)
    geo=history.trial.curve.settled.historical_geometry()
    past,rows,folds,daily = [],[],[],[]
    for d in range(4,16):
        raw,y,ps=records(d)
        if d in (8,12):
            w=fit(rows)
            fixed_prior=historical(past)
            group=[]
        noon=[signals_at_noon(raw,p,geo) for p in ps]
        # Freeze the stock prior for an entire held-out training fold, as in deployment.
        prior=fixed_prior if d>=8 else historical(past)
        pair=[base.records(p,y,design(m,prior,len(p))) for p,m in zip(ps,noon)]
        if d>=8:
            group.append(pair)
            daily.append([base.score(a,w)[0] for a in pair])
            if d in (11,15):
                folds.append([base.score(m,w) for m in np.sum(group,0)])
        past.append(np.mean(noon,0))
        rows.append(pair)
    f,ds=np.array(folds),np.array(daily)
    ok=[j for j in range(4,len(ARMS)) if j!=10 and f[:,:,:,j].min()>0 and
        (ds[:,:,j]>0).sum(0).min()>=5 and (f[:,:,0,j].mean(0)>f[:,:,0,1:4].mean(0).max(1)).all()]
    pick=max(ok,key=lambda j:f[:,:,0,j].min()) if ok else 0
    w=fit(rows)
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,arms=ARMS,folds=f.tolist(),daily=ds.tolist(),
        weights=w.tolist(),windows=[[9,12],[13,16]]),indent=2))
    print('selected',ARMS[pick],f[:,0,0].round(6).tolist(),flush=True)
    if pick:
        np.savez(OUT/'model.npz',pick=pick-1,w=w[pick-1],prior=historical(past),k=geo[0],coef=geo[1],template=geo[2])
    return bool(pick)


def correction():
    with np.load(OUT/'model.npz') as f:
        idx,w,prior=int(f['pick']),float(f['w']),f['prior']
        geo=float(f['k']),f['coef'],f['template']
    def add(raw,p):
        if len(p)<=2401:
            return np.zeros(p.shape[:2],np.float32)
        return design(signals_at_noon(raw,p,geo),prior,len(p))[:,:,idx]*w
    return add


def layer():
    before,update=parent.layer(),correction()
    def apply(raw,initial):
        p=before(raw,initial).copy()
        p[:,:,2]+=update(raw,p)
        return p
    return apply


def replay_with():
    update=correction()
    def read(day,raw):
        q,h,n=history.read(day,raw)
        q[:,:,2]+=update(raw,q)
        h[:,:,2]+=update(raw[:n],h)
        return q,h,n
    return history.stack.verifier,read


if __name__=='__main__' and (sys.argv[1:]==['replay'] or train()):
    base.replay.replay(sys.modules[__name__])
