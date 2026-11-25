"""Fit distinct causal book-factor normalizers for Y3's near and distant payoffs."""
import json
import numpy as np
import experiment_y3_payoff_factor_transport_v1 as factors

clock=factors.clock
OUT=clock.ART/'y3_band_specific_factors_v1'


def record(d,geos):
    raw=clock.book.rawday(d);t=np.arange(d%7,len(raw)-60,7)
    y=clock.labels(d).reshape(-1,50,3)[t]
    near=clock.RATIO*y[:,:,1]
    bands=[clock.center(v).reshape(-1,1) for v in (near,y[:,:,2]-near)]
    out=[]
    for (basis,cov,_),v in zip(geos,bands):
        a,_,b=factors.designs(raw,t,basis,cov)[0]
        moments=[]
        for z in (a,b):
            x=z.reshape(len(v),-1).astype(float)
            moments.append((x.T@x/len(x),x.T@v/len(x)))
        out.append(moments)
    return out


def predict(raw,t,geos,ws):
    out=[];errors=[]
    for (basis,cov,_),w in zip(geos,ws):
        views,err=factors.designs(raw,t,basis,cov)
        v=(views[-1]@w[-1]-views[0]@w[0])[:,:,0]
        out.append(.30*clock.center(v))
        errors.append(np.mean(err,axis=0).tolist() if err else [])
    return out,errors


def evaluate(rows):
    p,y,a,b=(np.concatenate(v) for v in zip(*rows))
    near,truth=clock.RATIO*p[:,:,1],clock.RATIO*y[:,:,1]
    def gain(x,yy,z):
        corr=lambda u:float(np.corrcoef(u.ravel(),yy.ravel())[0,1])
        return corr(x+z)-corr(x)
    return dict(y3=gain(p[:,:,2],y[:,:,2],a+b),
                near=gain(clock.center(near),clock.center(truth),a),
                far=gain(clock.center(p[:,:,2]-near),clock.center(y[:,:,2]-truth),b))


def screen():
    prior=factors.training_geometry()
    geos=[factors.basis_for(prior,slice(0,2),2),factors.basis_for(prior,slice(2,None),3)]
    hist=[];folds=[];daily=[];quality=[]
    for end in (8,10,12,14):
        hist.extend(record(d,geos) for d in range(len(hist),end))
        ws=[factors.transport.fit(h) for h in zip(*hist)]
        groups=[[],[]]
        for d in (end,end+1):
            raw=clock.book.rawday(d);t=np.arange(d%3,len(raw),3);y=clock.labels(d).reshape(-1,50,3)[t]
            z,err=predict(raw,t,geos,ws);quality.append(err)
            if d==end:
                ok=t<319
                for a,b in zip(z,predict(raw[:319],t[ok],geos,ws)[0]):
                    np.testing.assert_allclose(a[ok],b,atol=6e-7,rtol=0)
            for k,ref in enumerate((clock.incumbent.fi.peer.reference,clock.incumbent.parent.fixed.earlier.early.baseline)):
                p=ref(d).reshape(-1,50,3)[t]
                groups[k].append((p,y,*z))
            daily.append(evaluate(groups[0][-1:]))
        folds.append([evaluate(g) for g in groups]);print('window',end+1,folds[-1],flush=True)
    err=np.mean(quality,0)
    keep=bool(min(v['y3'] for fold in folds for v in fold)>0 and sum(v['y3']>0 for v in daily)>=5 and
              np.mean([row[0]['far'] for row in folds])>0 and np.all(err[:,-1]<err[:,0]))
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=daily,covariance_error=quality,keep=keep),indent=2))
    if keep:
        hist.extend(record(d,geos) for d in (14,15))
        state={}
        for j,(geo,h) in enumerate(zip(geos,zip(*hist))):
            for k,v in zip(('basis','cov','eigen','w'),(*geo,factors.transport.fit(h))):state[f'{k}{j}']=v
        np.savez(OUT/'model.npz',**state)
    return keep


def layer():
    parent=factors.incumbent.layer()
    with np.load(OUT/'model.npz') as s:
        geos=[[s[f'{k}{j}'] for k in ('basis','cov','eigen')] for j in (0,1)]
        ws=[s[f'w{j}'] for j in (0,1)]
    def apply(raw,p):
        q=parent(raw,p)
        for t in np.array_split(np.arange(len(q)),min(4,len(q))):
            z,_=predict(raw,t,geos,ws)
            q[t,:,2]+=z[0]+z[1]
        return q
    return apply


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if screen():factors.replay.verify(OUT,layer,factors.incumbent.OUT)
