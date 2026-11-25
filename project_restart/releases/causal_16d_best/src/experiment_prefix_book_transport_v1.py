"""Align stock book signals to a historical covariance using only prior observations of other stocks."""
import json

import numpy as np
import experiment_peer_crossing_payoff_v1 as parent
import replay_periodic_parent_v1 as replay

clock=parent.clock
OUT=clock.ART/'prefix_book_transport_v1'
BLOCK=80
INDEX=(0,1,2,3,4,5,6,7,8,10,12,14,15,17,18,19,20,21,22)
STOCKS=np.arange(50)


def observations(raw):
    _,local=clock.features(raw,np.arange(len(raw)))
    return local[:,:,INDEX]


def standard():
    daily=[]
    for day in range(8):
        x=observations(clock.book.rawday(day))[day%7::7]
        a=x.reshape(-1,x.shape[2]).astype(float)
        daily.append(a.T@a/len(a))
    g=np.mean(daily,axis=0)
    sd=np.sqrt(g.diagonal()).clip(.02)
    return sd,g/(sd[:,None]*sd[None,:])


def matrix_power(g,power):
    v,u=np.linalg.eigh(g)
    return (u*v.clip(.035)**power)@u.T


def features(raw,t,sd,cov):
    values=observations(raw)/sd
    dim=values.shape[2]
    # A small isotropic prior stabilizes directions shared by almost redundant fields.
    ref=.80*cov+.20*np.eye(dim)
    root=matrix_power(ref,.5);invroot=matrix_power(ref,-.5)
    arrays=[np.empty((len(t),50,dim),dtype='float32') for _ in range(3)]
    check=[]
    for b in np.unique(t//BLOCK):
        start=b*BLOCK
        history=values[max(0,start-720):start:3]
        ix=np.flatnonzero(t//BLOCK==b)
        sample=values[t[ix]]
        arrays[0][ix]=sample
        for side in (0,1):
            donors=(STOCKS%2==side)
            a=history[:,donors].reshape(-1,dim).astype(float)
            g=(a.T@a+2400*cov)/(len(a)+2400)
            g=.80*g+.20*np.eye(dim)
            # Positive symmetric covariance transport: M G M = reference covariance.
            transform=root@matrix_power(root@g@root,-.5)@root
            diag=np.sqrt(ref.diagonal()/g.diagonal()).clip(.6,1.7)
            receiver=~donors
            z=sample[:,receiver].astype(float)
            arrays[1][ix[:,None],STOCKS[receiver]]=z*diag
            arrays[2][ix[:,None],STOCKS[receiver]]=z@transform
            if start>=BLOCK:
                v=z.reshape(-1,dim);next_cov=v.T@v/len(v)
                # Check correlation drift on subsequent ticks in the other 25 stocks.
                def cov_error(mat):
                    aligned=mat.T@next_cov@mat
                    return float(((invroot@(aligned-cov)@invroot)**2).mean())
                check.append([cov_error(np.eye(dim)),cov_error(np.diag(diag)),cov_error(transform)])
    return arrays,check


def design(x):
    z=np.tanh(.75*x)
    return clock.center(np.concatenate((x,z,z[:,:,:9]*z[:,:,4:5],z[:,:,:9]*z[:,:,6:7]),axis=2))


def record(day,sd,cov):
    raw=clock.book.rawday(day);t=np.arange(day%7,len(raw)-60,7)
    y=clock.labels(day).reshape(-1,50,3)[t].astype(float)
    y[:,:,2]-=clock.RATIO*y[:,:,1]
    target=clock.center(y).reshape(-1,3)
    output=[]
    for a in features(raw,t,sd,cov)[0]:
        x=design(a).reshape(len(target),-1).astype(float)
        output.append((x.T@x/len(x),x.T@target/len(x)))
    return output


def fit(hist):
    models=[]
    for records in zip(*hist):
        gs,bs=(np.stack(c) for c in zip(*records))
        g=gs.mean(0)
        penalty=.55*np.diag(g.diagonal().clip(.0001))+.0004*np.eye(len(g))
        w=.7*np.linalg.solve(g+penalty,bs.mean(0))+.3*np.median(np.linalg.solve(gs+penalty,bs),axis=0)
        models.append(w)
    return np.stack(models)


def changes(raw,t,sd,cov,w):
    arr,check=features(raw,t,sd,cov)
    a,b,c=(design(v)@m for v,m in zip(arr,w))
    return [.32*clock.center(b-a),.32*clock.center(c-b)],check


def screen():
    sd,cov=standard()
    hist=[];folds=[];days=[];quality=[];far=[]
    for end in (8,10,12,14):
        hist.extend(record(day,sd,cov) for day in range(len(hist),end))
        w=fit(hist);groups=[[[],[]],[[],[]]];tail=[[],[]]
        for day in (end,end+1):
            raw=clock.book.rawday(day);t=np.arange(day%3,len(raw),3)
            y=clock.labels(day).reshape(-1,50,3)[t]
            z,check=changes(raw,t,sd,cov,w)
            quality.append(np.mean(check,axis=0).tolist())
            if day==end:
                ok=t<347
                for a,b in zip(z,changes(raw[:347],t[ok],sd,cov,w)[0]):
                    np.testing.assert_allclose(a[ok],b,rtol=0,atol=3e-7)
            for j,ref in enumerate((clock.incumbent.fi.peer.reference,clock.incumbent.parent.fixed.earlier.early.baseline)):
                p=ref(day).reshape(-1,50,3)[t]
                for k,diff in enumerate(z):
                    groups[j][k].append((p,y,diff))
                    if not j:
                        v=clock.center(p[:,:,2]-clock.RATIO*p[:,:,1])
                        true=clock.center(y[:,:,2]-clock.RATIO*y[:,:,1])
                        tail[k].append((v.ravel(),(v+diff[:,:,2]).ravel(),true.ravel()))
            days.append([clock.deltas(g[-1:]) for g in groups[0]])
        folds.append([[clock.deltas(g) for g in gs] for gs in groups])
        scores=[]
        for g in tail:
            a,b,y=(np.concatenate(c) for c in zip(*g))
            scores.append(float(np.corrcoef(b,y)[0,1]-np.corrcoef(a,y)[0,1]))
        far.append(scores)
        print('window',end+1,folds[-1][0], 'observable covariance error',quality[-1],flush=True)
    keep=(np.asarray(folds).min((0,1))>0)&((np.array(days)>0).sum(0)>=5)
    keep[:,2]&=np.mean(far,0)>0
    keep&=(np.diff(np.mean(quality,0))<0)[:,None]
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,days=days,far=far,observable_error=quality,keep=keep.tolist()),indent=2))
    if keep.any():
        hist.extend(record(d,sd,cov) for d in (14,15))
        np.savez(OUT/'model.npz',sd=sd,cov=cov,w=fit(hist),keep=keep)
    return keep.any()


def layer():
    base=parent.layer()
    with np.load(OUT/'model.npz') as state:
        sd,cov,w,keep=(state[k] for k in ('sd','cov','w','keep'))
    def predict(raw,p):
        q=base(raw,p)
        for t in np.array_split(np.arange(len(q)),min(len(q),4)):
            for z,mask in zip(changes(raw,t,sd,cov,w)[0],keep):
                q[t]+=z*mask
        return q
    return predict


def load():
    import experiment_periodic_near_carry_v1 as baseline
    upstream=baseline.load()
    update=layer()
    return lambda raw,p: update(raw,upstream(raw,p))


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if screen():replay.verify(OUT,layer,parent.OUT)
