"""Align the covariance of supervised book factors trained on distinct segments of the return path."""
import json
import numpy as np
import experiment_prefix_book_transport_v1 as transport
import experiment_grouped_book_channels_v1 as incumbent
import experiment_dominant_far_block_v1 as diagnostic
import replay_periodic_parent_v1 as replay

clock=transport.clock
OUT=clock.ART/'y3_payoff_factor_transport_v1'
EDGES=(0,5,20,30,40,50,60)
K=5


def book(raw,t,ids):
    x=clock.center(np.tanh(np.arcsinh(raw[:,:,ids].astype(np.float32))))
    c=np.concatenate((np.zeros_like(x[:1],dtype=float),x.cumsum(0,dtype=float)))
    hi=t+1;lo=(hi-32).clip(0)
    past=((c[hi]-c[lo])/(hi-lo)[:,None,None]).astype(np.float32)
    return np.concatenate((x[t]-past,past),axis=2)


CHANNELS=tuple(np.array_split(np.arange(83),8))


def training_geometry():
    hist=[]
    for d in range(8):
        raw=clock.book.rawday(d);t=np.arange(d%11,len(raw)-60,11)
        x=np.concatenate([book(raw,t,ids) for ids in CHANNELS],axis=2)
        a=x.reshape(-1,x.shape[2]).astype(float)
        price=(raw[:,:,0].astype(float)-.00136).cumsum(0)
        pay=clock.center(np.diff(np.stack([price[t+h] for h in EDGES],axis=2),axis=2))
        y=pay.reshape(-1,len(EDGES)-1)
        hist.append((a.T@a/len(a),a.T@y/len(a)))
    g,b=(np.mean(v,0) for v in zip(*hist))
    sd=np.sqrt(g.diagonal()).clip(.025)
    gram=g/(sd[:,None]*sd)
    eig,vec=np.linalg.eigh(gram+.4*np.eye(len(gram)))
    root=(vec*eig**-.5)@vec.T
    # Retain feature directions whose link to the same lead-time returns repeats in disjoint dates.
    rhs=[root@(np.mean([v[1] for v in hist[k::2]],0)/sd[:,None]) for k in (0,1)]
    return g,sd,root,rhs


def basis_for(prior,columns,rank):
    g,sd,root,full_rhs=prior
    rhs=[m[:,columns] for m in full_rhs]
    size=np.sqrt(np.mean([np.sum(a*a,axis=0) for a in rhs],0)).clip(.01)
    a,b=(v/size for v in rhs)
    eigen,u=np.linalg.eigh(.5*(a@b.T+b@a.T))
    basis=(root@u[:,-rank:])/sd[:,None]
    cov=basis.T@g@basis
    units=np.sqrt(cov.diagonal()).clip(.02)
    return basis/units,cov/(units[:,None]*units),eigen[-rank:]


def geometry():
    return basis_for(training_geometry(),slice(None),K)


def factors(raw,basis):
    result=np.zeros((*raw.shape[:2],basis.shape[1]),np.float32)
    pos=0;t=np.arange(len(raw))
    for ids in CHANNELS:
        width=2*len(ids)
        result+=book(raw,t,ids)@basis[pos:pos+width].astype(np.float32)
        pos+=width
    return result


def designs(raw,t,basis,cov):
    x=factors(raw,basis)
    width=len(cov)
    ref=.8*cov+.2*np.eye(width);root=transport.matrix_power(ref,.5)
    outs=[x[t]]+[np.empty((len(t),50,width),np.float32) for _ in range(2)]
    error=[]
    for b in np.unique(t//80):
        end=80*b
        hist=x[max(0,end-720):end:3]
        ix=np.flatnonzero(t//80==b)
        for side in (0,1):
            stocks=np.arange(1-side,50,2)
            h=hist[:,side::2].reshape(-1,width).astype(float)
            g=(h.T@h+2400*cov)/(len(h)+2400)
            g=.8*g+.2*np.eye(width)
            single=np.diag(np.sqrt(ref.diagonal()/g.diagonal()).clip(.6,1.7))
            joint=root@transport.matrix_power(root@g@root,-.5)@root
            v=x[t[ix]][:,stocks]
            loc=np.ix_(ix,stocks)
            for dest,matrix in zip(outs[1:],(single,joint)):dest[loc]=v@matrix
            if end>0:
                observed=v.reshape(-1,width).astype(float);gram=observed.T@observed/len(observed)
                error.append([float(((m.T@gram@m-cov)**2).mean()) for m in (np.eye(width),single,joint)])
    def lift(a):
        s=np.tanh(a)
        return clock.center(np.concatenate((a,s,s*abs(s),s*s,s*np.tanh(a[:,:,0:1]*a[:,:,1:2])),axis=2))
    return [lift(a) for a in outs], error


def record(d,basis,cov):
    raw=clock.book.rawday(d);t=np.arange(d%7,len(raw)-60,7)
    y=clock.labels(d).reshape(-1,50,3)[t]
    v=clock.center(y[:,:,2]-clock.RATIO*y[:,:,1]).ravel()
    out=[]
    for z in designs(raw,t,basis,cov)[0]:
        a=z.reshape(len(v),-1).astype(float)
        out.append((a.T@a/len(a),a.T@v[:,None]/len(a)))
    return out


def change(raw,t,basis,cov,w):
    views,err=designs(raw,t,basis,cov)
    a,b,c=((v@m)[:,:,0] for v,m in zip(views,w))
    return [.30*clock.center(z) for z in (b-a,c-b)],err


def screen():
    basis,cov,eigen=geometry();hist=[];folds=[];daily=[];errors=[]
    for end in (8,10,12,14):
        hist.extend(record(d,basis,cov) for d in range(len(hist),end))
        w=transport.fit(hist);groups=[[[],[]],[[],[]]]
        for d in (end,end+1):
            raw=clock.book.rawday(d);t=np.arange(d%3,len(raw),3)
            y=clock.labels(d).reshape(-1,50,3)[t]
            zz,err=change(raw,t,basis,cov,w);errors.append(np.mean(err,0).tolist())
            if d==end:
                mask=t<347
                for a,b in zip(zz,change(raw[:347],t[mask],basis,cov,w)[0]):
                    np.testing.assert_allclose(a[mask],b,rtol=0,atol=5e-7)
            for k,ref in enumerate((clock.incumbent.fi.peer.reference,clock.incumbent.parent.fixed.earlier.early.baseline)):
                p=ref(d).reshape(-1,50,3)[t]
                for g,z in zip(groups[k],zz):g.append((p,y,z))
            daily.append([diagnostic.gain(g[-1:]) for g in groups[0]])
        folds.append([[diagnostic.gain(g) for g in row] for row in groups]);print('window',end+1,folds[-1][0],flush=True)
    keep=[min(row[j]['y3'] for fold in folds for row in fold)>0 and sum(v[j]['y3']>0 for v in daily)>=5
          and np.mean([fold[0][j]['relative_far'] for fold in folds])>0 for j in range(2)]
    keep=np.array(keep)&(np.diff(np.mean(errors,0))<0)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=daily,eigen=eigen.tolist(),
                                        covariance_error=errors,keep=keep.tolist()),indent=2))
    if keep.any():
        hist.extend(record(d,basis,cov) for d in (14,15))
        np.savez(OUT/'model.npz',basis=basis,cov=cov,w=transport.fit(hist),keep=keep)
    return keep.any()


def layer():
    parent=incumbent.layer()
    with np.load(OUT/'model.npz') as s:basis,cov,w,keep=[s[k] for k in ('basis','cov','w','keep')]
    def apply(raw,p):
        q=parent(raw,p)
        for t in np.array_split(np.arange(len(q)),min(4,len(q))):
            for z,yes in zip(change(raw,t,basis,cov,w)[0],keep):
                if yes:q[t,:,2]+=z
        return q
    return apply


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if screen():replay.verify(OUT,layer,incumbent.OUT)
