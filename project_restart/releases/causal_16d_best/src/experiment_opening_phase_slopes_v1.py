"""Fit time-varying book slopes for the twenty-tick cycle, including a phase-shifted control."""
import json
import numpy as np
import experiment_opening_clock_strength_v1 as opening

clock,current,prior=opening.clock,opening.current,opening.prior
OUT=clock.ART/'opening_phase_slopes_v1'


def record(d,shift):
    raw=clock.book.rawday(d);t=np.arange(d%3,len(raw)-60,3)
    y=clock.labels(d).reshape(-1,50,3)[t].astype(float)
    market=y.mean(1);y-=market[:,None]
    cycle=clock.phase(t,shift)
    gate=opening.basis(t).ravel()
    result=[]
    for x,v in zip(clock.features(raw,t),(market,y)):
        parts=[]
        for k in range(20):
            ok=cycle==k
            a=x[ok].reshape(-1,x.shape[-1]).astype(float)
            b=v[ok].reshape(-1,3)
            kappa=np.repeat(gate[ok],len(a)//int(ok.sum()))
            moments=[]
            for wt in (kappa,1-kappa):
                moments.append(((a*wt[:,None]).T@a,(a*wt[:,None]).T@b,wt.sum()))
            parts.append(moments)
        result.append(tuple(np.array([[r[j] for r in p] for p in parts]) for j in (0,1,2)))
    return result


def fit(hist):
    result=[]
    for rows in zip(*hist):
        gram,rhs,n=(np.sum(c,axis=0) for c in zip(*rows))
        def smooth(x):
            return sum(np.roll(x,j,axis=0)*wt for j,wt in ((-2,.1),(-1,.2),(0,.4),(1,.2),(2,.1)))
        gram,rhs,n=map(smooth,(gram,rhs,n))
        reg=.65*np.diag((gram.sum((0,1))/n.sum()).diagonal().clip(.001))+.001*np.eye(gram.shape[-1])
        g=gram/n[:,:,None,None];b=rhs/n[:,:,None,None]
        # Remove the average early/late slope before estimating phase-specific changes.
        avg=np.linalg.solve(gram.sum(0)/n.sum(0)[:,None,None]+reg,rhs.sum(0)/n.sum(0)[:,None,None])
        phase_weights=np.linalg.solve(g+g.mean(0)+reg,b-g@avg)
        f=phase_weights[:,0]-phase_weights[:,1]
        result.append(f-f.mean(0))
    return result


def changes(raw,t,shift,w,offset=0):
    k=(clock.phase(t,shift)+offset)%20
    m,r=clock.features(raw,t)
    gate=.4*(opening.basis(t)-.10)
    return gate*np.einsum('ti,tij->tj',m,w[0][k])[:,None],gate*clock.center(r@w[1][k])


def screen():
    shift,_=clock.clock_geometry();h=[];h20=[];cal=[];folds=[];daily=[];placebo=[]
    for end in (8,10,12,14):
        h.extend(record(d,shift) for d in range(len(h),end))
        h20.extend(clock.statistics(d,shift) for d in range(len(h20),end))
        w=fit(h);w20=clock.weights(h20)
        amp=prior.coefficients(cal) if cal else np.ones(3);amp[2]=1
        groups=[[[],[]],[[],[]]];wrong=[[],[]]
        for d in (end,end+1):
            raw=clock.book.rawday(d);t=np.arange(d%3,len(raw),3);y=clock.labels(d).reshape(-1,50,3)[t]
            base=prior.corrections(clock.changes(raw,t,shift,w20))
            z=changes(raw,t,shift,w);fake=changes(raw,t,shift,w,5)
            if d==end:
                ok=t<311
                for a,b in zip(z,changes(raw[:311],t[ok],shift,w)):np.testing.assert_allclose(a[ok],b,atol=2.e-7,rtol=0)
            for j,ref in enumerate((clock.incumbent.fi.peer.reference,clock.incumbent.parent.fixed.earlier.early.baseline)):
                p=ref(d).reshape(-1,50,3)[t]
                if not j:
                    tr=t<len(raw)-60;cal.append(prior.moments(p[tr],y[tr],base[tr]))
                p=(p+base*amp).astype('float32')
                for i in (0,1):
                    groups[j][i].append((p,y,np.broadcast_to(z[i],y.shape)))
                    if not j:wrong[i].append((p,y,np.broadcast_to(fake[i],y.shape)))
            daily.append([clock.deltas(row[-1:]) for row in groups[0]])
        folds.append([[clock.deltas(row) for row in grp] for grp in groups])
        placebo.append([clock.deltas(row) for row in wrong]);print(end+1,folds[-1][0],flush=True)
    f=np.array(folds)
    keep=(f.min((0,1))>0)&((f[:,0]-np.array(placebo)).mean(0)>0)&((np.array(daily)>0).sum(0)>=5)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,shifted=placebo,daily=daily,keep=keep.tolist()),indent=2))
    if keep.any():
        h.extend(record(d,shift) for d in (14,15))
        m,r=fit(h);np.savez(OUT/'model.npz',m=m,r=r,shift=shift,keep=keep)
    return keep.any()


def layer():
    base=current.layer()
    with np.load(OUT/'model.npz') as f:w=f['m'],f['r'];shift=int(f['shift']);keep=f['keep']
    def apply(raw,p):
        q=base(raw,p)
        for z,k in zip(changes(raw,np.arange(len(q)),shift,w),keep):q+=z*k
        return q
    return apply


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if screen():opening.replay.verify(OUT,layer,current.OUT)
