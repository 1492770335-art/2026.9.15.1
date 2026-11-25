"""Estimate how the strength of a twenty-tick cycle response changes from the opening."""
import json
import numpy as np
import experiment_phase_conditioned_forecasts_v1 as current
import replay_periodic_parent_v1 as replay

prior, clock = current.prior, current.clock
OUT = clock.ART/'opening_clock_strength_v1'


def basis(t):
    return np.exp(-t/300.)[:,None,None]


def statistics(p,y,z,t):
    x = np.stack((z, z*basis(t)),axis=-1).reshape(-1,3,2).astype(float)
    a,b=(v.reshape(-1,3).astype(float) for v in (p,y))
    x-=x.mean(0); a-=a.mean(0); b-=b.mean(0)
    variance=(a*a).mean(0).clip(1.e-7)
    scale=((a*b).mean(0)/variance).clip(.4,2.5)
    x-=a[:,:,None]*((x*a[:,:,None]).mean(0)/variance[:,None])
    x=x.transpose(1,0,2)
    return x.transpose(0,2,1)@x/len(a), (x.transpose(0,2,1)@(b/scale).T[:,:,None])[:,:,0]/len(a)


def fit(hist):
    gs,bs=(np.stack(v) for v in zip(*hist))
    g,b=gs.mean(0),bs.mean(0)
    # Orthogonalize the opening-specific adjustment against the constant one.
    shift=g[:,0,1]/g[:,0,0].clip(1.e-10)
    contrast=np.stack((-shift,np.ones(3)),axis=-1)
    variance=np.einsum('ki,dkij,kj->dk',contrast,gs,contrast).clip(1.e-11)
    slope=np.einsum('ki,dki->dk',contrast,bs)
    coeff=.55*(.65*slope.mean(0)/variance.mean(0)+.35*np.median(slope/variance,axis=0))
    return contrast*coeff.clip(-12,12)[:,None]


def change(z,t,w):
    return z*(w[:,0]+w[:,1]*basis(t))


def screen():
    shift,_=clock.clock_geometry()
    history=[];cal=[];local=[];folds=[];daily=[];trained=[]
    for end in (8,10,12,14):
        history.extend(clock.statistics(d,shift) for d in range(len(history),end))
        w=clock.weights(history)
        amp=prior.coefficients(cal) if cal else np.ones(3)
        amp[2]=1
        extra=fit(local) if local else None
        groups=[[],[]]
        for d in (end,end+1):
            raw=clock.book.rawday(d);t=np.arange(d%3,len(raw),3)
            z=prior.corrections(clock.changes(raw,t,shift,w))
            y=clock.labels(d).reshape(-1,50,3)[t]
            for j,ref in enumerate((clock.incumbent.fi.peer.reference,clock.incumbent.parent.fixed.earlier.early.baseline)):
                p=ref(d).reshape(-1,50,3)[t]
                if not j:
                    ok=t<len(raw)-60
                    cal.append(prior.moments(p[ok],y[ok],z[ok]))
                    local.append(statistics(p[ok],y[ok],z[ok],t[ok]))
                if extra is not None:groups[j].append((p+z*amp,y,change(z,t,extra)))
            if extra is not None:daily.append(clock.deltas(groups[0][-1:]))
        if extra is not None:
            folds.append([clock.deltas(rows) for rows in groups]);trained.append(extra.tolist())
            print('window',end+1,folds[-1],flush=True)
    keep=(np.array(folds).min((0,1))>0)&((np.array(daily)>0).sum(0)>=4)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=daily,coefficients=trained,retained=keep.tolist()),indent=2))
    if keep.any():np.save(OUT/'weights.npy',fit(local)*keep[:,None])
    return keep.any()


def layer():
    parent=current.layer()
    with np.load(clock.OUT/'model.npz') as f:w=f['m'],f['r'];shift=int(f['shift'])
    coeff=np.load(OUT/'weights.npy')
    def apply(raw,p):
        q=parent(raw,p)
        t=np.arange(len(q));z=prior.corrections(clock.changes(raw,t,shift,w))
        return (q+change(z,t,coeff)).astype(q.dtype)
    return apply


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if screen():replay.verify(OUT,layer,current.OUT)
