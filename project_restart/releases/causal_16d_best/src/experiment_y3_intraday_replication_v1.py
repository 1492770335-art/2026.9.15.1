"""Adapt a stock's synthetic peer basket using a historical fit block and a separate tracking block."""
import json
import sys

import numpy as np

import y3_donor_migration_references as history

parent,base = history.incumbent,history.base
OUT = parent.OUT.parent/'y3_intraday_replication_v1'
ARMS = ('unchanged','static tracking portfolio','reassigned tracking portfolios','adaptive portfolio',
        'adaptive beyond static','validated portfolio change','failed tracking change','validated price and forecast change',
        'two-horizon tracking agreement','adaptive change versus reassigned')
WIDTH = 5


def returns(raw):
    return np.clip(raw[:,:,0].astype(float)-.00136,-15,15)


def covariance(r,h):
    n = len(r)//h
    if n<2:
        return np.eye(50)
    x = r[-n*h:].reshape(n,h,50).sum(1)/np.sqrt(h)
    return x.T@x/n


def solve(cov,anchor,penalty):
    # Accelerated projected gradient for 50 separate nonnegative, unit-sum tracking portfolios.
    gram = cov+np.eye(50)*penalty
    step = 1/max(np.linalg.eigvalsh(gram)[-1],1e-6)
    rhs = cov+penalty*anchor
    w,v = anchor.copy(),anchor.copy()
    for i in range(32):
        proposal = v-step*(v@gram-rhs)
        np.fill_diagonal(proposal,-1e6)
        ordered = -np.sort(-proposal,axis=1)
        level = (ordered.cumsum(1)-1)/np.arange(1,51)
        threshold = level[np.arange(50),(ordered>level).sum(1)-1]
        new = (proposal-threshold[:,None]).clip(0)
        v = new+(i/(i+3))*(new-w)
        w = new
    return w


def prior():
    mats = []
    for d in range(4):
        raw = history.trial.clock.book.rawday(d)
        for lo,hi in base.segments(len(raw)):
            r = returns(raw[lo:hi])
            mats.append(.5*(covariance(r,8)+covariance(r,24)))
    g = np.mean(mats,0)
    anchor = (1-np.eye(50))/49
    return solve(g,anchor,.55*np.mean(g.diagonal()))


def reliability(w,initial,c):
    eye = np.eye(50)
    def risk(mat):
        return np.sum(((eye-mat)@c)*(eye-mat),axis=1)
    old,new = risk(initial),risk(w)
    return ((old-new)/(old+.02)).clip(-.5,.5)


def stages(raw,initial):
    banks = []
    for lo,hi in base.segments(len(raw)):
        r = returns(raw[lo:hi])
        sums = np.vstack((np.zeros((1,50)),r.cumsum(0)))
        ends = np.arange(1,len(r)+1)
        move = [(sums[ends]-sums[(ends-width).clip(0)])/np.sqrt(width) for width in (8,40,100)]
        for start in range(0,len(r),60):
            sl = slice(start,min(len(r),start+60))
            w = initial
            a = b = np.zeros(50)
            if start>=360:
                past = r[max(0,start-864):start-144]
                g = .5*(covariance(past,8)+covariance(past,24))
                w = solve(g,initial,1.8*g.diagonal().mean().clip(.02))
                check = r[start-144:start-48]
                a,b = [reliability(w,initial,covariance(check,width)) for width in (8,24)]
            shifted = np.roll(w,(13,13),axis=(0,1))
            banks.append((slice(lo+sl.start,lo+sl.stop),*[v[sl] for v in move],w,shifted,a,b))
    return banks


def design(p,banks,initial):
    out = np.empty((len(p),50,len(ARMS)-1,WIDTH),np.float32)
    for sl,fast,med,slow,w,wrong,a,b in banks:
        far = p[sl,:,2]-base.RATIO*p[sl,:,1]
        bank = np.stack((fast,med,slow,p[sl,:,2],far),2)
        def spread(mat):
            z = bank-np.einsum('ij,tjf->tif',mat,bank,optimize=True)
            return z-z.mean(1,keepdims=True)
        fixed,dynamic,displaced = [spread(h) for h in (initial,w,wrong)]
        delta = dynamic-fixed
        gated = delta*(np.maximum(a+b,0)*.5)[None,:,None]
        failed = delta*(np.minimum(a+b,0)*.5)[None,:,None]
        signed = np.sign(gated)*np.sqrt(abs(gated)*abs(delta))
        both = delta*np.minimum(a,b).clip(0)[None,:,None]
        blocks = (fixed,displaced,dynamic,delta,gated,failed,signed,both,dynamic-displaced)
        out[sl] = np.stack([z-z.mean(1,keepdims=True) for z in blocks],2)
    return out


def observe(x,p,y):
    target = []
    for k in (0,1):
        yp = p[:,:,2]-k*base.RATIO*p[:,:,1]
        yt = y[:,:,2]-k*base.RATIO*y[:,:,1]
        yp -= yp.mean(1,keepdims=True)
        yt -= yt.mean(1,keepdims=True)
        beta = (yp*yt).sum()/(yp*yp).sum().clip(1e-7)
        target.append(yt/beta.clip(.35,2.8)-yp)
    mat = x.transpose(2,0,1,3).reshape(x.shape[2],-1,WIDTH).astype(float)
    v = np.stack(target,2).reshape(-1,2)
    return mat.transpose(0,2,1)@mat/mat.shape[1],mat.transpose(0,2,1)@v/mat.shape[1]


def fit(rows):
    gs,bs = [np.asarray(v) for v in zip(*rows)]
    g = gs.mean(0)
    b = bs.mean(0)
    b *= abs(b)/(abs(b)+bs.std(0)/np.sqrt(len(bs))+.000002)
    target = np.where(b[:,:,0]*b[:,:,1]>0,np.sign(b[:,:,0])*abs(b).min(2),0.)
    return .3*np.stack([np.linalg.solve(v+1.2*np.diag(v.diagonal().clip(.000002)),u) for v,u in zip(g,target)])


def train():
    OUT.mkdir(exist_ok=True)
    initial = prior()
    rows,folds,daily,track = [],[],[],[]
    for d in range(4,16):
        raw = history.trial.clock.book.rawday(d)
        y = history.trial.clock.labels(d).reshape(-1,50,3)
        ps = history.references(d,raw) if d>=8 else parent.parent.training_references(d,raw)
        bank = stages(raw,initial)
        active = [b for b in bank if np.any(b[-2])]
        track.append(float(np.mean([np.mean((b[-2]+b[-1])/2) for b in active])))
        if d in (8,12):
            w = fit(rows)
            group = []
        obs,pair = [],[]
        for p in ps:
            x = design(p,bank,initial)
            obs.append(observe(x[d%5::5],p[d%5::5],y[d%5::5]))
            if d>=8:
                z = np.einsum('tsgf,gf->tsg',x,w).astype('float32')
                if d==8:
                    n = 618
                    np.testing.assert_allclose(x[:n],design(p[:n],stages(raw[:n],initial),initial),rtol=0,atol=2e-6)
                pair.append(base.records(p,y,z))
        rows.append(tuple(np.mean(v,0) for v in zip(*obs)))
        if d>=8:
            group.append(pair)
            daily.append([base.score(m,np.ones(len(ARMS)-1))[0] for m in pair])
            if d in (11,15):
                folds.append([base.score(m,np.ones(len(ARMS)-1)) for m in np.sum(group,0)])
    f,ds = np.array(folds),np.array(daily)
    ok = [i for i in (3,4,5,7,8,9) if f[:,:,:,i].min()>0 and (ds[:,:,i]>0).sum(0).min()>=5 and
          (f[:,:,0,i].mean(0)>f[:,:,0,1:3].mean(0).max(1)).all()]
    pick = max(ok,key=lambda i:f[:,:,0,i].min()) if ok else 0
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,arms=ARMS,folds=f.tolist(),daily=ds.tolist(),
        mean_tracking_risk_reduction=track,windows=[[9,12],[13,16]]),indent=2))
    print('selected',ARMS[pick],flush=True)
    if pick:
        np.savez(OUT/'model.npz',pick=pick-1,w=fit(rows)[pick-1],initial=initial)
    return bool(pick)


def correction():
    with np.load(OUT/'model.npz') as f:
        initial,w,pick = f['initial'],f['w'],int(f['pick'])
    return lambda raw,p:(design(p,stages(raw,initial),initial)[:,:,pick]@w).astype('float32')


def layer():
    before,update = parent.layer(),correction()
    def apply(raw,initial):
        p = before(raw,initial).copy()
        p[:,:,2] += update(raw,p)
        return p
    return apply


def replay_with():
    update = correction()
    def read(d,raw):
        p,h,n = history.read(d,raw)
        p[:,:,2] += update(raw,p)
        h[:,:,2] += update(raw[:n],h)
        return p,h,n
    return history.stack.verifier,read


if __name__=='__main__' and (sys.argv[1:]==['replay'] or train()):
    base.replay.replay(sys.modules[__name__])
