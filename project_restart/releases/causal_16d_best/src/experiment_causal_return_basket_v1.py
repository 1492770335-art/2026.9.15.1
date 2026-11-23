"""Fit causal multiscale price and forecast corrections with an estimated return basket."""
import json
import numpy as np
import cache_opening_walkforward as cache

prior = cache.opening.prior
OUT = prior.ART/'causal_return_basket_v1'
HORIZONS = (5,20,60)


def grid(day):
    return np.load(prior.CACHE/f'day_{day:02d}_x.npy',mmap_mode='r')


def moving_sum(x,width):
    z=np.concatenate((np.zeros_like(x[:1]),x.cumsum(0,dtype=np.float64)))
    end=np.arange(1,len(x)+1)
    return z[end]-z[np.maximum(0,end-width)]


def basket(train_end):
    cov=np.zeros((50,50))
    for d in range(train_end):
        r=grid(d)[::1,:,0].astype(float)
        block=moving_sum(r,20)[19::20]
        cov+=np.cov(block.T)
    scale=np.sqrt(np.diag(cov))
    corr=cov/np.outer(scale,scale)
    np.fill_diagonal(corr,0)
    corr=np.maximum(corr,0)**2
    return corr/np.maximum(1e-8,corr.sum(1,keepdims=True))


def features(x,p,w):
    ticks=len(x); t=np.arange(ticks)
    price=np.asarray(x[:,:,0],dtype=float)
    means=[]; relative=[]
    def add(v):
        if v.ndim==2:
            v=v[:,:,None]
        m=v.mean(1,keepdims=True)
        means.append(m[:,0])
        relative.append(v-m)
    add(p)
    for h in (4,16,60,240):
        a=moving_sum(p,h)/np.minimum(t+1,h)[:,None,None]
        add(a)
    rms=np.sqrt((4.+moving_sum(price**2,128))/(4.+np.minimum(128,t+1))[:,None])
    scale=(.35+.65*rms).clip(.45,4.)
    stale=np.where(np.abs(price)>.01,t[:,None],-1)
    stale=np.maximum.accumulate(stale,axis=0)
    age=(t[:,None]-stale).clip(0,100)
    cum=price.cumsum(0)
    for h in (1,3,5,12,20,40,60,120,256):
        r=moving_sum(price,h)/np.sqrt(h)
        peer=r@w.T
        add(np.stack((r,peer-r,r/scale,(peer-r)*np.minimum(age/8,1)),axis=2))
    ref=cum@w.T
    row=np.maximum(stale,0)
    drift=(ref-ref[row,np.arange(50)])/np.sqrt(1.+age)
    add(np.stack((drift,drift/scale),axis=2))
    for j,h in enumerate(HORIZONS):
        # Only already-observed returns enter the error of a matured old forecast.
        realised=moving_sum(price,h)*(.559,.272,.161)[j]
        err=realised-p[np.maximum(t-h,0),:,j]
        err[:h]=0
        for n in (1,16,64):
            add(moving_sum(err,n)/np.minimum(n,t+1)[:,None])
    fields=(0,5,7,11,16,17,18,24,25,26,27,40,41,42,52,60,61,65,76,82,84)
    u=x[:,:,fields].astype(float)
    add(np.tanh(u))
    add(np.tanh(u)*scale[:,:,None])
    add(p*scale[:,:,None])
    add(p*np.exp(-t/256)[:,None,None])
    common=np.column_stack(means)
    rel=np.concatenate(relative,axis=-1)
    return common,rel


def moments(x,y):
    return x.T@x,x.T@y,len(x)


def fit(history,ridge):
    g,b,n=map(sum,zip(*history))
    diag=np.maximum(np.diag(g),n*1e-7)
    return np.linalg.solve(g+ridge*np.diag(diag),b)


def main():
    OUT.mkdir(exist_ok=True)
    plan={'features':'causal past returns on 1-256 frames, trained 20-tick cross-stock return baskets, stale-price peer drift, matured forecast errors at 5/20/60 frames, causal volatility and cross-sectional inputs.',
          'training':'9-12 -> 13-14, 9-14 -> 15-16. Basket matrix from all dates before validation. Separate ridge fits on common and relative OOF forecast errors. Select strength per target and component from 0,.25,.5,1 using pooled early validation gains. Refit 9-16 and evaluate 17-20.'}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    early=[]; report={}
    for end,stop in ((12,14),(14,16),(16,20)):
        w=basket(end); cm=[]; cr=[]
        for d in range(8,end):
            p=np.asarray(cache.load(d),dtype=float); y=prior.labels(d).reshape(p.shape)
            m,r=features(grid(d),p,w); err=y-p
            avg=err.mean(1)
            cm.append(moments(m,avg))
            cr.append(moments(r[::3].reshape(-1,r.shape[-1]),(err-avg[:,None,:])[::3].reshape(-1,3)))
        am=fit(cm,.25); ar=fit(cr,.1)
        rows=[]
        for d in range(end,stop):
            p=np.asarray(cache.load(d),dtype=float);y=prior.labels(d).reshape(p.shape)
            m,r=features(grid(d),p,w)
            if d==end:
                ms,rs=features(grid(d)[:313],p[:313],w)
                np.testing.assert_allclose(m[:313],ms,atol=1e-8)
                np.testing.assert_allclose(r[:313],rs,atol=1e-8)
            q=m@am;v=r@ar
            np.testing.assert_allclose(v.mean(1),0,atol=2e-8)
            rows.append((p,y,q,v))
        if end<16:
            early.extend(rows)
        data=early if end<16 else None
        if end<16:
            summary={}
            p,y,q,v=map(np.concatenate,zip(*early))
            best=np.full(3,-np.inf);weights=np.zeros((2,3))
            for a in [0,.25,.5,1.]:
                for b in [0,.25,.5,1.]:
                    cor=prior.corr((p+a*q[:,None,:]+b*v).reshape(-1,3),y.reshape(-1,3))
                    improve=cor>best
                    weights[:,improve]=np.array([[a],[b]])
                    best=np.maximum(best,cor)
            print('through early day',stop,'strengths',weights.tolist(),'pooled corr',best,flush=True)
        else:
            np.savez(OUT/'model.npz',basket=w,market=am,relative=ar,strength=weights)
        p,y,q,v=map(np.concatenate,zip(*rows))
        pred=p+q[:,None,:]*weights[0]+v*weights[1]
        assessment=prior.measure([(p.reshape(-1,3),pred.reshape(-1,3),y.reshape(-1,3))])
        report[str(stop)]={'weights':weights.tolist(),'result':assessment}
        if end==16:
            np.testing.assert_allclose(assessment['baseline_score'],.33553985,atol=1e-7)
            for d,(p,y,q,v) in zip(range(end,stop),rows):
                np.save(OUT/f'prediction_{d}.npy',(p+q[:,None,:]*weights[0]+v*weights[1]).astype(np.float32))
        print('end',stop,assessment,flush=True)
        (OUT/'results.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
