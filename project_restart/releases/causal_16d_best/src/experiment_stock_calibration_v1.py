"""Calibrate cross-sectional slopes and market loadings using previous out-of-fold days."""
import json
import numpy as np
import cache_opening_walkforward as cache
import experiment_factor_exposure_return_v1 as ex
import experiment_separate_market_relative_v1 as best
prior=cache.opening.prior
OUT=prior.ART/'stock_calibration_v1'


def forecasts(d):
    if d<12:return np.asarray(cache.load(d),dtype=float)
    return np.load(best.OUT/f'pred_{d}.npy').astype(float)


def covariates(p,raw):
    m=p.mean(1,keepdims=True);rel=p-m
    rv=raw[:,:,0].astype(float)**2
    s=np.vstack((np.zeros(50),rv.cumsum(0)))
    t=np.arange(len(p))+1;left=np.maximum(t-96,0)
    vol=np.sqrt((s[t]-s[left]+4)/((t-left+4)[:,None])).clip(.4,3.5)
    return np.concatenate((rel,np.broadcast_to(m,p.shape),
        rel*np.log(vol)[:,:,None],np.broadcast_to(m,p.shape)*np.log(vol)[:,:,None]),axis=2)


def fit(end):
    gm=[];bm=[]
    for d in range(8,end):
        p=forecasts(d);y=prior.labels(d).reshape(p.shape);e=y-p
        e=e-e.mean(1,keepdims=True)
        z=np.ascontiguousarray(covariates(p,ex.grid(d)).transpose(1,0,2))
        yy=e.transpose(1,0,2)
        gm.append((z.transpose(0,2,1)@z)/len(p));bm.append((z.transpose(0,2,1)@yy)/len(p))
    g=np.mean(gm,axis=0);b=np.mean(bm,axis=0);shared_g=g.mean(0);shared_b=b.mean(0)
    penalty=np.diag(np.maximum(shared_g.diagonal(),1e-7))
    shared=np.linalg.solve(shared_g+.7*penalty,shared_b)
    coefs=[]
    for i,(a,t) in enumerate(zip(g,b)):
        strong=a+1.25*penalty
        # Shrink toward shared response and toward the median day response.
        joint=shared+np.linalg.solve(strong,t-a@shared)
        daily=[shared+np.linalg.solve(gg[i]+1.25*penalty,bb[i]-gg[i]@shared) for gg,bb in zip(gm,bm)]
        coefs.append(.65*joint+.35*np.median(daily,axis=0))
    return np.stack(coefs)


def true_beta(end):
    ratios=[]
    for d in range(end):
        y=prior.labels(d).reshape(-1,50,3).astype(float);y-=y.mean(0)
        m=y.mean(1,keepdims=True)
        ratios.append((y*m).mean(0)/(m*m).mean(0))
    return .5*(np.mean(ratios,0)+np.median(ratios,0))


def forecast_beta(end):
    betas=[]
    for d in range(8,end):
        p=forecasts(d);p=p-p.mean(0)
        m=p.mean(1,keepdims=True)
        betas.append((p*m).mean(0)/(m*m).mean(0))
    return np.median(betas,0)


def main():
    OUT.mkdir(exist_ok=True);early=[];report={}
    for end,stop in [(12,14),(14,16),(16,20)]:
        a=fit(end);diff=true_beta(end)-forecast_beta(end)
        diff-=diff.mean(0)
        rows=[]
        for d in range(end,stop):
            p=forecasts(d);y=prior.labels(d).reshape(p.shape)
            h=np.einsum('tsf,sfk->tsk',covariates(p,ex.grid(d)),a,optimize=True)
            h-=h.mean(1,keepdims=True)
            g=diff[None]*p.mean(1,keepdims=True)
            rows.append((p,y,h,g))
        if end<16:
            early.extend(rows);p,y,h,g=map(np.concatenate,zip(*early))
            peak=np.full(3,-np.inf);w=np.zeros((2,3))
            for i in [0,.25,.5,.75,1.]:
                for j in [0,.15,.35,.6]:
                    score=prior.corr((p+i*h+j*g).reshape(-1,3),y.reshape(-1,3))-.00025*((i-.5)**2+(j-.15)**2)
                    ix=score>peak;peak=np.maximum(score,peak);w[:,ix]=[[i],[j]]
        else:np.savez(OUT/'model.npz',coef=a,beta_diff=diff,strength=w)
        entries=[]
        for d,(p,y,h,g) in zip(range(end,stop),rows):
            q=p+h*w[0]+g*w[1]
            np.testing.assert_allclose(q.mean(1),p.mean(1),atol=1e-8)
            np.save(OUT/f'pred_{d}.npy',q.astype(np.float32))
            entries.append((p.reshape(-1,3),q.reshape(-1,3),y.reshape(-1,3)))
        report[stop]={'strength':w.tolist(),'result':prior.measure(entries)}
        (OUT/'results.json').write_text(json.dumps(report,indent=2));print('score',stop,report[stop],flush=True)


if __name__=='__main__':main()
