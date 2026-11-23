"""Scale common and relative forecasts with observed multihorizon return commonality."""
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from experiment_causal_cross_horizon_ridge_v1 import ART,CACHE,corr,labels,measure
import experiment_eight_date_expert_contrasts_v1 as utility

OUT=ART/'causal_return_commonality_v1'
WIDTHS=(5,20,60)
WINDOWS=(256,1152)


def trailing(x,w):
    c=np.concatenate([np.zeros_like(x[:1]),x.cumsum(0,dtype=np.float64)])
    end=np.arange(1,len(c))
    return c[end]-c[(end-w).clip(0)]


def rawday(d):
    return np.load(CACHE/f'day_{d:02d}_x.npy',mmap_mode='r')


def ratios(raw):
    r=raw[:,:,0].astype(float)-.00136
    s=np.sqrt((10+trailing(r*r,120))/(10+np.minimum(np.arange(len(r))+1,120))[:,None]).clip(.4,4.)
    t=np.arange(len(raw))
    columns=[]
    for h in WIDTHS:
        # Scale by volatility at the beginning of the realized return interval.
        z=trailing(r,h)/np.sqrt(h)
        for w in WINDOWS:
            block=[]
            for x in (z,z/s[np.maximum(0,t-h)]):
                m=x.mean(1)
                total=(x*x).mean(1)
                shared=m*m
                # Data before the first complete h-tick interval have zero weight.
                num=trailing(np.where(t>=h,shared,0),w)
                den=trailing(np.where(t>=h,total,0),w)
                share=(num+8*.16)/(den+8)
                block.append(np.log(share.clip(.003,.95)/(1-share.clip(.003,.95))))
            columns.append(.5*(block[0]+block[1]))
    return np.column_stack(columns)


def reference():
    v=[]
    for d in range(8):
        a=ratios(rawday(d))
        t=np.minimum(np.arange(4800),len(a)-1)
        v.append(a[t])
    return gaussian_filter1d(np.median(v,axis=0),35,axis=0,mode='nearest')


def basis(p,ratio,ref):
    t=np.arange(len(p))
    dif=(ratio-ref[t.clip(0,len(ref)-1)]).reshape(len(p),3,2)
    phase=(1-np.exp(-t/130))[:,None]
    z=phase*np.tanh(dif.mean(2))
    term=phase*np.tanh(dif[:,:,0]-dif[:,:,1])
    m=p.mean(1);r=p-m[:,None]
    common=[m*z,m*term,m*z[:,2:3],m*(z[:,2:3]-z[:,:1])]
    common.extend([(m[:,1:]-m[:,:1])*z[:,2:3], np.abs(m)*z[:,2:3]*np.tanh(4*m)])
    local=[r*z[:,None],r*term[:,None],r*(z[:,2:3]-z[:,:1])[:,None]]
    return np.column_stack(common).astype('float32'),np.concatenate(local,axis=2).astype('float32')


def main():
    OUT.mkdir(exist_ok=True)
    ref=reference()
    hist=[];report={};daily=[]
    for d in range(8,20):
        if d in (10,12,14,16):
            w=utility.fit(hist);rows=[]
            if d==16:
                a=np.array(daily)
                good=(a.reshape(3,2,2,3).mean(1)>0).sum(0)>=2
                good&=(a.mean(0)>0)&(a[-2:].mean(0)>0)&((a>0).sum(0)>=4)
                report['eligible']=good.tolist()
                if not good.any():break
                w=[b*g for b,g in zip(w,good)]
                np.savez(OUT/'model.npz',market=w[0],relative=w[1],reference=ref)
        raw=rawday(d);p=utility.baseline(d)
        m,r=basis(p,ratios(raw),ref)
        if d>=10:
            a=(m@w[0])[:,None];b=r@w[1];q=(p+a+b).astype('float32')
            if d in (10,16):
                mh,rh=basis(p[:237],ratios(raw[:237]),ref)
                np.testing.assert_allclose(q[:237],p[:237]+(mh@w[0])[:,None]+rh@w[1],atol=2e-6)
            if d>=16:np.save(OUT/f'pred_{d}.npy',q)
        y=labels(d).reshape(p.shape)
        if 10<=d<16:
            old=corr(p.reshape(-1,3),y.reshape(-1,3))
            daily.append([corr((p+v).reshape(-1,3),y.reshape(-1,3))-old for v in (a,b)])
        if d<16:hist.append(utility.moments(p,y,(m,r)))
        if d>=10:rows.append(tuple(v.reshape(-1,3) for v in (p,q,y)))
        if d in (11,13,15,19):
            report[str(d+1)]=measure(rows)
            print(d+1,report[str(d+1)],flush=True)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()

