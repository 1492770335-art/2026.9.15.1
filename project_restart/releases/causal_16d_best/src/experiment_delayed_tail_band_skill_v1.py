"""Calibrate the late return band using only already observed subintervals of earlier forecasts."""
import json
import numpy as np
from experiment_causal_cross_horizon_ridge_v1 import ART,CACHE,corr,labels,measure
import experiment_eight_date_expert_contrasts_v1 as earlier
import experiment_causal_return_commonality_v1 as clock

OUT=ART/'delayed_tail_band_skill_v1'
SCALE=np.array([.559,.272,.161])
BANDS=[(0,5,0),(5,12,1),(12,20,1),(5,20,1),(20,32,2),(32,46,2),(46,60,2),(20,60,2)]


def base(d):
    if d>=14:return np.load(ART/'pairwise_book_tree_v1'/f'pred_{d}.npy')
    return earlier.baseline(d)


def features(raw,p):
    r=raw[:,:,0].astype(float)-.00136
    price=np.vstack((np.zeros((1,50)),r.cumsum(0)))
    full=p/SCALE
    band=np.concatenate((full[:,:,:1],np.diff(full,axis=2)),axis=2)
    t=np.arange(len(p));cols=[[],[]]
    avg=band.mean(1,keepdims=True)
    for lo,hi,j in BANDS:
        ix=(t-hi).clip(0)
        # At time t this interval of the vintage forecast ends exactly at the current tick.
        actual=price[t+1]-price[np.maximum(0,t+1-hi+lo)]
        actual[:hi]=0
        mc=actual.mean(1,keepdims=True)
        fraction=(hi-lo)/(5,15,40)[j]
        for k,(observed,origin,all_p) in enumerate(((mc,avg[ix,:,j],avg),
                                              (actual-mc,(band-avg)[ix,:,j],band-avg))):
            origin=origin.copy()*fraction
            origin[:hi]=0.
            cov=(origin*observed).mean(1)
            var=(origin*origin).mean(1)
            energy=(observed*observed).mean(1)
            for width in (320,1400):
                norm=clock.trailing(var,width)
                signal=clock.trailing(cov,width)
                noise=clock.trailing(energy,width)
                # Shrink when a short realized segment has little cumulative signal.
                count=np.minimum((t-hi+1).clip(0),width)
                ridge=210 if k else 560
                prior=(norm+.002*(hi-lo)*24)/(count+24)
                varfloor=.003*(hi-lo)*fraction
                gain=(signal-norm)/(norm+ridge*np.maximum(prior,varfloor))
                ic=signal/np.sqrt((noise+ridge*.12*(hi-lo))*(norm+ridge*np.maximum(prior,varfloor)))
                for a in (np.tanh(gain),np.tanh(4*ic)):
                    v=(all_p[:,:,2]*SCALE[2])*a[:,None]
                    cols[k].append(v.astype('float32'))
                    if j==2:
                        v=(all_p[:,:,1]*SCALE[2])*a[:,None]
                        cols[k].append(v.astype('float32'))
    return np.column_stack(cols[0]),np.stack(cols[1],axis=2)


def fit(hist):
    w=[]
    for alpha,part in zip((4.,2.3),zip(*hist)):
        g,b=(np.mean(v,0) for v in zip(*part))
        reg=alpha*np.diag(g.diagonal().clip(.00002,None))
        daily=np.stack([np.linalg.solve(a+reg,c) for a,c in part])
        w.append((.8*(.55*np.linalg.solve(g+reg,b)+.45*np.median(daily,0))*np.abs(np.sign(daily).mean(0))).clip(-.21,.21))
    return w


def main():
    OUT.mkdir(exist_ok=True)
    hist=[];report={};daily=[]
    for d in range(8,20):
        if d in (10,12,14,16):
            w=fit(hist);rows=[]
            if d==16:
                a=np.stack(daily);ok=((a>0).sum(0)>=4)&(a.mean(0)>0)&(a[-2:].mean(0)>0)
                report['eligible']=ok.tolist()
                if not ok.any():break
                w=[c*g for c,g in zip(w,ok)]
                np.savez(OUT/'model.npz',market=w[0],relative=w[1])
        p=base(d);raw=clock.rawday(d);m,r=features(raw,p)
        if d>=10:
            a=(m@w[0])[:,None];b=r@w[1];q=p.copy()
            q[:,:,2]+=a+b
            if d in (10,16):
                mh,rh=features(raw[:317],p[:317])
                np.testing.assert_allclose(q[:317,:,2],p[:317,:,2]+(mh@w[0])[:,None]+rh@w[1],atol=3e-6)
            if d>=16:np.save(OUT/f'pred_{d}.npy',q)
        y=labels(d).reshape(p.shape)
        if d<16:
            t=np.arange(d%3,len(p),3)
            stats=earlier.moments(p[t],y[t],(m[t],r[t]))
            hist.append([(g,h[:,2]) for g,h in stats])
        if d>=10:
            row=tuple(v.reshape(-1,3) for v in (p,q,y));rows.append(row)
            if d<16:
                py=p[:,:,2].reshape(-1,1);yy=y[:,:,2].reshape(-1,1)
                old=corr(py,yy)[0]
                daily.append([corr((p[:,:,2]+term).reshape(-1,1),yy)[0]-old for term in (a,b)])
        if d in (11,13,15,19):
            report[str(d+1)]=measure(rows);print(d+1,report[str(d+1)],flush=True)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()

