"""Suppress strongly date-dependent field offsets using train-estimated variance and only the day's observed prefix."""
import json
import numpy as np
import torch
import experiment_phase_future_book_v1 as phase
import experiment_cross_horizon_rank_readout_v1 as rank
import experiment_variance_ratio_calibration_v1 as older

OUT=phase.inn.prior.ART/'causal_price_level_centering_v1'

def geometry():
    means=[];variances=[]
    for d in range(8):
        a=np.arcsinh(phase.inn.grid(d)[::13,:,:83].astype(np.float32)).clip(-4,4)
        means.append(a.mean(0));variances.append(a.var(0))
    b=np.var(means,axis=0);w=np.mean(variances,axis=0)
    icc=b/(b+w+.00001)
    fields=np.where(np.median(icc,axis=0)>.32)[0];fields=fields[fields!=0]
    return dict(fields=fields,center=np.mean(means,axis=0)[:,fields],
                strength=(b/(b+.15*w+1e-5))[:,fields].clip(0,.96),
                pseudo=(30*w/(b+.0002))[:,fields].clip(4,160))

def transform(raw,geom):
    ids=geom['fields'];a=np.arcsinh(raw[:,:,ids].astype(np.float32))
    target=geom['center'];n=np.arange(1,len(raw)+1)[:,None,None]
    offset=(a-target).cumsum(0,dtype=float)/(n+geom['pseudo'])
    corrected=np.sinh((a-offset*geom['strength']).clip(-7,7))
    result=raw.copy();result[:,:,ids]=corrected.astype(result.dtype)
    return result

def original(d):
    return np.load(rank.OUT/f'pred_{d}.npy') if d>=14 else older.baseline(d)

def basis(p,n):
    return (n-p).mean(1).astype('float32'),rank.features(p,[n])

def moments(p,y,features):
    m,r=features;pp=p.reshape(-1,3).astype(float);pp-=pp.mean(0)
    yy=y.reshape(-1,3).astype(float);yy-=yy.mean(0)
    slope=((pp*yy).sum(0)/(pp*pp).sum(0)).clip(.4,2.5)
    e=y/slope-p;em=e.mean(1)
    m=m.astype(float);r=r.reshape(-1,3).astype(float)
    return ((m.T@m/len(m),m.T@em/len(m)),(r.T@r/len(r),r.T@(e-em[:,None]).reshape(-1,3)/len(r)))

def fit(hist):
    coef=[]
    for part in zip(*hist):
        g,b=(np.mean(z,0) for z in zip(*part))
        reg=1.35*np.diag(g.diagonal().clip(2e-5,None))
        daily=np.stack([np.linalg.solve(a+reg,c) for a,c in part])
        beta=.55*np.linalg.solve(g+reg,b)+.45*np.median(daily,axis=0)
        coef.append((.65*beta*abs(np.sign(daily).mean(0))).clip(-.18,.2))
    return np.stack(coef)

def main():
    OUT.mkdir(exist_ok=True);geom=geometry();np.savez(OUT/'input.npz',**geom)
    print('train-derived offset fields',geom['fields'],flush=True)
    def features(raw,t):return phase.spline.features(transform(raw,geom),t)
    hist=[];report={};mask=np.ones((2,3),bool)
    for end,stop in ((12,14),(14,16),(16,20)):
        ck=OUT/f'net_{end}.pth'
        if ck.exists():
            state=torch.load(ck,weights_only=True,map_location='cpu')
            net=phase.BookNet(state['current.0.weight'].shape[1]).cuda().eval();net.load_state_dict(state)
        else:net=phase.train(end,seed=281,out=OUT,make_features=features)
        if end>=14:
            coef=fit(hist)*mask[:,None,:]
            if end==16:np.save(OUT/'coef.npy',coef)
        rows=[];daily=[]
        for d in range(end,stop):
            raw=phase.inn.grid(d);n=phase.infer(net,raw,make_features=features)
            if d==end:np.testing.assert_allclose(n[:165],phase.infer(net,raw[:165],make_features=features),atol=1e-4)
            p=original(d);y=phase.inn.prior.labels(d).reshape(p.shape);z=basis(p,n)
            if end<16:hist.append(moments(p,y,z))
            np.save(OUT/f'side_{d}.npy',n)
            if end>=14:
                terms=[(z[0]@coef[0])[:,None],z[1]@coef[1]]
                q=(p+sum(terms)).astype('float32');np.save(OUT/f'pred_{d}.npy',q)
                rows.append(tuple(a.reshape(-1,3) for a in (p,q,y)))
                if end==14:
                    old=phase.inn.prior.corr(p.reshape(-1,3),y.reshape(-1,3))
                    daily.append([phase.inn.prior.corr((p+a).reshape(-1,3),y.reshape(-1,3))-old for a in terms])
        if end>=14:
            val=phase.inn.prior.measure(rows);report[str(stop)]=val
            (OUT/'results.json').write_text(json.dumps(report,indent=2));print(stop,val,flush=True)
            if end==14:
                mask=np.min(daily,axis=0)>0
                print('held-date component gate',mask,flush=True)
                if not mask.any():break
if __name__=='__main__':main()
