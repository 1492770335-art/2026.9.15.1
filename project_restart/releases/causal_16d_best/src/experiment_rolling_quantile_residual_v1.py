"""Learn a residual forecast from each field's position in its own trailing distribution."""
import json
import numpy as np
import experiment_endpoint_innovation_v1 as inn
import experiment_relative_residual_tree_v1 as tree
prior=inn.prior
OUT=prior.ART/'rolling_quantile_residual_v1'
FIELDS=[0,1,3,4,5,7,10,11,13,14,16,17,18,20,22,23,24,25,26,29,31,34,36,37,38,39,40,41,42,46,50,51,53,54,55,56,60,64,67,70,72,75,77,79,82,87,90,92,94,97,99]

def features(raw,t,projection):
    x=raw[:,:,FIELDS]; cur=x[t].astype(np.float32)
    parts=[];ms=[]
    for lags in ([1,2,3,4,6,8,11,15,20],[8,14,23,35,48,64,88,128,170],
                 [60,83,128,191,256,384,512,768,1050]):
        count=np.zeros_like(cur); tie=count.copy()
        for lag in lags:
            past=x[np.maximum(t-lag,0)].astype(np.float32)
            count+=(cur>past);count-=(cur<past);tie+=(cur==past)
        rank=count/len(lags);tie/=len(lags)
        parts.extend([rank,rank*(1-tie),rank**3])
        avg=rank.mean(1);ms.extend([avg,(rank**3).mean(1),(rank*(1-tie)).mean(1)])
        ms.append(np.einsum('ks,tsf->tkf',projection,rank,optimize=True).reshape(len(t),-1))
    z=np.concatenate(parts,axis=2);z-=z.mean(1,keepdims=True)
    return np.column_stack(ms),z

def fit(end,proj):
    mh=[];rh=[]
    for d in range(8,end):
        p=previous(d);y=prior.labels(d).reshape(p.shape)
        raw=inn.grid(d);t=np.arange((d*3)%5,len(raw),5)
        m,z=features(raw,t,proj);e=y[t]-p[t];avg=e.mean(1)
        mh.append((m.T@m/len(m),m.T@avg/len(m)))
        z=z.reshape(-1,z.shape[-1]).astype(float);er=(e-avg[:,None]).reshape(-1,3)
        rh.append((z.T@z/len(z),z.T@er/len(z)))
    return inn.fit(mh,1.3),inn.fit(rh,.4)

def previous(d):
    return inn.forecast(d) if d<12 else np.load(tree.OUT/f'pred_{d}.npy').astype(float)

def main():
    import experiment_pooled_price_impact_v1 as impact
    OUT.mkdir(exist_ok=True);early=[];report={}
    for end,stop in [(12,14),(14,16),(16,20)]:
        proj=impact.exposure(end)[0][-3:]
        cm,cr=fit(end,proj);rows=[]
        for d in range(end,stop):
            p=previous(d);raw=inn.grid(d);a=np.empty((len(raw),3));r=np.empty_like(p);t=np.arange(len(p))
            for i in range(8):
                ix=t[i::8];m,z=features(raw,ix,proj);a[ix]=m@cm;r[ix]=z@cr
            if d==end:
                m,z=features(raw[:102],t[:102],proj);np.testing.assert_allclose(a[:102],m@cm,atol=4e-6);np.testing.assert_allclose(r[:102],z@cr,atol=4e-6)
            rows.append((p,prior.labels(d).reshape(p.shape),a,r))
        if end<16:early.extend(rows);weights=inn.strength(early)
        else:np.savez(OUT/'model.npz',proj=proj,cm=cm,cr=cr,weights=weights)
        entries=[]
        for d,(p,y,a,r) in zip(range(end,stop),rows):
            q=p+weights[0]*a[:,None]+weights[1]*r;np.save(OUT/f'pred_{d}.npy',q.astype(np.float32))
            entries.append(tuple(z.reshape(-1,3) for z in (p,q,y)))
        report[stop]={'w':weights.tolist(),'result':prior.measure(entries)}
        (OUT/'results.json').write_text(json.dumps(report,indent=2));print(stop,report[stop],flush=True)
if __name__=='__main__':main()
