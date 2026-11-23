"""Causal, cross-section centered correction conditional on prediction dispersion."""
import json
import numpy as np
import experiment_causal_cross_horizon_ridge_v1 as prior

OUT = prior.ART/'relative_dispersion_v1'

def inputs(p):
    g=p.reshape(-1,50,3).astype(np.float64)
    r=g-g.mean(1,keepdims=True)
    # A continuous bounded proxy for an unusually strong per-stock signal.
    high=np.minimum(np.abs(r)/np.maximum(np.sqrt((r*r).mean(1,keepdims=True)),.01),3.)-1
    # Curvature within each signal and interactions between target predictions.
    cols=(r[:,:,:,None]*high[:,:,None,:]).reshape(len(g),50,9)
    cols=cols-cols.mean(1,keepdims=True)
    return np.concatenate([r,cols],axis=-1).reshape(len(p),12)

def weights(moments):
    xx,xy,zz,zr,zx=map(sum,zip(*moments))
    base=np.linalg.solve(xx+.05*np.diag(np.maximum(np.diag(xx),1e-6)),xy)
    adjust=np.linalg.solve(zz+.2*np.diag(np.maximum(np.diag(zz),1e-6)),zr-.5*zx@base)
    return base,adjust

def main():
    OUT.mkdir(exist_ok=True)
    moments=[];report={'early':{},'daily':{},'plan':'half-strength shared relative-only correction for all 3 targets; ridge .2; both early blocks positive and >=4/6 days positive'}
    for fold,days in [('A',range(8,12)),('B',range(12,16))]:
        rows=[]
        for day,p,y in prior.predictions(fold,days):
            x=prior.features(p);z=inputs(p);res=y.astype(np.float64)-p
            if day in (10,12):
                base,adjust=weights(moments)
            if day>=10:
                old=p+.5*x@base
                rows.append((old,old+.5*z@adjust,y))
                report['daily'][day]=prior.measure(rows[-1:])
            moments.append((x.T@x,x.T@res,z.T@z,z.T@res,z.T@x))
        report['early'][fold]=prior.measure(rows)
    report['passed']=all(v['gain']>0 for v in report['early'].values()) and sum(v['gain']>0 for v in report['daily'].values())>=4
    if report['passed']:
        base,adjust=weights(moments)
        np.testing.assert_allclose(base,np.load(prior.OUT/'coefficients.npy'),atol=1e-9)
        np.save(OUT/'coefficients.npy',adjust)
        rows=[]
        for day,p,y in prior.predictions('outer',range(16,20)):
            old=p+.5*prior.features(p)@base
            rows.append((old,old+.5*inputs(p)@adjust,y))
            report['daily'][day]=prior.measure(rows[-1:])
        report['outer']=prior.measure(rows)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    print(report['early'], report.get('outer'),flush=True)

if __name__=='__main__':
    main()
