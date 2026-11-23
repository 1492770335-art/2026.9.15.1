"""Nested walk-forward tuning of the existing two calibration layers' joint strength."""
import json
import numpy as np
import experiment_relative_dispersion_v1 as previous

prior = previous.prior
OUT = prior.ART/'target_strength_v1'
GRID = np.array([0.,.25,.5,.75,1.,1.25,1.5,1.75,2.])


def evidence(p, delta, y):
    return np.stack([prior.corr(p+a*delta,y) for a in GRID])


def choose(past):
    # Select the best average daily correlation per target; shrink halfway toward deployed strength.
    mean = np.mean(past,axis=0)
    return .5*(1.+GRID[mean.argmax(axis=0)])


def main():
    OUT.mkdir(exist_ok=True)
    plan={'grid':GRID.tolist(),'shrink_to_one':.5,'fit':'choose by equal-day mean Pearson on earlier out-of-fold corrections only',
          'blocks':'calibrate with internal checks on days 10,11 -> test 12,13; add checks from 12,13 -> test 14,15; refit using 10..15 -> final 16..19',
          'outer_gate':'both blocks positive weighted gain and at least 3/4 daily positive'}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    past=[]; moments=[];report={'strengths':{},'daily':{},'early':{}}
    for fold,days in [('A',range(8,12)),('B',range(12,16))]:
        for day,p,y in prior.predictions(fold,days):
            x=prior.features(p);z=previous.inputs(p);res=y.astype(np.float64)-p
            if day in (10,12):
                base,dispersion=previous.weights(moments)
            if day>=10:
                correction=.5*(x@base+z@dispersion)
                if day in (12,14):
                    strength=choose(past)
                    report['strengths'][day]=strength.tolist()
                    rows=[]
                if day>=12:
                    rows.append((p+correction,p+correction*strength,y))
                    report['daily'][day]=prior.measure(rows[-1:])
                    if day in (13,15):
                        report['early'][day]=prior.measure(rows)
                past.append(evidence(p,correction,y))
            moments.append((x.T@x,x.T@res,z.T@z,z.T@res,z.T@x))
    report['passed']=all(r['gain']>0 for r in report['early'].values()) and sum(r['gain']>0 for r in report['daily'].values())>=3
    if report['passed']:
        strength=choose(past)
        report['strengths'][16]=strength.tolist()
        base,dispersion=previous.weights(moments)
        np.testing.assert_allclose(base,np.load(prior.OUT/'coefficients.npy'),atol=1e-9)
        np.testing.assert_allclose(dispersion,np.load(previous.OUT/'coefficients.npy'),atol=1e-9)
        rows=[]
        for day,p,y in prior.predictions('outer',range(16,20)):
            correction=.5*(prior.features(p)@base+previous.inputs(p)@dispersion)
            rows.append((p+correction,p+correction*strength,y))
            report['daily'][day]=prior.measure(rows[-1:])
        report['outer']=prior.measure(rows)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    print(report['early'],report['strengths'],report.get('outer'),flush=True)

if __name__=='__main__':
    main()
