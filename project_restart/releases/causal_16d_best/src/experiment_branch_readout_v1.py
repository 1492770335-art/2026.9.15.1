"""Walk-forward residual readouts for the auxiliary and disagreement heads."""
import json
import numpy as np
import experiment_relative_dispersion_v1 as previous

prior = previous.prior
OUT = prior.ART / 'branch_readout_v1'
BRANCH = prior.ART / 'peer_temporal_arch'
WIDTHS = (1, 16, 64)
# Within each window: individual channels followed by their frame means.
DISAGREEMENT = [6*i+j for i in range(3) for j in (1,2,4,5)]
VARIANTS = {'disagreement': DISAGREEMENT, 'auxiliary_and_disagreement': list(range(18))}

def branch_predictions(fold):
    paths = ((BRANCH/'outer_joint_heads.npy', prior.DUAL/'outer_lru_heads.npy') if fold=='outer'
             else tuple(BRANCH/f'{name}_{fold}_heads.npy' for name in ('tcn_local','complex_lru')))
    return tuple(np.load(path,mmap_mode='r') for path in paths)


def inputs(p, tcn, lru):
    residuals = np.column_stack((.5*(tcn[:,0]+lru[:,0])-p[:,1], tcn[:,0]-lru[:,0], tcn[:,1]-lru[:,1]))
    grid = residuals.astype(np.float64).reshape(-1,50,3)
    end = np.arange(1,len(grid)+1)
    total = np.concatenate((np.zeros_like(grid[:1]),grid.cumsum(0)))
    columns = []
    for width in WIDTHS:
        start = np.maximum(0,end-width)
        mean = (total[end]-total[start])/(end-start)[:,None,None]
        market = mean.mean(1,keepdims=True)
        columns.extend((mean-market,np.broadcast_to(market,mean.shape)))
    return np.concatenate(columns,axis=-1).reshape(len(p),18)


def coefficients(old_moments, new_moments):
    base, dispersion = previous.weights(old_moments)
    gram, residual, cross_x, cross_z = map(sum,zip(*new_moments))
    rhs = residual - .5*(cross_x@base+cross_z@dispersion)
    readouts = {}
    for mode,keep in VARIANTS.items():
        g = gram[np.ix_(keep,keep)]
        weights = np.linalg.solve(g+.2*np.diag(np.maximum(np.diag(g),1e-6)),rhs[keep])
        full = np.zeros_like(rhs)
        full[keep] = weights
        readouts[mode]=full
    return base,dispersion,readouts


def blocks(fold,days):
    tcn,lru=branch_predictions(fold)
    start=0
    for day,p,y in prior.predictions(fold,days):
        stop=start+len(y)
        yield day,p,y,inputs(p,tcn[start:stop],lru[start:stop])
        start=stop
    assert len(tcn)==len(lru)==start


def main():
    OUT.mkdir(exist_ok=True)
    plan={'widths':WIDTHS,'variants':VARIANTS,'ridge':.2,'strength':.5,
          'select':'positive score on both early blocks and 4/6 positive days; choose largest minimum block gain',
          'folds':'calendar order: fit on 8,9 for 10,11; fit 8..11 for 12..15; refit 8..15 for 16..19'}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    old_moments=[]; new_moments=[]
    report={'early':{},'daily':{}}
    old_report=json.loads((previous.OUT/'results.json').read_text())
    for fold,days in [('A',range(8,12)),('B',range(12,16))]:
        records={name:[] for name in VARIANTS}
        for day,p,y,h in blocks(fold,days):
            x=prior.features(p)
            z=previous.inputs(p)
            res=y.astype(np.float64)-p
            if day in (10,12):
                base,dispersion,models=coefficients(old_moments,new_moments)
            if day>=10:
                old=p+.5*(x@base+z@dispersion)
                report['daily'][day]={}
                for name,weights in models.items():
                    item=(old,old+.5*(h@weights),y)
                    records[name].append(item)
                    report['daily'][day][name]=prior.measure([item])
            old_moments.append((x.T@x,x.T@res,z.T@z,z.T@res,z.T@x))
            new_moments.append((h.T@h,h.T@res,h.T@x,h.T@z))
        report['early'][fold]={name:prior.measure(items) for name,items in records.items()}
        for rec in report['early'][fold].values():
            np.testing.assert_allclose(rec['baseline_score'],old_report['early'][fold]['score'],rtol=0,atol=1e-9)
    minima={m:min(b[m]['gain'] for b in report['early'].values()) for m in VARIANTS}
    eligible=[m for m in VARIANTS if minima[m]>0 and sum(v[m]['gain']>0 for v in report['daily'].values())>=4]
    selected=max(eligible,key=minima.get) if eligible else None
    report['selected']=selected
    if selected:
        base,dispersion,models=coefficients(old_moments,new_moments)
        weights=models[selected]
        np.save(OUT/'coefficients.npy',weights)
        records=[]
        for day,p,y,h in blocks('outer',range(16,20)):
            old=p+.5*(prior.features(p)@base+previous.inputs(p)@dispersion)
            records.append((old,old+.5*h@weights,y))
            report['daily'][day]={selected:prior.measure(records[-1:])}
        report['outer']=prior.measure(records)
        np.testing.assert_allclose(report['outer']['baseline_score'],old_report['outer']['score'],rtol=0,atol=1e-9)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    print(report['early'],selected,report.get('outer'),flush=True)

if __name__=='__main__':
    main()
