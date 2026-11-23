"""Chronological test of opening-dependent expert mixing and amplitude calibration."""
import json
import numpy as np
import experiment_spectral_expert_fusion_v1 as spec

prior = spec.prior
OUT = prior.ART/'opening_expert_v1'
MODES = ('constant_mix', 'opening_mix', 'opening_amplitude')


def opening(n):
    return np.repeat(np.exp(-np.arange(n//50)/256.),50)


def expert_difference(fold,days):
    if fold=='outer':
        with np.load(prior.ART/'y2_normalized_outer/validation_components.npz') as data:
            raw=np.column_stack([data['y0'].ravel(),data['y1'].ravel()])
        norm=np.column_stack([np.load(prior.ART/d/'outer_prediction.npy').ravel() for d in ('normalized_y0','normalized_y1')])
    else:
        raw=np.column_stack([np.load(prior.ART/'normalized_y0'/f'raw_prediction_{fold}.npy').ravel(),
            np.concatenate([np.load(prior.ART/'y2_oof_corrector'/f'day_{d:02d}_oof.npy')[:,0] for d in days])])
        norm=np.column_stack([np.load(prior.ART/'normalized_y0'/f'normalized_prediction_{fold}.npy').ravel(),
                              np.load(prior.ART/'normalized_y1'/f'prediction_{fold}.npy').ravel()])
    return raw-norm,.5*(raw+norm)


def frames(fold,days,proj,settings,spectral_history):
    diff,average=expert_difference(fold,days)
    offset=0
    for day,p,y,tcn,lru in spec.fold_data(fold,days):
        sl=slice(offset,offset+len(p));offset+=len(p)
        np.testing.assert_allclose(p[:,:2],average[sl],atol=2e-6,rtol=0)
        if day<10:continue
        conf=settings[min(16,day//2*2)]
        if day in (10,12,14,16):
            mix=spec.fit(spectral_history) if spectral_history else np.zeros(6)
            if day==16:
                np.testing.assert_allclose(mix,np.load(spec.OUT/'layer.npz')['weights'],atol=1e-9)
        h=spec.layer(tcn,lru,proj,'four_factors')
        unsplit=spec.calibrate(p,conf)
        raw=p.copy()
        raw[:,2]+=h@mix
        yield day,raw,spec.calibrate(raw,conf),diff[sl],y,conf
        if day<16:spectral_history.append((h.T@h,h.T@(y[:,2]-unsplit[:,2])))
    assert offset==len(diff)


def stats(signal,target,o):
    basis=np.stack((signal,signal*o[:,None]),axis=2)
    return (np.einsum('nki,nkj->kij',basis,basis),np.einsum('nki,nk->ki',basis,target))


def fit(moments,mode):
    gram,rhs=map(sum,zip(*moments))
    values=[]
    for g,b in zip(gram,rhs):
        if mode=='constant_mix':
            values.append([np.clip(.5*b[0]/(1.15*g[0,0]),-.3,.3),0.])
        elif mode=='opening_mix':
            values.append(np.clip(.5*np.linalg.solve(g+.15*np.diag(np.diag(g)),b),-.3,.3))
        else:
            # Free constant slope cancels in per-target Pearson; keep relative opening slope only.
            reg=np.diag([0.,.15*g[1,1]])
            slope=np.linalg.solve(g+reg,b)
            values.append([0.,np.clip(.5*slope[1]/max(.25,slope[0]),-.35,.35)])
    return np.array(values)


def apply(mode,coef,raw,base,diff,conf):
    o=opening(len(raw))[:,None]
    effect=coef[:,0]+o*coef[:,1]
    if mode=='opening_amplitude':
        return base*(1+effect)
    delta=diff*np.clip(effect,-.35,.35)
    changed=raw.copy()
    changed[:,:2]+=delta
    second=delta[:,1].reshape(-1,50)
    changed[:,2]+=(.5*second+.25*second.mean(1,keepdims=True)).ravel()
    return spec.calibrate(changed,conf)


def main():
    OUT.mkdir(exist_ok=True)
    plan=dict(modes=MODES,phase='exp(-observed zero-based tick/256); resets each day',
              fitting='OOF from days 10,11 for 12,13; through day 13 for 14,15. Day 10/11 use zero spectral correction for fitting; no future coefficients.',
              selection='both pairs positive and >=3/4 positive days; opening_mix also beats constant_mix both pairs; greatest minimum gain wins; refit through day 15')
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    report={'early':{},'daily':{},'selected':None,'opening_diagnostics':{}}
    settings=spec.settings();hist=[];amp=[];mix=[]
    baseline=json.loads((spec.OUT/'results.json').read_text())
    for fold,days,end in [('A',range(8,12),8),('B',range(12,16),12)]:
        for d,raw,base,diff,y,conf in frames(fold,days,spec.projector(range(end)),settings,hist):
            if d in (12,14):
                coef={m:fit(amp if m=='opening_amplitude' else mix,m) for m in MODES}
                records={m:[] for m in MODES}
            if d>=12:
                report['daily'][d]={}
                report['opening_diagnostics'][d]={}
                for key,sl in [('first256',slice(0,12800)),('middle',slice(12800,-12800))]:
                    a=base[sl];b=y[sl].astype(float)
                    report['opening_diagnostics'][d][key]={'target_std':b.std(0).tolist(),'slope':((a*b).sum(0)/(a*a).sum(0)).tolist()}
                for m in MODES:
                    cand=apply(m,coef[m],raw,base,diff,conf)
                    records[m].append((base,cand,y))
                    report['daily'][d][m]=prior.measure(records[m][-1:])
                if d in (13,15):
                    report['early'][d]={m:prior.measure(records[m]) for m in MODES}
                    np.testing.assert_allclose(report['early'][d][MODES[0]]['baseline_score'],baseline['early'][str(d)]['four_factors']['score'],atol=1e-9,rtol=0)
            o=opening(len(raw))
            amp.append(stats(base,y.astype(float),o))
            mix.append(stats(diff.astype(float),y[:,:2]-base[:,:2],o))
    passing=[]
    for m in MODES:
        gains=[r[m]['gain'] for r in report['early'].values()]
        if m=='opening_mix' and not all(r[m]['gain']>r['constant_mix']['gain'] for r in report['early'].values()):continue
        if min(gains)>0 and sum(r[m]['gain']>0 for r in report['daily'].values())>=3:passing.append((min(gains),m))
    if passing:
        m=max(passing)[1];report['selected']=m
        w=fit(amp if m=='opening_amplitude' else mix,m)
        report['weights']=w.tolist()
        rows=[]
        for d,raw,base,diff,y,conf in frames('outer',range(16,20),spec.projector(range(16)),settings,hist):
            rows.append((base,apply(m,w,raw,base,diff,conf),y))
            report['daily'][d]={m:prior.measure(rows[-1:])}
        report['outer']=prior.measure(rows)
        np.testing.assert_allclose(report['outer']['baseline_score'],baseline['outer']['score'],rtol=0,atol=1e-9)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    print([(d,{m:round(v['gain'],7) for m,v in b.items()}) for d,b in report['early'].items()], report['selected'],report.get('outer'),flush=True)

if __name__=='__main__':
    main()
