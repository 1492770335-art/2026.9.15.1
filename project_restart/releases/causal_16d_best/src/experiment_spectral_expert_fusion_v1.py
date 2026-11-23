"""Spectral expert fusion inside the y3 branch, before market routing/calibration."""
import json
import numpy as np
import experiment_branch_readout_v1 as branches
import experiment_target_strength_v1 as current

prior=current.prior
previous=current.previous
OUT=prior.ART/'spectral_expert_fusion_v1'
MODES=('market_relative','four_factors')


def projector(days):
    cov=np.zeros((50,50))
    for d in days:
        y=prior.labels(d)[:,2].reshape(-1,50).astype(float)
        y-=y.mean(1,keepdims=True)
        cov+=y.T@y/len(y)
    axes=np.linalg.eigh(cov)[1][:,-4:]
    return axes@axes.T


def groups(grid,proj,mode):
    market=grid.mean(1,keepdims=True)
    common=np.broadcast_to(market,grid.shape)
    rem=grid-market
    if mode=='market_relative':
        return (common,rem)
    factor=rem@proj
    return (common,factor,rem-factor)


def layer(tcn,lru,proj,mode):
    a=tcn[:,1].astype(float).reshape(-1,50)
    b=lru[:,1].astype(float).reshape(-1,50)
    terms=[]
    for i,(gain,contrast) in enumerate(zip(groups(.5*(a+b),proj,mode),groups(.5*(a-b),proj,mode))):
        # The unchanged downstream routing multiplies market by .25, relative by .5.
        scale=.25 if i==0 else .5
        terms.extend([scale*gain,scale*contrast])
    return np.column_stack([v.ravel() for v in terms])


def calibrate(p,settings):
    base,disp,strength=settings
    return p+.5*(prior.features(p)@base+previous.inputs(p)@disp)*strength


def settings():
    moments=[];past=[];result={}
    for fold,days in [('A',range(8,12)),('B',range(12,16))]:
        for d,p,y in prior.predictions(fold,days):
            x=prior.features(p);z=previous.inputs(p);r=y.astype(float)-p
            if d in (10,12):base,disp=previous.weights(moments)
            if d in (10,12,14):
                result[d]=(base,disp,np.ones(3) if d==10 else current.choose(past))
            if d>=10:
                past.append(current.evidence(p,.5*(x@base+z@disp),y))
            moments.append((x.T@x,x.T@r,z.T@z,z.T@r,z.T@x))
    result[16]=(*previous.weights(moments),current.choose(past))
    return result


def fit(history):
    gram,rhs=map(sum,zip(*history))
    penalty=.25*np.diag(np.maximum(np.diag(gram),1e-6))
    return .5*np.linalg.solve(gram+penalty,rhs).clip(-.6,.6)


def fold_data(tag,days):
    tcn,lru=branches.branch_predictions(tag)
    joint=np.load(prior.DUAL/('outer_joint_heads.npy' if tag=='outer' else f'heads_{tag}.npy'),mmap_mode='r')
    np.testing.assert_allclose(.5*(tcn+lru),joint,atol=1e-6,rtol=0)
    offset=0
    for d,p,y in prior.predictions(tag,days):
        sl=slice(offset,offset+len(y))
        yield d,p,y,tcn[sl],lru[sl]
        offset+=len(y)
    assert offset==len(joint)


def diagnose(raw,y,proj):
    a=raw[:,2].reshape(-1,50); b=y[:,2].astype(float).reshape(-1,50)
    result={}
    for name,u,v in zip(('market','four_factors','remainder'),groups(a,proj,'four_factors'),groups(b,proj,'four_factors')):
        uf=u.ravel();vf=v.ravel()
        result[name]=dict(variance_share=float(v.var()/b.var()),correlation=float(np.corrcoef(uf,vf)[0,1]),
                          slope=float(np.mean(uf*vf)/np.mean(uf*uf)),mse_share=float(np.mean((u-v)**2)/np.mean((a-b)**2)))
    return result


def entry(p,old,y,x,coef,config):
    raw=p.copy()
    raw[:,2]+=x@coef
    return old,calibrate(raw,config),y


def main():
    OUT.mkdir(exist_ok=True)
    plan=dict(location='TCN/LRU branch y3 gains and disagreement weights before deployed routing and downstream calibration',
              groups=['market plus stock-relative control','market, top four relative y3 train covariance directions, orthogonal remainder'],
              training='past OOF residuals from day 10 onward; half-strength .25 diagonal ridge, final coefficients bounded +/-0.3',
              selection='both early pairs positive, >=3/4 days positive; four-factor version must also beat two-group control both pairs; largest minimum gain',
              dates='zero based 10,11 -> 12,13; 10..13 -> 14,15; selected parameters refit 10..15 -> 16..19')
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    baseline=json.loads((current.OUT/'results.json').read_text());conf=settings()
    report={'daily':{},'early':{},'diagnostic':{},'selected':None}
    history={m:[] for m in MODES}
    report['first_vs_second_six_day_subspace_overlap']=float(np.trace(projector(range(6))@projector(range(6,12)))/4)
    for fold,days,prefix in [('A',range(8,12),8),('B',range(12,16),12)]:
        proj=projector(range(prefix))
        for d,p,y,tcn,lru in fold_data(fold,days):
            if d<10:continue
            config=conf[d//2*2];old=calibrate(p,config)
            if d in (12,14):
                weights={m:fit(history[m]) for m in MODES}; records={m:[] for m in MODES}
            if d>=12:
                report['diagnostic'][d]=diagnose(old,y,proj)
                report['daily'][d]={}
            for m in MODES:
                x=layer(tcn,lru,proj,m)
                if d>=12:
                    row=entry(p,old,y,x,weights[m],config);records[m].append(row)
                    report['daily'][d][m]=prior.measure([row])
                history[m].append((x.T@x,x.T@(y[:,2]-old[:,2])))
            if d in (13,15):
                report['early'][d]={m:prior.measure(records[m]) for m in MODES}
                np.testing.assert_allclose(report['early'][d][MODES[0]]['baseline_score'],baseline['early'][str(d)]['score'],atol=1e-9,rtol=0)
    eligible=[]
    for m in MODES:
        gains=[r[m]['gain'] for r in report['early'].values()]
        if m==MODES[1] and not all(r[m]['gain']>r[MODES[0]]['gain'] for r in report['early'].values()):continue
        if min(gains)>0 and sum(v[m]['gain']>0 for v in report['daily'].values())>=3:
            eligible.append((min(gains),m))
    if eligible:
        mode=max(eligible)[1];report['selected']=mode
        w=fit(history[mode]);proj=projector(range(16));rows=[]
        np.savez(OUT/'layer.npz',weights=w,projector=proj)
        for d,p,y,tcn,lru in fold_data('outer',range(16,20)):
            old=calibrate(p,conf[16])
            rows.append(entry(p,old,y,layer(tcn,lru,proj,mode),w,conf[16]))
            report['daily'][d]={mode:prior.measure(rows[-1:])}
        report['outer']=prior.measure(rows)
        np.testing.assert_allclose(report['outer']['baseline_score'],baseline['outer']['score'],atol=1e-9,rtol=0)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    print(report['early'],report['selected'],report.get('outer'),flush=True)

if __name__=='__main__':
    main()
