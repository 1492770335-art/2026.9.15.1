"""Infer relative precision from cross-expert covariances and test it against the equal-weight consensus."""
import json
import numpy as np
import experiment_consensus_horizon_portfolios_v1 as blend
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure
from experiment_matured_portfolio_posterior_v1 import components as recent,advance as add_recent,OUT as PARENT

OUT=ART/'consensus_error_variance_v1'


def observed(fs):
    x=np.stack([blend.bands(v) for v in fs],axis=-1)
    x-=x.mean(1,keepdims=True)
    return x


def covariances(x):
    return np.einsum('tshe,tshj->thej',x,x,optimize=True)/x.shape[1]


def features(fs,prior):
    x=observed(fs)
    gram=covariances(x);t=np.arange(len(x))+1
    total=np.concatenate((np.zeros_like(gram[:1],dtype=float),gram.cumsum(0,dtype=float)))
    sets=[]
    for width in (32,360):
        start=(t-width).clip(0)
        # The early-date prior stabilizes estimates at the start of every session.
        c=(total[t]-total[start]+48*prior)/(t-start+48)[:,None,None,None]
        variance=c.diagonal(axis1=-2,axis2=-1).clip(.00001,None)
        common=[]
        for i in range(4):
            indices=[j for j in range(4) if j!=i]
            samples=[]
            for j,k in ((indices[0],indices[1]),(indices[0],indices[2]),(indices[1],indices[2])):
                lower=.15*np.sqrt(variance[:,:,j]*variance[:,:,k])
                samples.append(c[:,:,i,j]*c[:,:,i,k]/np.maximum(lower,c[:,:,j,k]))
            common.append(np.median(samples,axis=0))
        shared=np.stack(common,axis=-1).clip(.12*variance,.94*variance)
        amplitude=np.sqrt(shared)
        noise=(variance-shared)/shared
        weights=1/(.24+noise)
        weights/=weights.sum(2,keepdims=True)
        calibrated=x*(amplitude.mean(2,keepdims=True)/amplitude).clip(.5,2)[:,None]
        basic=x.mean(3);rescale=calibrated.mean(3)-basic
        precision=(calibrated*weights[:,None]).sum(3)-calibrated.mean(3)
        sets.extend((rescale,precision))
    return np.concatenate(sets,axis=2).astype('float32')


def stats(z,p,y):
    a=p.reshape(-1,3).astype(float);a-=a.mean(0);b=y.reshape(-1,3).astype(float);b-=b.mean(0)
    beta=((a*b).mean(0)/(a*a).mean(0)).clip(.35,3)
    e=y/beta-p;e-=e.mean(1,keepdims=True)
    x=z.reshape(-1,z.shape[-1]).astype(float)
    return x.T@x/len(x),x.T@e.reshape(-1,3)/len(x)


def fit(history):
    g,b=map(np.stack,zip(*history))
    reg=1.2*np.diag(g.mean(0).diagonal().clip(.00001,None))
    daily=np.linalg.solve(g+reg,b)
    pooled=np.linalg.solve(g.mean(0)+reg,b.mean(0))
    return (.65*(.6*pooled+.4*np.median(daily,axis=0))*abs(np.sign(daily).mean(0))).clip(-.28,.28).astype('float32')


def main():
    OUT.mkdir(exist_ok=True);hist=[];priors=[];cov=[];report={'folds':[]};daily=[]
    for end in (8,10,12,14):
        c=blend.fixed_weight(priors);w=fit(hist) if hist else None
        prior=np.mean(cov,axis=0) if cov else None;rows=[]
        for d in range(end,end+2):
            p,y,fs,stat=blend.training_day(d,priors);priors.append(stat)
            p=blend.parent.fixed_forecast(p,fs,c)
            # Early days supply the starting covariance without using any return labels.
            if end==8:
                cov.append(covariances(observed([v[::6] for v in fs])).mean(0))
                local=cov[-1]
            else:local=prior
            z=features(fs,local)
            if d==end:
                head=features([v[:337] for v in fs],local)
                np.testing.assert_allclose(z[:337],head,rtol=0,atol=1e-6)
            if w is not None:
                q=(p+z@w).astype('float32')
                row=tuple(a.reshape(-1,3) for a in (p,q,y));rows.append(row)
                s=measure([row]);daily.append(np.subtract(s['candidate'],s['base']))
            hist.append(stats(z[::3],p[::3],y[::3]))
        if rows:
            s=measure(rows);report['folds'].append(s);print(end+1,s,flush=True)
    gains=np.array([np.subtract(f['candidate'],f['base']) for f in report['folds']])
    gate=(gains.min(0)>0)&((np.array(daily)>0).sum(0)>=4);gate[0]=False
    report['retained']=gate.tolist();path=OUT/'results.json';path.write_text(json.dumps(report,indent=2))
    if not gate.any():return
    from experiment_replicated_expert_consensus_v1 import sources
    from experiment_predictive_book_modes_v1 import validation
    experts=sources();c=np.load(blend.PARENT/'weights.npy')[0];pw=np.load(PARENT/'weights.npy')
    w=fit(hist)*gate;prior=np.mean(cov,axis=0)
    np.savez(OUT/'weights.npz',w=w,prior=prior)
    def advance(raw,p):
        fs=[fn(raw) for fn in experts];p=blend.parent.fixed_forecast(p,fs,c)
        p=add_recent(p,recent(raw,p,fs),pw)
        return (p+features(fs,prior)@w).astype('float32')
    result=validation(advance);ref=json.loads((PARENT/'raw_replay.json').read_text())
    result['gain_over_parent']=result['pooled']['score']-ref['pooled']['score']
    report['development']=result;path.write_text(json.dumps(report,indent=2));print(result,flush=True)


if __name__=='__main__':main()
