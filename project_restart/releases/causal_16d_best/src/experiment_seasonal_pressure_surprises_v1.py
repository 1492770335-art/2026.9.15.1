"""Measure signed order flow and participation relative to each stock's historical intraday profile."""
import json
import numpy as np
import experiment_pairwise_book_tree_v1 as book
import experiment_consensus_horizon_portfolios_v1 as blend
from experiment_eight_date_expert_contrasts_v1 import moments,fit
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure
from experiment_predictive_book_modes_v1 import validation

OUT=ART/'seasonal_pressure_surprises_v1'
BLOCK=100
BINS=48


def flows(raw):
    def col(i):return np.arcsinh(raw[:,:,i].astype('float32'))
    ret=col(0)
    signed=np.stack((ret,col(54)-col(55),col(77)-col(78),col(41)-col(42),
                     col(26)-col(38),col(20)-col(23)),axis=2)
    activity=np.stack((abs(ret),col(8),col(39),abs(signed[:,:,1]),abs(signed[:,:,2]),abs(signed[:,:,3])),axis=2)
    return np.concatenate((signed,activity),axis=2)


def profile():
    by_day=[];second=[]
    for d in range(8):
        values=flows(book.rawday(d))
        means=[];squares=[]
        for k in range(BINS):
            group=values[k*BLOCK:(k+1)*BLOCK]
            if not len(group):
                means.append(means[-1]);squares.append(squares[-1]);continue
            means.append(group.mean(0));squares.append((group*group).mean(0))
        by_day.append(means);second.append(squares)
    def smooth(x):
        x=np.median(x,axis=0)
        trend=np.stack([np.mean(x[max(0,i-2):i+3],axis=0) for i in range(len(x))])
        return .65*trend+.22*x+.13*x.mean(0)
    mean=smooth(by_day)
    scale=(smooth(second)-mean**2).clip(.003,None)**.5
    return mean.astype('float32'),scale.astype('float32')


def average(x,width):
    cs=np.concatenate((np.zeros_like(x[:1],dtype=float),x.cumsum(0,dtype=float)))
    t=np.arange(1,len(cs));left=(t-width).clip(0)
    return ((cs[t]-cs[left])/(t-left)[:,None,None]).astype('float32')


def features(raw,p,profile,seasonal):
    mean,scale=profile;s=flows(raw);clock=np.minimum(np.arange(len(p))//BLOCK,BINS-1)
    mu=mean[clock] if seasonal else mean.mean(0)
    unit=scale[clock] if seasonal else scale.mean(0)
    obs=((s-mu)/unit).clip(-5,5)
    f=obs[:,:,:6];activity=obs[:,:,6:]
    terms=[];near=p[:,:,0:1];far=p[:,:,2:3]-blend.panel.RATIO*p[:,:,1:2]
    for width in (32,320,1600):
        a=np.tanh(average(activity,width));flow=average(f,width)
        level=a.mean(2,keepdims=True)
        terms.extend((flow,np.tanh(flow)*a,near*a,far*a,flow*level))
    # The accumulated surprise distinguishes today's persistently unusual volume from a brief spike.
    running=average(obs,len(p))
    terms.extend((near*np.tanh(running[:,:,6:]),far*np.tanh(running[:,:,6:])))
    z=np.concatenate(terms,axis=-1).astype('float32')
    m=z.mean(1)
    return m,z-m[:,None]


def apply(p,x,w):
    return (p+(x[0]@w[0])[:,None]+x[1]@w[1]).astype('float32')


def main():
    OUT.mkdir(exist_ok=True);geom=profile();hist=[[],[]];report={'folds':[]};daily=[];priors=[]
    for end in (8,10,12,14):
        ws=[fit(h) if h else None for h in hist];c=blend.fixed_weight(priors);rows=[[],[]]
        for d in range(end,end+2):
            p,y,fs,stat=blend.training_day(d,priors);priors.append(stat)
            p=blend.parent.fixed_forecast(p,fs,c);raw=book.rawday(d)
            for i in range(2):
                z=features(raw,p,geom,bool(i))
                if d==end:
                    short=features(raw[:277],p[:277],geom,bool(i))
                    for a,b in zip(z,short):np.testing.assert_allclose(a[:277],b,atol=1e-6,rtol=0)
                if ws[i] is not None:
                    q=apply(p,z,ws[i]);row=tuple(a.reshape(-1,3) for a in (p,q,y));rows[i].append(row)
                    if i:
                        s=measure([row]);daily.append(np.subtract(s['candidate'],s['base']))
                hist[i].append(moments(p[::3],y[::3],(z[0][::3],z[1][::3])))
        if rows[0]:
            fold={k:measure(r) for k,r in zip(('constant','seasonal'),rows)};report['folds'].append(fold)
            print(end+1,fold,flush=True)
    gains=np.stack([np.subtract(f['seasonal']['candidate'],f['seasonal']['base']) for f in report['folds']])
    adv=np.stack([np.subtract(f['seasonal']['candidate'],f['constant']['candidate']) for f in report['folds']])
    gate=(gains.min(0)>0)&(adv.mean(0)>0)&((np.stack(daily)>0).sum(0)>=4)
    path=OUT/'results.json';report['retained']=gate.tolist();path.write_text(json.dumps(report,indent=2))
    if not gate.any():return
    from experiment_replicated_expert_consensus_v1 import sources
    from experiment_matured_portfolio_posterior_v1 import components as recent,advance as add_recent,OUT as recent_out
    import experiment_consensus_error_variance_v1 as variance
    w=np.stack(fit(hist[1]))*gate;experts=sources()
    c=np.load(blend.PARENT/'weights.npy')[0];ow=np.load(recent_out/'weights.npy')
    with np.load(variance.OUT/'weights.npz') as st:initial,cw=st['prior'],st['w']
    np.savez(OUT/'model.npz',mean=geom[0],scale=geom[1],w=w)
    def forecast(raw,p):
        fs=[fn(raw) for fn in experts];p=blend.parent.fixed_forecast(p,fs,c)
        p=add_recent(p,recent(raw,p,fs),ow);p=(p+variance.features(fs,initial)@cw).astype('float32')
        return apply(p,features(raw,p,geom,True),w)
    result=validation(forecast)
    reference=json.loads((variance.OUT/'results.json').read_text())['development']['pooled']['score']
    result['gain_over_parent']=result['pooled']['score']-reference
    report['development']=result;path.write_text(json.dumps(report,indent=2));print(result,flush=True)


if __name__=='__main__':main()
