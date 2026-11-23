"""Update a low-dimensional expert readout from completed public price intervals with a fixed prior."""
import json
import numpy as np
import experiment_consensus_horizon_portfolios_v1 as blend
import experiment_pairwise_book_tree_v1 as book
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure
from experiment_eight_date_expert_contrasts_v1 import moments,fit
from experiment_predictive_book_modes_v1 import validation

OUT=ART/'matured_portfolio_posterior_v1'
H=(5,20,60)
BLOCK=32
WINDOW=1800
STEP=4


def components(raw,p,experts):
    n=len(raw);t=np.arange(n);start=np.arange(0,n,BLOCK)
    price=np.cumsum(raw[:,:,0].astype(float)-.00136,axis=0)
    parts=np.stack(experts,axis=-1)
    avg=parts.mean(3)
    # Relative expert contrasts are in the same units as the original three forecasts.
    x=np.concatenate((p,avg-p,(parts[:,:,:,0]-parts[:,:,:,1]),(parts[:,:,:,2]-parts[:,:,:,3]),
                      .5*(parts[:,:,:,0]+parts[:,:,:,1]-parts[:,:,:,2]-parts[:,:,:,3])),axis=2)
    market=x.mean(1,keepdims=True)
    result=[]
    for group,z in enumerate((market,x-market)):
        output=np.zeros((n,z.shape[1],3),np.float32)
        for j,h in enumerate(H):
            origin=np.arange(0,max(0,n-h),STEP)
            if not len(origin):continue
            mature=origin+h
            a=z[origin].astype(float)
            # At a block boundary only targets with maturity strictly before the boundary enter the readout.
            error=(price[mature]-price[origin])*blend.UNITS[j]-p[origin,:,j]
            em=error.mean(1,keepdims=True)
            e=(em if group==0 else error-em)[:,:,None]
            gram=a.transpose(0,2,1)@a/a.shape[1]
            rhs=(a.transpose(0,2,1)@e)[:,:,0]/a.shape[1]
            hi=np.searchsorted(mature,start);lo=np.searchsorted(mature,start-WINDOW)
            count=(hi-lo).clip(1)
            def trailing(arr):
                cs=np.concatenate((np.zeros_like(arr[:1]),arr.cumsum(0,dtype=float)))
                return (cs[hi]-cs[lo])/count.reshape((-1,)+(1,)*(arr.ndim-1))
            g=trailing(gram);b=trailing(rhs)
            diag=g.diagonal(axis1=1,axis2=2).clip(.0008,None)
            # The prior is strongest early in the session and for market-wide fluctuations.
            pseudo=(2400 if group==0 else 900)/STEP
            reg=diag*(.32+pseudo/count[:,None])
            w=np.linalg.solve(g+np.eye(a.shape[-1])[None]*reg[:,None,:],b[:,:,None])[:,:,0].clip(-.8,.8)
            output[:,:,j]=np.einsum('tsf,tf->ts',z,w[t//BLOCK],optimize=True)
        result.append(output[:,0] if group==0 else output)
    return result


def fit_strength(hist):
    # Fit one scalar per horizon and component. Adjacent horizons have separate maturities.
    weights=[]
    for points in zip(*hist):
        g,b=map(np.stack,zip(*points))
        variance=g.diagonal(axis1=1,axis2=2)
        response=b.diagonal(axis1=1,axis2=2)
        ridge=.55*variance.mean(0).clip(.000006,None)
        daily=response/(variance+ridge)
        value=.55*response.mean(0)/(variance.mean(0)+ridge)+.45*np.median(daily,axis=0)
        weights.append((.68*value*abs(np.sign(daily).mean(0))).clip(0,1))
    return np.stack(weights)


def advance(p,z,w):
    return (p+(z[0]*w[0])[:,None]+z[1]*w[1]).astype('float32')


def main():
    OUT.mkdir(exist_ok=True);hist=[];priors=[];daily=[];report={'folds':[]}
    for end in (8,10,12,14):
        c=blend.fixed_weight(priors);w=fit_strength(hist) if hist else None
        rows=[[],[]]
        for d in range(end,end+2):
            p,y,fs,stat=blend.training_day(d,priors);priors.append(stat)
            p=blend.parent.fixed_forecast(p,fs,c);raw=book.rawday(d);z=components(raw,p,fs)
            if d==end:
                head=components(raw[:341],p[:341],[v[:341] for v in fs])
                for a,b in zip(z,head):np.testing.assert_allclose(a[:341],b,atol=1e-6,rtol=0)
            if w is not None:
                change=[]
                for i in range(2):
                    wi=np.zeros_like(w);wi[i]=w[i];q=advance(p,z,wi)
                    row=tuple(a.reshape(-1,3) for a in (p,q,y));rows[i].append(row);s=measure([row])
                    change.append(np.subtract(s['candidate'],s['base']))
                daily.append(change)
            hist.append(moments(p[::3],y[::3],(z[0][::3],z[1][::3])))
        if rows[0]:
            info=[measure(r) for r in rows];report['folds'].append(info);print(end+1,info,flush=True)
    gains=np.stack([[np.subtract(s['candidate'],s['base']) for s in fold] for fold in report['folds']])
    gate=(gains.min(0)>0)&((np.array(daily)>0).sum(0)>=4)
    report['retained']=gate.tolist();path=OUT/'results.json';path.write_text(json.dumps(report,indent=2))
    if not gate.any():return
    from experiment_replicated_expert_consensus_v1 import sources
    models=sources();c=np.load(blend.PARENT/'weights.npy')[0];w=fit_strength(hist)*gate
    np.save(OUT/'weights.npy',w)
    def predict(raw,p):
        fs=[fn(raw) for fn in models];p=blend.parent.fixed_forecast(p,fs,c)
        return advance(p,components(raw,p,fs),w)
    result=validation(predict)
    ref=json.loads((blend.PARENT/'raw_replay.json').read_text())
    result['gain_over_fixed']=result['pooled']['score']-ref['pooled']['score']
    report['development']=result;path.write_text(json.dumps(report,indent=2));print(result,flush=True)


if __name__=='__main__':main()
