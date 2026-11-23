"""Forecast the common return from causal liquidity-weighted flow portfolios in event time."""
import json
import numpy as np
import experiment_pooled_price_impact_v1 as impact
import experiment_factor_exposure_return_v1 as ex
import experiment_multiscale_return_net_v1 as v1
from experiment_causal_return_basket_v1 import moving_sum
prior=v1.prior
OUT=prior.ART/'liquidity_clock_market_v1'


def design(raw):
    n=len(raw); ix=np.arange(n);ed=ix+1
    a=np.arcsinh(raw.astype(np.float32))
    price=raw[:,:,0].astype(float)-.00136
    hit=abs(price)>.01
    activity=(12+moving_sum(hit,96))/(24+np.minimum(ed,96))[:,None]
    vol=np.sqrt((4+moving_sum(price**2,96))/(4+np.minimum(ed,96))[:,None])
    stale=ix[:,None]-np.maximum.accumulate(np.where(hit,ix[:,None],0),axis=0)
    flow=np.stack((a[:,:,0],a[:,:,10],a[:,:,11],a[:,:,14],a[:,:,16],
                   a[:,:,20:24].mean(2),a[:,:,24:28].mean(2),a[:,:,28:32].mean(2),
                   a[:,:,41]-a[:,:,42],a[:,:,38],a[:,:,50]-a[:,:,51],
                   a[:,:,52]-a[:,:,53],a[:,:,54]-a[:,:,55],a[:,:,56]-a[:,:,57],
                   a[:,:,58:62].mean(2),a[:,:,62:66].mean(2),a[:,:,66:70].mean(2),
                   a[:,:,70:74].mean(2),a[:,:,75:77].mean(2),raw[:,:,84].astype(float)-1.,
                   a[:,:,1]-a[:1,:,1],a[:,:,5],a[:,:,17]),axis=2)
    weights=(np.ones_like(vol),1/(.35+vol),.05+activity,np.sqrt(1+np.minimum(stale,50)),.2+np.minimum(4,abs(price)))
    portfolios=[]
    for j,w in enumerate(weights):
        f=np.einsum('tsf,ts->tf',flow,w)/w.sum(1)[:,None]
        portfolios.append(f if j==0 else f-portfolios[0])
    values=np.column_stack(portfolios)
    # Accumulated trading activity defines the lookback boundaries at tick t.
    speed=(hit.mean(1)+.10)/.40
    clock=np.r_[0.,speed.cumsum()]
    total=np.vstack((np.zeros(values.shape[1]),values.cumsum(0)))
    columns=[values]
    for low,high in [(1,5),(5,20),(20,60),(60,180)]:
        lo=np.maximum(0,ed-high);hi=np.maximum(0,ed-low)
        direct=(total[hi]-total[lo])/np.maximum(1,hi-lo)[:,None]
        columns.append(direct)
        lo=np.searchsorted(clock,np.maximum(0,clock[ed]-high),side='left')
        hi=np.searchsorted(clock,np.maximum(0,clock[ed]-low),side='left')
        event=(total[hi]-total[lo])/np.maximum(1,hi-lo)[:,None]
        columns.append(event[:,:flow.shape[2]]-direct[:,:flow.shape[2]])
    shared=values[:,:flow.shape[2]]
    for width in (1,8,40):
        s=moving_sum(shared,width)/np.minimum(ed,width)[:,None]
        columns.extend((np.tanh(4*s),s*np.exp(-ix/320.)[:,None],s*np.log(.2+vol.mean(1))[:,None]))
    return np.column_stack(columns),vol.mean(1)


def fit(end):
    linear=[];weighted=[]
    for d in range(end):
        x,vol=design(v1.grid(d))
        y=prior.labels(d).reshape(-1,50,3).astype(float).mean(1)
        for w,dest in [(np.ones(len(x)),linear),(1/np.sqrt(.3+vol),weighted)]:
            a=x*w[:,None];b=y*w[:,None];dest.append((a.T@a/len(x),a.T@b/len(x)))
    coefs=[]
    for hist in [linear,weighted]:
        g,b=(np.mean(v,0) for v in zip(*hist))
        coefs.append(impact.solve(g,b,.50))
    return np.mean(coefs,axis=0)


def main():
    OUT.mkdir(exist_ok=True)
    old=json.loads((impact.OUT/'results.json').read_text())['16'];ow=np.array(old['weights'])
    early=[];report={}
    for end,stop in [(12,14),(14,16),(16,20)]:
        coef=fit(end);rows=[]
        for d in range(end,stop):
            raw=v1.grid(d);x,vol=design(raw)
            if d==end:np.testing.assert_allclose(x[:194],design(raw[:194])[0],atol=1e-8)
            p=v1.blend(ex.v1_base(d),np.load(impact.OUT/f'side_{d}.npy'),ow)
            y=prior.labels(d).reshape(p.shape)
            rows.append((p,y,x@coef))
        if end<16:
            early.extend(rows);p,y,q=map(np.concatenate,zip(*early))
            delta=q[:,None,:]-p.mean(1,keepdims=True)
            best=np.full(3,-np.inf);alpha=np.zeros(3)
            for a in [0,.1,.25,.4,.6,.8]:
                c=prior.corr((p+a*delta).reshape(-1,3),y.reshape(-1,3))
                idx=c>best;alpha[idx]=a;best=np.maximum(best,c)
        else:np.savez(OUT/'model.npz',coef=coef,alpha=alpha)
        entries=[]
        for d,(p,y,q) in zip(range(end,stop),rows):
            pred=p+alpha*(q[:,None,:]-p.mean(1,keepdims=True))
            np.save(OUT/f'market_{d}.npy',q.astype(np.float32))
            entries.append((p.reshape(-1,3),pred.reshape(-1,3),y.reshape(-1,3)))
        report[stop]={'alpha':alpha.tolist(),'result':prior.measure(entries)}
        (OUT/'results.json').write_text(json.dumps(report,indent=2));print('score',stop,report[stop],flush=True)

if __name__=='__main__':main()
