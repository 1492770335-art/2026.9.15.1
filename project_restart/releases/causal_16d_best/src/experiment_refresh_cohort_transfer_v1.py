"""Transfer forecast changes between recently repriced and quiet stocks with matched long-run risk."""
import json
import numpy as np
import experiment_pairwise_book_tree_v1 as book
import experiment_consensus_horizon_portfolios_v1 as blend
from experiment_causal_cross_horizon_ridge_v1 import ART, measure
from experiment_eight_date_expert_contrasts_v1 import moments, fit
from experiment_book_transition_rewards_v1 import replay

OUT=ART/'refresh_cohort_transfer_v1'


def geometry():
    activity=[]
    risk=[]
    betas=[]
    for d in range(8):
        raw=book.rawday(d)
        r=raw[:,:,0].astype(float)-.00136
        activity.append((abs(r)>.022).mean(0))
        risk.append(r.std(0))
        b=r[:len(r)//24*24].reshape(-1,24,50).sum(1)
        m=b.mean(1)
        betas.append((b*m[:,None]).mean(0)/(m*m).mean())
    activity=np.median(activity,axis=0).clip(.015,.95)
    features=np.column_stack((np.log(np.median(risk,axis=0)),np.median(betas,axis=0),np.log(activity)))
    features=(features-features.mean(0))/features.std(0).clip(.01)
    distance=((features[:,None]-features[None])**2).sum(2)
    w=np.exp(-.32*distance)
    np.fill_diagonal(w,0)
    w/=w.sum(1,keepdims=True)
    # Matched random pairs retain the marginal quote clocks but break the association with each stock.
    np.fill_diagonal(distance,np.inf)
    plus=np.random.default_rng(13280).permutation(50)
    avail=set(range(50))
    order=np.empty(50,int)
    for i in plus:
        if i not in avail:continue
        avail.remove(i)
        j=min(avail,key=lambda a:distance[i,a])
        avail.remove(j)
        order[i],order[j]=j,i
    return activity.astype('float32'),w.astype('float32'),order


def features(raw,p,geom,placebo):
    activity,w,order=geom
    r=raw[:,:,0].astype('float32')-.00136
    t=np.arange(len(raw))[:,None]
    last=np.maximum.accumulate(np.where(abs(r)>.022,t,0),axis=0)
    freshness=np.exp(-(t-last)*activity[None]*.75).astype('float32')
    if placebo:freshness=freshness[:,order]
    x=p.copy()
    x[:,:,2]-=blend.panel.RATIO*p[:,:,1]
    x[:,:,1]-=.272/.559*p[:,:,0]
    lag=(t[:,0]-8).clip(0)
    streams=np.concatenate((x,x-x[lag],np.tanh(r[:,:,None])),axis=2)
    recent=w@((.12+freshness)[:,:,None]*streams)
    recent/=w@(.12+freshness)[:,:,None]
    quiet=w@((1.12-freshness)[:,:,None]*streams)
    quiet/=w@(1.12-freshness)[:,:,None]
    delta=recent-quiet
    own=(1-freshness)[:,:,None]
    side=np.tanh(.7*(raw[:,:,41].astype('float32')-raw[:,:,42].astype('float32')))
    direction=np.tanh(10*delta[:,:,:3]*x)
    z=np.concatenate((delta,own*delta,delta[:,:,:3]*direction,
                      own*delta[:,:,:3]*np.tanh(4*side[:,:,None]*x)),axis=2)
    m=z.mean(1)
    return m,z-m[:,None]


def advance(p,raw,geo,w,placebo=False):
    m,r=features(raw,p,geo,placebo)
    return (p+(m@w[0])[:,None]+r@w[1]).astype('float32')


def main():
    OUT.mkdir(exist_ok=True)
    geo=geometry()
    hist=[[],[]]
    prior=[]
    folds=[]
    daily=[]
    for end in (8,10,12,14):
        c=blend.fixed_weight(prior)
        ws=[fit(h) if h else None for h in hist]
        rows=[[],[]]
        for d in range(end,end+2):
            raw=book.rawday(d)
            p,y,fs,stat=blend.training_day(d,prior)
            prior.append(stat)
            p=blend.parent.fixed_forecast(p,fs,c)
            for j,placebo in enumerate((True,False)):
                m,r=features(raw,p,geo,placebo)
                if ws[j] is not None:
                    a,b=ws[j]
                    q=(p+(m@a)[:,None]+r@b).astype('float32')
                    if d==end:np.testing.assert_allclose(q[:273],advance(p[:273],raw[:273],geo,ws[j],placebo),atol=1e-5,rtol=0)
                    row=tuple(v.reshape(-1,3) for v in (p,q,y))
                    rows[j].append(row)
                    if j:
                        s=measure([row])
                        daily.append(np.subtract(s['candidate'],s['base']))
                hist[j].append(moments(p[::4],y[::4],(m[::4],r[::4])))
        if rows[0]:
            f={key:measure(group) for key,group in zip(('permuted','real'),rows)}
            folds.append(f)
            print(end+1,[v['gain'] for v in f.values()],flush=True)
    gains=np.stack([np.subtract(f['real']['candidate'],f['real']['base']) for f in folds])
    edge=np.stack([np.subtract(f['real']['candidate'],f['permuted']['candidate']) for f in folds])
    gate=(gains.min(0)>0)&(edge.mean(0)>0)&((np.stack(daily)>0).sum(0)>=4)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,retained=gate.tolist()),indent=2))
    if gate.any():
        w=np.stack(fit(hist[1]))*gate
        np.savez(OUT/'model.npz',activity=geo[0],peers=geo[1],order=geo[2],w=w)
        result=replay(lambda raw,p:advance(p,raw,geo,w))
        (OUT/'raw_replay.json').write_text(json.dumps(result,indent=2))
        print(result['pooled'],flush=True)


if __name__=='__main__':
    main()
