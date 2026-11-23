"""Track the cost and age of a hypothetical net flow position using observable executions."""
import json
import numpy as np
import experiment_execution_location_markouts_v1 as location
import experiment_pairwise_book_tree_v1 as book
import experiment_consensus_horizon_portfolios_v1 as blend
import experiment_refresh_cohort_transfer_v1 as cohort
import replay_refresh_y3_specialist_v1 as specialist
from experiment_book_transition_rewards_v1 import replay
from experiment_causal_cross_horizon_ridge_v1 import ART,measure
from experiment_eight_date_expert_contrasts_v1 import fit,moments

OUT=ART/'netted_execution_cost_v1'


def decode_geometry():
    with np.load(location.OUT/'model.npz') as st:
        return {k:st[k] for k in st.files if k not in ('relative','market')}


def state(raw,geometry,netted):
    r,q,_,execution,_,sign,_,_,_,volume=location.values(raw,geometry)
    price=np.cumsum(r,axis=0)
    trade=price+execution
    signed=volume*sign
    n,stocks=price.shape
    life=np.array([16.,80.,320.])[None,:]
    shape=(stocks,3)
    qty=np.zeros(shape)
    basis=np.zeros(shape)
    times=np.zeros(shape)
    buy=np.zeros(shape)
    sell=np.zeros(shape)
    buy_value=np.zeros(shape)
    sell_value=np.zeros(shape)
    buy_time=np.zeros(shape)
    sell_time=np.zeros(shape)
    result=np.empty((n,stocks,3,5),np.float32)
    for t in range(n):
        px=trade[t,:,None]
        here=price[t,:,None]
        delta=signed[t,:,None]
        vol=volume[t,:,None]
        # A modest time decay also expires positions during inactive stretches.
        decay=np.exp(-(.25+vol.clip(0,12))/life)
        qty*=decay
        if netted:
            same=(qty*delta>=0)
            after=qty+delta
            flip=(after*qty<0)
            weight=np.divide(abs(delta),abs(after),out=np.zeros_like(qty),where=same&(abs(after)>1e-8))
            basis=np.where(flip,px,basis+weight*(px-basis))
            times=np.where(flip,t,times+weight*(t-times))
            qty=after
            distance=here-basis
            age=t-times
        else:
            added_buy=np.maximum(delta,0)
            added_sell=np.maximum(-delta,0)
            buy=buy*decay+added_buy
            sell=sell*decay+added_sell
            buy_value=buy_value*decay+added_buy*px
            sell_value=sell_value*decay+added_sell*px
            buy_time=buy_time*decay+added_buy*t
            sell_time=sell_time*decay+added_sell*t
            qty=buy-sell
            active=qty>=0
            amount=np.where(active,buy,sell)
            avg=np.divide(np.where(active,buy_value,sell_value),amount,out=here.repeat(3,axis=1),where=amount>1e-8)
            when=np.divide(np.where(active,buy_time,sell_time),amount,out=np.full(shape,float(t)),where=amount>1e-8)
            distance=here-avg
            age=t-when
        normalized=np.tanh(qty/(.6*np.sqrt(life)))
        movement=np.tanh(distance/(.15+q[t,:,None]*np.sqrt(1+age.clip(0)/5)))
        profit=np.tanh(np.sign(qty)*distance/(.35+q[t,:,None]))
        result[t]=np.stack((normalized,normalized*profit,normalized*np.maximum(profit,0),
                            movement*abs(normalized),normalized*np.log1p(age.clip(0))/5),axis=2)
    return result.reshape(n,stocks,-1)


def features(s,p):
    m=s.mean(1,keepdims=True)
    far=p[:,:,2:]-blend.panel.RATIO*p[:,:,1:2]
    f=np.concatenate((s,(s-m)*np.tanh(5*far)*np.sign(s),s*np.tanh(abs(far)*5)),axis=2)
    a=f.mean(1)
    return a,f-a[:,None]


def main():
    OUT.mkdir(exist_ok=True)
    geo=cohort.geometry()
    decode=decode_geometry()
    histories=[[],[]]
    prior=[]
    bh=[]
    daily=[]
    folds=[]
    for end in (8,10,12,14):
        c=blend.fixed_weight(prior)
        pw=np.stack(fit(bh)) if bh else None
        if pw is not None:pw[:,:,:2]=0
        weights=[fit(h) if h else None for h in histories]
        rows=[[],[]]
        for d in range(end,end+2):
            raw=book.rawday(d)
            p,y,fs,s=blend.training_day(d,prior)
            prior.append(s)
            p=blend.parent.fixed_forecast(p,fs,c)
            z=cohort.features(raw,p,geo,False)
            bh.append(moments(p[::4],y[::4],tuple(a[::4] for a in z)))
            if pw is not None:p=cohort.advance(p,raw,geo,pw)
            for j,netted in enumerate((False,True)):
                s=state(raw,decode,netted)
                m,r=features(s,p)
                if weights[j] is not None:
                    a,b=weights[j]
                    q=(p+(m@a)[:,None]+r@b).astype('float32')
                    if d==end:
                        sm,sr=features(state(raw[:203],decode,netted),p[:203])
                        np.testing.assert_allclose(q[:203],p[:203]+(sm@a)[:,None]+sr@b,atol=2e-6,rtol=0)
                    row=tuple(v.reshape(-1,3) for v in (p,q,y))
                    rows[j].append(row)
                    if j:
                        sc=measure([row])
                        daily.append(np.subtract(sc['candidate'],sc['base']))
                histories[j].append(moments(p[::4],y[::4],(m[::4],r[::4])))
        if rows[0]:
            f={k:measure(group) for k,group in zip(('separate','netted'),rows)}
            folds.append(f)
            print(end+1,[v['gain'] for v in f.values()],flush=True)
    gains=np.stack([np.subtract(f['netted']['candidate'],f['netted']['base']) for f in folds])
    adv=np.stack([np.subtract(f['netted']['candidate'],f['separate']['candidate']) for f in folds])
    gate=(gains.min(0)>0)&(adv.mean(0)>0)&((np.stack(daily)>0).sum(0)>=4)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,retained=gate.tolist()),indent=2))
    if gate.any():
        w=np.stack(fit(histories[1]))*gate
        np.save(OUT/'weights.npy',w)
        base=specialist.load()
        def forward(raw,p):
            q=base(raw,p)
            m,r=features(state(raw,decode,True),q)
            return (q+(m@w[0])[:,None]+r@w[1]).astype('float32')
        s=replay(forward)
        s['gain_over_specialist']=s['pooled']['score']-json.loads((specialist.OUT/'raw_replay.json').read_text())['pooled']['score']
        (OUT/'raw_replay.json').write_text(json.dumps(s,indent=2))
        print(s['pooled'],s['gain_over_specialist'],flush=True)

if __name__=='__main__':
    main()
