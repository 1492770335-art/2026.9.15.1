"""Scale market and stock-relative forecasts using causal multiscale variance ratios."""
import json
import numpy as np
import experiment_endpoint_innovation_v1 as inn

prior=inn.prior
OUT=prior.ART/'variance_ratio_calibration_v1'

def baseline(d):
    if d<12:
        return inn.forecast(d).astype(np.float32)
    name='phase_future_book_v1' if d<14 else 'phase_conditional_blend_v1'
    return np.load(prior.ART/name/f'pred_{d}.npy')

def states(raw):
    r=raw[:,:,0].astype(float)-.00136
    t=np.arange(len(r))
    market=r.mean(1,keepdims=True)
    def estimators(series):
        cols=[]
        for w in (96,640):
            length=np.minimum(t+1,w)[:,None]
            single=(8.+inn.moving_sum(series**2,w))/(8.+length)
            long=inn.moving_sum(series,12)/np.sqrt(12.)
            multi=(8.+inn.moving_sum(long**2,w))/(8.+length)
            cols.extend([(.5*np.log(single.clip(.025,15))).clip(-1.7,1.6),
                         (.5*np.log(multi/single)).clip(-1.35,1.35)])
        return cols
    stock=estimators(r)
    market_state=estimators(market)
    absolute=stock[0]
    relative_vol=absolute-absolute.mean(1,keepdims=True)
    op=np.exp(-t[:,None]/350)
    stock += [relative_vol,op*relative_vol, op*stock[1],stock[1]*relative_vol]
    market_state += [absolute.mean(1,keepdims=True),stock[1].mean(1,keepdims=True),
                     op,op*market_state[1]]
    return stock,market_state

def design(p, s, j):
    rel,market=s
    v=p[:,:,j].astype(float)
    m=v.mean(1,keepdims=True)
    a=v-m
    relative=[a*x for x in rel]
    # Separate the correlation of the market portfolio from within-market stock ranking.
    loc=[x-x.mean(1,keepdims=True) for x in relative]
    common=[np.broadcast_to(m*x,v.shape) for x in market]
    common.append(np.broadcast_to(m,v.shape))
    return np.stack(loc+common,axis=-1).astype('float32')

def stats(x,p,y):
    x=x[::3].reshape(-1,x.shape[-1]).astype(float)
    p=p[::3].flatten().astype(float)
    y=y[::3].flatten().astype(float)
    y-=y.mean();p-=p.mean();x-=x.mean(0)
    x-=np.outer(p,(p@x)/(p@p))
    scale=np.clip((y@p)/(p@p),.4,2.5)
    return x.T@x/len(x),x.T@(y/scale-p)/len(x)

def fit(hist):
    out=[]
    for j in range(3):
        g,b=[np.mean(parts,0) for parts in zip(*[day[j] for day in hist])]
        reg=.9*np.diag(g.diagonal().clip(2e-6,None))
        mean=np.linalg.solve(g+reg,b)
        daily=np.array([np.linalg.solve(day[j][0]+reg,day[j][1]) for day in hist])
        out.append((.55*mean+.45*np.median(daily,axis=0)).clip(-.5,.5)*.65)
    return np.stack(out)

def apply(raw,p,coef):
    s=states(raw)
    return np.stack([p[:,:,j]+design(p,s,j)@coef[j] for j in range(3)],axis=-1).astype(np.float32)

def main():
    OUT.mkdir(exist_ok=True);hist=[];report={}
    for d in range(8,20):
        if d in (12,14,16):
            if d==16 and min(report['14']['gain'],report['16']['gain'])<=0:break
            coef=fit(hist);entries=[]
            if d==16:np.save(OUT/'coef.npy',coef)
        p=baseline(d);raw=inn.grid(d);s=states(raw);y=prior.labels(d).reshape(p.shape)
        daily=[];q=p.copy()
        for j in range(3):
            x=design(p,s,j)
            if d<16:daily.append(stats(x,p[:,:,j],y[:,:,j]))
            if d>=12:q[:,:,j]+=x@coef[j]
        del x
        if d<16:hist.append(daily)
        if d>=12:
            if d in (12,16):
                np.testing.assert_allclose(q[:231],apply(raw[:231],p[:231],coef),atol=3e-6)
            entries.append(tuple(a.reshape(-1,3) for a in (p,q,y)))
            np.save(OUT/f'pred_{d}.npy',q)
        if d in (13,15,19):
            report[str(d+1)]=prior.measure(entries)
            (OUT/'results.json').write_text(json.dumps(report,indent=2))
            print(d+1,report[str(d+1)],flush=True)

if __name__=='__main__':main()
