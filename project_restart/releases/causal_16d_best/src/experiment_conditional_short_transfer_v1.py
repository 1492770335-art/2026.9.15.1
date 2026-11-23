"""Estimate how a short-horizon correction carries across longer horizons in different liquidity states."""
import json
import numpy as np
import experiment_delayed_tail_band_skill_v1 as early
import experiment_pairwise_book_tree_v1 as book
import replay_verified_forecast_parent_v1 as replay
from experiment_causal_cross_horizon_ridge_v1 import ART, labels, measure

ORIGIN=ART/'range_horizon_transfer_v1'
PARENT=ART/'session_restart_v1'
OUT=ART/'conditional_short_transfer_v1'


def states(raw):
    r=raw[:,:,0].astype(float)-.00136
    t=np.arange(len(r))+1
    start=(t-96).clip(0)
    def running(values):
        cs=np.concatenate((np.zeros_like(values[:1]),values.cumsum(0)))
        return (cs[t]-cs[start])/(t-start)[:,None]
    active=running((abs(r)>.018).astype(float))
    center=r.mean(1,keepdims=True)
    share=running(center**2)/(.02+running((r*r).mean(1,keepdims=True)))
    # Knots are fixed across all dates; each interval is available from its observed prefix.
    return np.minimum(3,(share/.10).astype(int)), np.minimum(3,(active/.16).astype(int))


def fit():
    byday=[]
    for d in range(8,16):
        p=early.base(d).astype(float);y=labels(d).reshape(p.shape)
        pm=p.mean(1,keepdims=True);ym=y.mean(1,keepdims=True)
        day=[]
        for state,a,b in zip(states(book.rawday(d)),(pm,p-pm),(ym,y-ym)):
            a=a.reshape(-1,3);b=b.reshape(-1,3)
            a=a-a.mean(0);b=b-b.mean(0)
            slope=((a*b).sum(0)/(a*a).sum(0)).clip(.4,2.5)
            error=b/slope-a
            gram=error.T@error/len(error)
            coeff=[]
            for k in range(4):
                e=error[state.ravel()==k]
                reg=gram*.12
                g=reg+e.T@e/len(error)
                coeff.append(g[0,1:]/(g[0,0]*1.035))
            day.append(coeff)
        byday.append(day)
    return np.median(byday,axis=0)


def advance(raw,p,origin,w):
    change=p[:,:,0]-origin[:,:,0]
    shared=change.mean(1,keepdims=True)
    q=p.copy()
    for part,state,coef in zip((shared,change-shared),states(raw),w):
        q[:,:,1:]+=part[:,:,None]*coef[state]
    return q


def main():
    OUT.mkdir(exist_ok=True)
    w=fit()
    np.save(OUT/'coefficients.npy',w)
    rows=[]
    for d in range(16,20):
        p=np.load(PARENT/f'pred_{d}.npy');origin=np.load(ORIGIN/f'pred_{d}.npy')
        assert np.array_equal(p[:,:,1:],origin[:,:,1:])
        q=advance(book.rawday(d),p,origin,w)
        np.save(OUT/f'pred_{d}.npy',q)
        rows.append((p.reshape(-1,3),q.reshape(-1,3),labels(d)))
    result=measure(rows)
    (OUT/'results.json').write_text(json.dumps({'20':result},indent=2))
    print(result,flush=True)
    if result['gain']>0:
        replay.main(PARENT,OUT,lambda raw,p,origin:advance(raw,p,origin,w),(ORIGIN,))


if __name__=='__main__':
    main()
