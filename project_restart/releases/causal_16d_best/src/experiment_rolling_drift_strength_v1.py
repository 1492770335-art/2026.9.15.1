"""Choose the strength of time-varying forecast coefficients using six preceding out-of-fold dates."""
import json
import numpy as np
import experiment_forecast_coefficient_dynamics_v1 as drift
import experiment_supervised_gradient_kernel_v1 as kernel
import experiment_delayed_tail_band_skill_v1 as ref
import experiment_causal_price_level_centering_v1 as readout
import experiment_pairwise_book_tree_v1 as pair
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,corr,measure

OUT=ART/'rolling_drift_strength_v1'
GRID=np.array([0.,.5,1.,1.5,2.,3.,4.5,6.,9.,12.])


def gains(p,y,term):
    a=p.reshape(-1,3).astype(float);b=y.reshape(-1,3).astype(float)
    c=np.broadcast_to(term,p.shape).reshape(-1,3).astype(float)
    a-=a.mean(0);b-=b.mean(0);c-=c.mean(0)
    u=(a*b).sum(0);v=(c*b).sum(0);aa=(a*a).sum(0);ac=(a*c).sum(0);cc=(c*c).sum(0);bb=(b*b).sum(0)
    g=GRID[:,None]
    ic=(u+g*v)/np.sqrt(bb*(aa+2*g*ac+g*g*cc))
    return ic-ic[0]


def choose(history,gate):
    perf=np.array(history)
    multipliers=np.ones((2,3))
    decisions=[]
    for part,j in zip(*np.nonzero(gate)):
        samples=perf[:,part,:,j]
        picks=[]
        for drop in range(len(samples)):
            keep=np.delete(samples,drop,axis=0)
            utility=.48*np.median(keep,axis=0)+.32*keep.mean(0)+.20*np.quantile(keep,.25,axis=0)
            picks.append(GRID[utility.argmax()])
        # Half of a proposed change remains anchored to the unscaled model on the log scale.
        mult=np.sqrt(max(.25,float(np.median(picks))))
        if not np.isfinite(mult):raise ValueError('invalid multiplier')
        multipliers[part,j]=mult
        decisions.append(dict(part=int(part),target=int(j),leave_day_out=picks,chosen=mult,mean_gain=samples.mean(0).tolist()))
    return multipliers,decisions


def main():
    OUT.mkdir(exist_ok=True)
    geo=kernel.geometry();training=[];mom=[];history=[]
    with np.load(drift.OUT/'model.npz') as state:
        gate=np.any(state['coef']!=0,axis=1)
    for end,stop in ((8,10),(10,12),(12,14),(14,16)):
        training.extend(drift.stats(d,geo) for d in range(len(training),end))
        w=drift.changing(training)
        if end>=10:coef=readout.fit(mom)
        for day in range(end,stop):
            p=ref.base(day);raw=pair.rawday(day);delta=kernel.predict(raw,geo,w)
            m=delta.mean(1);r=pair.innovation(p,delta-m[:,None])
            y=labels(day).reshape(p.shape)
            if end>=10:history.append([gains(p,y,(m@coef[0])[:,None]),gains(p,y,r@coef[1])])
            mom.append(readout.moments(p,y,(m,r)))
    multi,decisions=choose(history,gate)
    np.save(OUT/'multiplier.npy',multi)
    print('frozen multipliers',multi.tolist(),flush=True)
    _,w,coef=kernel.load_fit(drift.OUT)
    rows=[];daily={}
    for day in range(16,20):
        p=np.load(kernel.OUT/f'pred_{day}.npy');raw=pair.rawday(day)
        delta=kernel.predict(raw,geo,w);m=delta.mean(1);r=pair.innovation(p,delta-m[:,None])
        a=(m@coef[0])[:,None];b=r@coef[1]
        parent=np.load(drift.OUT/f'pred_{day}.npy')
        np.testing.assert_allclose(p+a+b,parent,atol=2e-6)
        q=(p+a*multi[0]+b*multi[1]).astype('float32');np.save(OUT/f'pred_{day}.npy',q)
        y=labels(day);row=tuple(z.reshape(-1,3) for z in (parent,q,y));rows.append(row);daily[day+1]=measure([row])
    score=measure(rows)
    (OUT/'results.json').write_text(json.dumps({'20':score,'daily':daily,'grid':GRID.tolist(),'earlier_dates':decisions},indent=2))
    print(score,flush=True)


def refinement():
    _,first=kernel.refinement()
    geo,w,coef=kernel.load_fit(drift.OUT);mul=np.load(OUT/'multiplier.npy')
    def refine(raw,p):
        base=first(raw,p);side=kernel.predict(raw,geo,w);m=side.mean(1)
        r=pair.innovation(base,side-m[:,None])
        return (base+(m@coef[0])[:,None]*mul[0]+(r@coef[1])*mul[1]).astype('float32')
    return OUT,refine


if __name__=='__main__':
    import sys
    if '--replay' in sys.argv:
        import replay_paired_book_trees_v1 as replay
        replay.main(refinement())
    else:main()
