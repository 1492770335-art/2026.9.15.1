"""Use a single revision coefficient only when all preceding dates agree on its direction."""
import json
import numpy as np
import experiment_backbone_revision_size_v1 as previous
import replay_verified_forecast_parent_v1 as replay
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

PARENT=previous.PARENT
OUT=ART/'replicated_revision_slope_v1'


def parts(p,o):
    delta=o-p
    market=delta.mean(1,keepdims=True)
    return market,delta-market


def day_weights(p,o,y):
    # Closed-form maximum of corr(p + a*d, y), after centering over the day.
    pp=p.reshape(-1,3).astype(float);pp-=pp.mean(0)
    yy=y.reshape(-1,3).astype(float);yy-=yy.mean(0)
    pv=(pp*pp).mean(0);py=(pp*yy).mean(0)
    result=[]
    for d in parts(p,o):
        d=np.broadcast_to(d,p.shape).reshape(-1,3).astype(float)
        d-=d.mean(0)
        pd=(pp*d).mean(0);dv=(d*d).mean(0);dy=(d*yy).mean(0)
        derivative=dy*pv-py*pd
        den=py*dv-dy*pd
        a=np.divide(derivative,den,out=np.sign(derivative)*.5,where=den>1.e-12)
        result.append(a.clip(-.5,.5))
    return result


def fit():
    daily=[]
    for d in range(12,16):
        p=np.load(ART/'intraday_relative_calibration_v1'/f'pred_{d}.npy')
        o=np.load(ART/'opening_oof_cache_v1'/f'day_{d:02d}.npy')
        daily.append(day_weights(p,o,labels(d)))
    daily=np.stack(daily)
    sign=np.sign(daily)
    unanimous=(sign==sign[0]).all(0)
    # The most conservative signed coefficient still supported on every date.
    w=.8*np.min(abs(daily),axis=0)*sign[0]*unanimous
    return w,daily


def predict(p,o,w):
    market,relative=parts(p,o)
    return (p+market*w[0]+relative*w[1]).astype('float32')


def main():
    OUT.mkdir(exist_ok=True)
    w,daily=fit();np.save(OUT/'weights.npy',w)
    rows=[]
    for d in range(16,20):
        p=np.load(PARENT/f'pred_{d}.npy')
        o=np.load(ART/'opening_oof_cache_v1'/f'day_{d:02d}.npy')
        q=predict(p,o,w)
        np.save(OUT/f'pred_{d}.npy',q)
        rows.append((p.reshape(-1,3),q.reshape(-1,3),labels(d)))
    result=measure(rows)
    (OUT/'results.json').write_text(json.dumps({'20':result,'day_weights':daily.tolist(),'frozen':w.tolist()},indent=2))
    print('weights',w.tolist(),'result',result,flush=True)
    if result['gain']>0:
        folder=ART/'opening_expert_v1/commit_opening_expert'
        ev=previous.opening.Evaluator('',str(folder),'cuda:0')
        mix=np.array(json.loads((folder/'conf/Model_0.json').read_text())['opening_mix'])
        def advance(raw,p):
            (_,o),_=previous.opening.replay_features(ev,raw,mix,2)
            return predict(p,o.reshape(p.shape),w)
        replay.main(PARENT,OUT,advance)


if __name__=='__main__':
    main()
