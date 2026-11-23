"""Predict market-neutral return bands in a bid/ask symmetric coordinate system."""
import json
import numpy as np
import experiment_price_band_depth_curve_v1 as depth
import experiment_delayed_tail_band_skill_v1 as early
import experiment_pairwise_book_tree_v1 as pair
import replay_verified_forecast_parent_v1 as replay
from experiment_causal_cross_horizon_ridge_v1 import ART, labels, corr, measure

OUT = ART / 'directional_depth_tree_v1'
PARENT = ART / 'range_horizon_transfer_v1'
UNIT = np.array([.559, .272, .161], np.float32)
WIDTH = np.sqrt([5., 15., 40.]).astype('float32')


def geometry():
    with np.load(depth.OUT / 'model.npz') as data:
        return {k: data[k] for k in data.files if k not in ('market', 'relative')}


def design(raw, ticks, g):
    bands, count, cap, total, _, tick = depth.curve(raw, g)
    buy, sell, quantity, bid, ask, spread, _, _ = depth.clock.decode(raw, g)
    skew = 2*bands - 1
    # Missing inner-band observations cannot supply a signed pressure.
    skew[:, :, 1:5] *= count > 0
    ret = raw[:, :, 0].astype(np.float32) - .00136
    flow = ((buy-sell)/(quantity + .03)).astype(np.float32)
    signed = np.concatenate((skew, skew*cap, np.stack((
        flow, np.arcsinh(buy-sell), np.arcsinh(bid+ask), ret,
        np.arcsinh(np.maximum(0, raw[:,:,77].astype(float) - g['floor'][3])) -
        np.arcsinh(np.maximum(0, raw[:,:,78].astype(float) - g['floor'][4]))), axis=2)), axis=2)
    sign = np.where(skew[ticks,:,0] >= 0, 1., -1.).astype(np.float32)
    even = [np.log1p(total[ticks]), np.log1p(quantity[ticks]), np.log1p(spread[ticks].clip(0)),
            np.log(tick[ticks]), np.broadcast_to(np.exp(-ticks[:,None]/400), sign.shape)]
    pieces = [np.stack(even, axis=2), np.log1p(count[ticks]), cap[ticks]]
    cols = [signed[ticks]]
    end = ticks+1
    selected = signed[:,:,[0,1,3,5,6,10,12,13,14,15]]
    prefix = np.concatenate((np.zeros_like(selected[:1]), selected.cumsum(0, dtype=float)))
    for low, high in ((0,5),(5,20),(20,60),(60,180)):
        stop = (end-low).clip(0)
        start = (end-high).clip(0)
        cols.append((prefix[stop]-prefix[start])/(stop-start).clip(1)[:,None,None])
    for z in cols:
        pieces.extend((sign[:,:,None]*z, sign[:,:,None]*z.mean(1,keepdims=True)))
    cs = np.vstack((np.zeros(50), np.cumsum(ret**2, axis=0,dtype=float)))
    left = (end-80).clip(0)
    risk = (.45+.55*np.sqrt((8+cs[end]-cs[left])/(8+end-left)[:,None])).clip(.6,3)
    pieces.append(np.log(risk)[:,:,None])
    x = np.concatenate(pieces,axis=2).astype('float32')
    return x.reshape(-1,x.shape[-1]), sign*risk.astype('float32')


def fit(end, g):
    xs, ys = [], []
    for d in range(end):
        raw = pair.rawday(d)
        t = np.r_[np.arange(d%4,600,4),np.arange(600+d%14,len(raw)-61,14)]
        x,scale = design(raw,t,g)
        y = labels(d).reshape(-1,50,3)[t]
        y = (y-y.mean(1,keepdims=True))/UNIT
        bands = np.diff(y, axis=2, prepend=np.zeros_like(y[:,:,:1]))/WIDTH
        xs.append(x)
        ys.append((bands/scale[:,:,None]).reshape(-1,3).clip(-5,5))
    x, y = np.concatenate(xs), np.concatenate(ys)
    del xs, ys
    models = []
    for j in range(3):
        train = pair.lgb.Dataset(x, label=y[:,j])
        config = dict(objective='huber',alpha=1.4,max_bin=47,num_threads=2,num_leaves=23,
                      min_data_in_leaf=240,learning_rate=.055,lambda_l2=25,feature_fraction=.8,
                      bagging_fraction=.85,bagging_freq=3,force_col_wise=True,verbosity=-1,seed=737+j)
        model = pair.lgb.train(config, train, num_boost_round=140)
        models.append(model)
        if end==16:
            model.save_model(str(OUT / f'band_{j}.txt'))
    print('fitted through day', end, flush=True)
    return models


def signal(raw, pred, models, g):
    x, scale = design(raw, np.arange(len(raw)), g)
    band = np.stack([m.predict(x,num_threads=2).reshape(scale.shape)*scale for m in models],axis=2)
    band -= band.mean(1,keepdims=True)
    return pair.innovation(pred, band)


def moments(z, p, y):
    pred = p.reshape(-1,3).astype(float)
    true = y.reshape(-1,3).astype(float)
    pred -= pred.mean(0)
    true -= true.mean(0)
    slope = ((pred*true).sum(0)/(pred**2).sum(0)).clip(.4,2.5)
    x = z.reshape(-1,3).astype(float)
    return x.T@x/len(x), x.T@(true/slope-pred)/len(x)


def main():
    OUT.mkdir(exist_ok=True)
    g = geometry()
    history, report = [], {}
    for end, stop in ((12,14),(14,16),(16,20)):
        if end==16 and not gate.any():
            break
        models = fit(end,g)
        if end>=14:
            w = pair.weights(history)
            if end==16:
                w *= gate
                np.save(OUT/'weights.npy',w)
        daily, rows = [], []
        for d in range(end,stop):
            p = np.load(PARENT/f'pred_{d}.npy') if end==16 else early.base(d)
            raw = pair.rawday(d)
            z = signal(raw,p,models,g)
            if d==end:
                np.testing.assert_allclose(z[:91],signal(raw[:91],p[:91],models,g),atol=5e-6)
            if end>=14:
                q = (p+z@w).astype('float32')
                if end==16:
                    np.save(OUT/f'pred_{d}.npy',q)
            y = labels(d)
            if end<16:
                history.append(moments(z,p,y))
            if end>=14:
                row=(p.reshape(-1,3),q.reshape(-1,3),y)
                rows.append(row)
                daily.append(corr(row[1],y)-corr(row[0],y))
        if end>=14:
            report[str(stop)]=measure(rows)
            if end==14:
                gate=np.min(daily,axis=0)>0
                report['eligible']=gate.tolist()
            print(stop,report[str(stop)],flush=True)
            (OUT/'results.json').write_text(json.dumps(report,indent=2))
    if report.get('20',{}).get('gain',0)>0:
        replay.main(PARENT,OUT,lambda raw,p:(p+signal(raw,p,models,g)@w).astype('float32'))


if __name__=='__main__':
    main()
