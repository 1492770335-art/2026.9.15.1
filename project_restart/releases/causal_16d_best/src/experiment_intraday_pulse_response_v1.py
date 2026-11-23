"""Identify synchronous repricing online and estimate the subsequent conditional return."""
import argparse
import json
import numpy as np
import experiment_observed_boundary_response_v1 as post
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'intraday_pulse_response_v1'
WIDTH=80
START=192


def intensity(raw):
    r=np.abs(raw[:,:,0].astype(float)-.00136)
    t=np.arange(len(r))
    cs=np.vstack((np.zeros(50),r.cumsum(0)))
    left=(t-160).clip(0)
    level=(cs[t]-cs[left]+8)/(t-left+16)[:,None]
    size=(r/level.clip(.12)).clip(0,8).mean(1)
    breadth=(r>.022).mean(1)
    return size*np.sqrt(breadth)


def fit_cut():
    vals=[]
    for d in range(8):
        a=intensity(post.before.book.rawday(d))
        t=np.arange(len(a))
        vals.append(a[(t>=START)&(abs(t-post.before.BOUNDARY)>150)])
    return float(np.quantile(np.concatenate(vals),.985))


def detect(raw,cut):
    accept=[]
    last=-WIDTH
    for tick in np.flatnonzero(intensity(raw)>cut):
        if tick<START or abs(tick-post.before.BOUNDARY)<=150:continue
        if tick-last>=WIDTH:
            accept.append(tick)
            last=tick
    return np.array(accept,dtype=int)


def subset(raw,b,t):
    lo=b-180
    return post.features(raw[lo:t[-1]+1],t-lo,b-lo)


def stats(raw,y,origins):
    xs=[[],[]];ys=[[],[]]
    for b in origins:
        t=np.arange(b,b+WIDTH,3)
        m=y[t].mean(1,keepdims=True)
        for x,z,a,v in zip(subset(raw,b,t),(y[t]-m,m),xs,ys):
            a.append(x.reshape(-1,x.shape[-1]));v.append(z.reshape(-1,3))
    out=[]
    for x,z in zip(xs,ys):
        a=np.concatenate(x);b=np.concatenate(z)
        out.append((a.mean(0),b.mean(0),a.T@a/len(a),a.T@b/len(a)))
    return out


def record(d,cut):
    raw=post.before.book.rawday(d)
    y=labels(d).reshape(-1,50,3)
    events=detect(raw,cut)
    events=events[events+WIDTH+60<len(raw)]
    if not len(events):raise ValueError('no training pulses')
    # Controls are 120 ticks from events; matching clock offsets keeps local seasonality similar.
    # Their future outcomes have no role in choosing the controls.
    centers=np.where(events-120>=START,events-120,events+120)
    return (stats(raw,y,centers),stats(raw,y,events)),len(events)


def changes(raw,t,model,cut):
    out=[np.zeros((len(t),50,3),np.float32),np.zeros((len(t),1,3),np.float32)]
    for b in detect(raw,cut):
        use=(t>=b)&(t<b+WIDTH)
        if not use.any():continue
        for i,(x,z) in enumerate(zip(subset(raw,b,t[use]),out)):
            pred=[(x-a[i][0])@a[i][1] for a in model]
            delta=.20*(pred[1]-pred[0]).clip(-2,2)
            if i==0:delta-=delta.mean(1,keepdims=True)
            z[use]=delta
    return out


def load():
    parent=post.joint.load()
    with np.load(OUT/'model.npz') as st:
        model=[[(st[f'c{i}{j}'],st[f'w{i}{j}']) for j in range(2)] for i in range(2)]
        mask=st['mask'];cut=float(st['cut'])
    def predict(raw,p):
        q=parent(raw,p).copy()
        for z,k in zip(changes(raw,np.arange(len(raw)),model,cut),mask):q+=z*k
        return q
    return predict


def screen():
    OUT.mkdir(exist_ok=True)
    cut=fit_cut()
    days,counts,daily,folds=[],[],[],[]
    for end in (8,10,12,14):
        for d in range(len(days),end):
            rec,n=record(d,cut);days.append(rec);counts.append(n)
        pars=post.before.fit(days);groups=[[],[]]
        for d in (end,end+1):
            raw=post.before.book.rawday(d);t=np.arange(len(raw))
            y=labels(d).reshape(-1,50,3)
            p=np.load(post.before.parent_model.block.segment.best.refit.OUT/f'reference_{d}.npy',mmap_mode='r').copy()
            p[:,:,1]=np.load(post.before.parent_model.linear.earlier.OUT/f'y2_day_{d}.npy',mmap_mode='r')
            corr=changes(raw,t,pars,cut)
            events=detect(raw,cut)
            for length in (events[0]+1,events[0]+17):
                for a,b in zip(changes(raw[:length],t[:length],pars,cut),corr):
                    np.testing.assert_allclose(a,b[:length],rtol=0,atol=2e-6)
            for rs,z in zip(groups,corr):rs.append(tuple(a.reshape(-1,3) for a in (p,p+z,y)))
            daily.append([post.before.parent_model.linear.gains(rs[-1:]) for rs in groups])
        folds.append([post.before.parent_model.linear.gains(rs) for rs in groups])
        print(end+1,'IC gains local/market',folds[-1],flush=True)
    keep=(np.min(folds,axis=0)>0)&((np.array(daily)>0).sum(0)>=5)
    for d in range(len(days),16):
        rec,n=record(d,cut);days.append(rec);counts.append(n)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=daily,retained=keep.tolist(),counts=counts,cut=cut),indent=2))
    if keep.any():
        st=dict(mask=keep,cut=cut)
        for i,part in enumerate(post.before.fit(days)):
            for j,(c,w) in enumerate(part):st.update({f'c{i}{j}':c,f'w{i}{j}':w})
        np.savez(OUT/'model.npz',**st)


def replay():
    if not np.any(json.loads((OUT/'results.json').read_text())['retained']):return
    baseline=json.loads((post.joint.OUT/'reload_verification.json').read_text())['pooled']['score']
    for name in ('raw_replay','reload_verification'):
        res=post.before.parent_model.linear.best.online.best.roll.gls.best.walk.replay(load())
        if name=='reload_verification':
            old=json.loads((OUT/'raw_replay.json').read_text())
            np.testing.assert_allclose(res['pooled']['candidate'],old['pooled']['candidate'],rtol=0,atol=1e-10)
        (OUT/f'{name}.json').write_text(json.dumps(res,indent=2))
        print(name,res['pooled']['score'],flush=True)
        if res['pooled']['score']<=baseline:break


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--replay',action='store_true')
    if parser.parse_args().replay:replay()
    else:screen()
