"""Compare reversion after equal-size moves that start at different places within their prior price range."""
import json
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
import experiment_peer_matched_factor_votes_v1 as votes
import experiment_eight_date_expert_contrasts_v1 as early

base,peer=votes.base,votes.peer
OUT=peer.ART/'pre_move_range_response_v1'


def extrema(x,width):
    # Pad the left boundary so each filter ends at the observed tick.
    pad=np.pad(x,((width-1,0),(0,0)),mode='edge')
    indices=np.arange(len(x))+(width-1)-width//2+(1 if width%2==0 else 0)
    return (minimum_filter1d(pad,width,axis=0,mode='nearest')[indices],
            maximum_filter1d(pad,width,axis=0,mode='nearest')[indices])


def features(raw,t):
    r=raw[:,:,0].astype(float)-.00136
    path=votes.center(r).cumsum(0)
    sq=np.vstack((np.zeros(50),(r*r).cumsum(0)))
    lo=(t+1-96).clip(0)
    risk=np.sqrt((8+sq[t+1]-sq[lo])/(8+t+1-lo)[:,None]).clip(.45,4)
    cols=[[],[]]
    extreme=extrema(path,160)
    for lag in (20,64,160):
        old=(t-lag).clip(0)
        start=path[old]
        low,high=(v[old] for v in extreme)
        live_low,live_high=(v[t] for v in extrema(path,lag+1))
        unit=risk*np.sqrt(lag)
        travel=(path[t]-start)/unit
        half=(high-low)/2
        for j,(left,right) in enumerate(((start-half,start+half),(low,high))):
            left,right=(left-start)/unit,(right-start)/unit
            escaped=(travel-right).clip(0)+(travel-left).clip(None,0)
            absorbed=travel-escaped
            overshoot=((live_high-start)/unit-right).clip(0)
            undershoot=(left-(live_low-start)/unit).clip(0)
            giveback=overshoot-undershoot-escaped
            # A failed escape in either direction is visible by the forecast origin.
            terms=[np.tanh(z) for z in (travel,escaped,absorbed,giveback,overshoot-undershoot)]
            terms.extend([np.tanh(escaped)*np.tanh(abs(travel)),
                          np.tanh(giveback)*np.tanh(half/unit),
                          np.tanh(travel)*np.tanh(abs(escaped)),
                          np.tanh(escaped)*np.log(risk)])
            cols[j].extend(terms)
    return [votes.center(np.stack(z,axis=2)).astype('float32') for z in cols]


def stats(raw,t,y):
    target=votes.center(y).reshape(-1,3).astype(float)
    rows=[]
    for x in features(raw,t):
        a=x.reshape(-1,x.shape[2]).astype(float)
        rows.append((a.T@a/len(a),a.T@target/len(a)))
    return rows


def fit(history):
    result=[]
    for part in zip(*history):
        gs,rs=(np.stack(v) for v in zip(*part))
        g=gs.mean(0)
        reg=.75*np.diag(g.diagonal().clip(.0001))
        by_day=np.linalg.solve(gs+reg,rs)
        result.append(.7*np.linalg.solve(g+reg,rs.mean(0))+.3*np.median(by_day,axis=0))
    return result


def change(raw,t,weights):
    plain,ranged=(x@w for x,w in zip(features(raw,t),weights))
    return (.25*votes.center(ranged-plain)).astype('float32')


def screen():
    history=[];folds=[];days=[]
    for end in (8,10,12,14):
        for d in range(len(history),end):
            raw=base.paths.book.rawday(d);t=np.arange(d%6,len(raw)-60,6)
            history.append(stats(raw,t,peer.labels(d).reshape(-1,50,3)[t]))
        w=fit(history);rows=[[],[]]
        for d in (end,end+1):
            raw=base.paths.book.rawday(d);t=np.arange(d%4,len(raw),4)
            z=change(raw,t,w);y=peer.labels(d).reshape(-1,50,3)[t]
            if d==end:
                st=t[t<385]
                np.testing.assert_allclose(z[:len(st)],change(raw[:385],st,w),atol=2e-6,rtol=0)
            for group,ref in zip(rows,(peer.reference,early.baseline)):
                p=ref(d).reshape(-1,50,3)[t]
                group.append(tuple(v.reshape(-1,3) for v in (p,p+z,y)))
            days.append([peer.delta(g[-1:]) for g in rows])
        folds.append([peer.delta(g) for g in rows])
        print(end+1,folds[-1],flush=True)
    f=np.array(folds)
    keep=(f[:,0].min(0)>0)&(f[:,1].mean(0)>0)&((np.array(days)[:,0]>0).sum(0)>=5)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=days,retained=keep.tolist()),indent=2))
    if keep.any():
        for d in (14,15):
            raw=base.paths.book.rawday(d);t=np.arange(d%6,len(raw)-60,6)
            history.append(stats(raw,t,peer.labels(d).reshape(-1,50,3)[t]))
        np.savez(OUT/'model.npz',weight=fit(history),mask=keep)
    return keep.any()


def load():
    parent=base.load()
    with np.load(OUT/'model.npz') as f:w,mask=f['weight'],f['mask']
    def predict(raw,p):
        q=parent(raw,p).copy()
        for start in range(0,len(q),600):
            t=np.arange(start,min(start+600,len(q)))
            q[t]+=change(raw,t,w)*mask
        return q
    return predict


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    # Validate the alignment of odd and even windows against the literal causal range.
    a=np.random.default_rng(713).normal(size=(195,3))
    for k in (21,64,160):
        lo,hi=extrema(a,k)
        for i in (0,1,59,125,194):
            win=a[max(0,i-k+1):i+1]
            np.testing.assert_array_equal(lo[i],win.min(0));np.testing.assert_array_equal(hi[i],win.max(0))
    if screen():votes.follow.replay_candidate(OUT,load)
