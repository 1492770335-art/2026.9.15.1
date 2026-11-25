"""Train an interval readout on expanding dates and score its disjoint return contributions."""
import json
import numpy as np

import experiment_y3_future_book_comoments_v1 as book
import joint_book_comoment_references as history

trial, stack, parent = history.trial, history.stack, history.incumbent


def gains(m,w):
    c = stack.covariance(m)
    v = w.copy()
    v[:,2] += 1
    n = v.shape[1]
    score = v@c[:n,n:]/np.sqrt(((v@c[:n,:n]*v).sum(1)*c[n,n])[:,None])
    return score-score[:1]


def train(mod):
    history = mod.history
    mod.OUT.mkdir(exist_ok=True)
    rows,daily,folds = [],[],[]
    for day in range(4,16):
        raw = trial.clock.book.rawday(day)
        y = trial.clock.labels(day).reshape(-1,50,3)
        refs = history.references(day,raw) if day>=8 else [history.study.screen.history.market.prior.path.early.load(day,raw)]
        if day%2==0:
            st = mod.params(day)
            if day>=8:
                w = book.model.weights(rows,mod.GROUPS)
                group = []
        stride = getattr(mod,'SCREEN_STRIDE',5)
        t = np.arange(day%stride,len(y)-60,stride)
        t = t[(t+60<2401)|(t>=2401)]
        z = mod.advice(raw,t,st)
        if day==8:
            ix = t<2477
            np.testing.assert_allclose(z[ix],mod.advice(raw[:2477],t[ix],st),atol=1.e-6,rtol=0)
        if hasattr(mod,'conditional'):
            designs = [mod.conditional(z,p[t]) for p in refs]
            if day==8:
                for p,v in zip(refs,designs):
                    np.testing.assert_allclose(v[ix],mod.conditional(z[ix],p[t[ix]]),atol=1.e-6,rtol=0)
        else:
            designs = [z]*len(refs)
        rows.append([book.model.stats(p,y,t,v) for p,v in zip(refs,designs)])
        if day>=8:
            ratio = book.model.observed.historical_geometry()[1][:,1]
            legs = np.stack([ratio[2]/ratio[1]*y[t+h,:,1] for h in (0,20,40)],axis=-1)
            grams = []
            for p,v in zip(refs,designs):
                a = np.concatenate((np.ones_like(v[:,:,:1]),p[t],v,y[t,:,2:3],legs),axis=2)
                a = a.reshape(-1,a.shape[-1]).astype(float)
                grams.append(a.T@a)
            group.append(grams)
            daily.append([gains(m,w) for m in grams])
            if day%2:
                folds.append([gains(m,w) for m in np.sum(group,axis=0)])
    f,dd = np.asarray(folds),np.asarray(daily)
    avg = f.mean(0)
    keep = [i for i in mod.CANDIDATES if (avg[:,i,0]>0).all() and (avg[:,i,3]>0).all()
            and (avg[:,i,1:3].sum(1)>0).all() and (f[-1,:,i,0]>0).all()
            and (f[:,:,i,0]>0).sum(0).min()>=3 and (dd[:,:,i,0]>0).sum(0).min()>=5
            and all((avg[:,i,0]>avg[:,j,0]).all() for j in mod.CONTROLS)]
    pick = max(keep,key=lambda i:avg[:,i,0].min()) if keep else 0
    (mod.OUT/'results.json').write_text(json.dumps(dict(arms=mod.ARMS,selected=pick,metrics=['Y3','1-20','21-40','41-60'],
                                                        mean=avg.tolist(),folds=f.tolist(),daily=dd.tolist())))
    print(mod.ARMS[pick],avg[:,:,0],flush=True)
    if pick:
        np.savez(mod.OUT/'model.npz',**mod.params(16),w=book.model.weights(rows,mod.GROUPS)[pick],pick=pick)
    return bool(pick)


def make_correction(mod):
    with np.load(mod.OUT/'model.npz') as f:
        st = dict(f)
    def change(raw,p):
        chunks = []
        for t in np.array_split(np.arange(len(p)),max(1,(len(p)+499)//500)):
            z = mod.advice(raw,t,st)
            if hasattr(mod,'conditional'):
                z = mod.conditional(z,p[t])
            chunks.append(p[t]@st['w'][:3]+z@st['w'][3:])
        return np.concatenate(chunks)
    return change


def replay_for(mod):
    change = mod.correction()
    def read(d,raw):
        p,h,n = mod.history.read(d,raw)
        q,head = p.copy(),h.copy()
        q[:,:,2] += change(raw,p)
        head[:,:,2] += change(raw[:n],h)
        return q,head,n
    return stack.verifier,read
