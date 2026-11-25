"""Align order-book signals by within-group arrival speed using only earlier trading days."""
import json
import numpy as np
import torch
import experiment_y3_live_sector_factor_v1 as incumbent

clock = incumbent.clock
sector = incumbent.incumbent
OUT = clock.ART / 'y3_peer_arrival_alignment_v1'
DELAYS = np.array([0, 4, 12, 24])


def signals(raw):
    def field(i):
        return np.arcsinh(raw[:, :, i].astype('float32'))
    return np.tanh(np.stack((field(0), field(11), field(14), field(26)-field(38),
                             field(41)-field(42), field(50)-field(51),
                             field(54)-field(55), field(77)-field(78)), axis=2))


def roles(end, ids):
    stats = []
    for day in range(end):
        raw = clock.book.rawday(day)
        state = clock.center(signals(raw))
        score = np.zeros(50)
        for group in np.unique(ids):
            cols = np.flatnonzero(ids == group)
            x = state[:, cols]
            others = (x.sum(1, keepdims=True) - x)/(len(cols)-1)
            for h in (4, 12, 24):
                a, b = x[h:]-x[:-h], others[h:]-others[:-h]
                norm = np.sqrt((a*a).mean(0)*(b*b).mean(0)) + .002
                earlier = (a[:-h]*b[h:]).mean(0)
                later = (a[h:]*b[:-h]).mean(0)
                score[cols] += ((earlier-later)/norm).mean(1)
        stats.append(score)
    speed = np.median(stats, axis=0)
    ranks = np.empty(50, dtype=int)
    for g in np.unique(ids):
        cols = np.flatnonzero(ids == g)
        order = cols[np.argsort(speed[cols])]
        ranks[order] = np.arange(len(cols))*len(DELAYS)//len(cols)
    return ranks, np.corrcoef(np.mean(stats[::2], axis=0), np.mean(stats[1::2], axis=0))[0, 1]


def design(raw, t, ids, ranks):
    signal = clock.center(signals(raw))
    pools = [signal[(t-h).clip(0)] for h in DELAYS]
    local = pools[0]
    arrays = [np.empty_like(local) for _ in range(3)]
    for g in np.unique(ids):
        cols = np.flatnonzero(ids == g)
        assigned = ranks[cols]
        reversed_ranks = np.empty_like(assigned)
        reversed_ranks[np.argsort(assigned,kind='stable')] = np.sort(assigned)[::-1]
        for k, lag in enumerate((None, assigned, reversed_ranks)):
            if lag is None:
                # Every donor gets this group's full empirical distribution of delays.
                mix = np.bincount(assigned, minlength=len(DELAYS))/len(cols)
                values = sum(m*x[:, cols] for m, x in zip(mix, pools)).astype('float32')
            else:
                values = np.empty_like(local[:, cols])
                for j, x in enumerate(pools):
                    values[:, lag == j] = x[:, cols[lag == j]]
            arrays[k][:, cols] = (values.sum(1, keepdims=True)-values)/(len(cols)-1)
    return [clock.center(np.concatenate((local, pools[2], peer, peer*np.abs(local),
                 np.tanh(2*(local-peer)), peer*np.tanh(local*(local-pools[2]))), 2)) for peer in arrays]


def fit(end, ids, ranks):
    history = []
    for d in range(end):
        raw = clock.book.rawday(d)
        t = np.arange(d%9, len(raw)-60, 9)
        y = clock.center(clock.labels(d).reshape(-1,50,3)[t])
        target = (y[:,:,2]-clock.RATIO*y[:,:,1]).clip(-4,4).ravel()
        stats = []
        for x in design(raw,t,ids,ranks):
            x=x.reshape(len(target),-1).astype(float)
            stats.append((x.T@x/len(x), x.T@target/len(x)))
        history.append(stats)
    weights = []
    for arm in zip(*history):
        a,b = [np.mean(v,axis=0) for v in zip(*arm)]
        weights.append(np.linalg.solve(a+.30*np.diag(a.diagonal().clip(.0001))+.00015*np.eye(len(a)), b))
    return np.stack(weights)


def predict(raw,t,ids,ranks,w):
    timing = [x@c for x,c in zip(design(raw,t,ids,ranks),w)]
    return .20*np.stack((timing[1]-timing[0], timing[2]-timing[0], timing[1]-timing[2]),2)


def run():
    folds, daily, stability = [], [], {}
    for end in (8,10,12,14):
        ids = sector.grouping(end)
        ranks, stability[end] = roles(end,ids)
        w = fit(end,ids,ranks)
        groups = [[[],[],[]],[[],[],[]]]
        for d in (end,end+1):
            raw=clock.book.rawday(d)
            t=np.arange(d%4,len(raw),4)
            change=predict(raw,t,ids,ranks,w)
            if d==end:
                sub=t[t<431]
                np.testing.assert_allclose(change[:len(sub)],predict(raw[:431],sub,ids,ranks,w),atol=2e-6,rtol=0)
            truth=clock.labels(d).reshape(-1,50,3)[t]
            for j,ref in enumerate(sector.incumbent.REFS):
                p=ref(d).reshape(-1,50,3)[t]
                for k in range(3):
                    groups[j][k].append((p,truth,change[:,:,k]))
            daily.append([incumbent.base.chain.metric.gain(g[-1:]) for g in groups[0]])
        fold=[[incumbent.base.chain.metric.gain(g) for g in side] for side in groups]
        folds.append(fold)
        print(end+1,'speed reproducibility',stability[end],'Y3',fold[0],flush=True)
    def passes(k):
        return all(sum(row[j][k]['y3']>0 for row in folds)>=3 and folds[-1][j][k]['y3']>0 and
                   np.mean([row[j][k]['y3'] for row in folds])>0 for j in (0,1)) and sum(v[k]['y3']>0 for v in daily)>=5
    # The reversed arm tests specificity; the only eligible forecast uses aligned donors.
    selected=bool(passes(0) and np.mean([row[0][2]['y3'] for row in folds])>0)
    result=dict(selected=selected,folds=folds,daily=daily,speed_reproducibility=stability,
                arms=['aligned minus uniform timing','reversed minus uniform timing','aligned minus reversed'])
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    if selected:
        ids=sector.grouping(16);ranks,_=roles(16,ids)
        np.savez(OUT/'model.npz',ids=ids,ranks=ranks,w=fit(16,ids,ranks))
    return selected


def layer():
    previous=incumbent.layer()
    with np.load(OUT/'model.npz') as s:
        ids,ranks,w=(s[k] for k in ('ids','ranks','w'))
    def apply(raw,p):
        q=previous(raw,p)
        for t in np.array_split(np.arange(len(q)),max(1,(len(q)+799)//800)):
            q[t,:,2]+=predict(raw,t,ids,ranks,w)[:,:,0]
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as ancestor
    parent,update=ancestor.load(),layer()
    return lambda raw,p:update(raw,parent(raw,p))


if __name__=='__main__':
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    if run():
        incumbent.base.chain.replay.verify(OUT,layer,incumbent.OUT)
