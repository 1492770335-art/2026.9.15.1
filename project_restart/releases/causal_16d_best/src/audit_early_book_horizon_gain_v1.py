"""Attribute the Pearson change to non-overlapping realized windows on fully observed paths."""
import json
import numpy as np
import experiment_compounded_return_audit_v1 as price
import joint_early_book_references as early

WINDOWS=('ticks_1_20','ticks_21_40','ticks_41_60','compounding_and_label_remainder')


def moments(p,q,parts):
    a=np.column_stack((np.ones(len(p)),p,q,parts))
    return a.T@a


def contributions(g):
    c=early.stack.covariance(g)
    std=np.sqrt(c.diagonal()[:2])
    target_sd=np.sqrt(c[2:,2:].sum())
    gains=(c[1,2:]/std[1]-c[0,2:]/std[0])/target_sd
    return dict(zip(WINDOWS+('total',),[float(v) for v in gains]+[float(gains.sum())]))


def stats(raw,p,q,y):
    with np.load(price.OUT/'model.npz') as f:
        valid=price.present(raw,f['row'])
        ratio=f['coeff'][2,1]/f['coeff'][1,1]
    # Retain only complete 60-tick returns within the same session.
    t=np.arange(len(y)-60)
    t=t[(t+60<2401)|(t>=2401)]
    cs=np.vstack((np.zeros(50,int),(~valid).cumsum(0)))
    complete=cs[t+61]==cs[t]
    pieces=ratio*np.stack([y[t+k,:,1] for k in (0,20,40)],2).astype(float)
    z=np.concatenate((pieces,y[t,:,2:3]-pieces.sum(2,keepdims=True)),2)
    return moments(p[t,:,2][complete],q[t,:,2][complete],z[complete])


def run(current=early, source=early.previous):
    train=[]
    for d in range(10,16):
        raw=early.trial.clock.book.rawday(d)
        y=early.trial.clock.labels(d).reshape(-1,50,3)
        p=source.references(d,raw)
        q=current.references(d,raw)
        train.append([stats(raw,a,b,y) for a,b in zip(p,q)])
    trace=early.stack.verifier.trace
    old=source.replay_with()[1]
    new=current.replay_with()[1]
    held=[]
    for d in range(16,20):
        raw=trace.replay.original.features(trace.replay.original.DATA,d)
        p=old(d,raw)[0]
        q=new(d,raw)[0]
        y=trace.replay.labels(d).reshape(p.shape)
        held.append(stats(raw,p,q,y))
    overall=contributions(sum(held))
    report=dict(training_reference_gains=[contributions(g) for g in np.sum(train,0)],
                validation=overall,validation_rows=int(sum(g[0,0] for g in held)),
                validation_by_day={str(i+17):contributions(g) for i,g in enumerate(held)},
                method='Pearson(new,Y3) minus Pearson(old,Y3), exactly allocated by realized 20-tick subreturn and a remainder on complete same-session paths.')
    (current.incumbent.OUT/'horizon_gain.json').write_text(json.dumps(report,indent=2))
    print(report,flush=True)


if __name__=='__main__':
    import sys
    if sys.argv[1:]==['medium-order-clocks']:
        import y3_medium_order_clock_references as medium
        run(medium,medium.previous.previous.previous)
    elif sys.argv[1:]==['joint-order-clocks']:
        import joint_order_clock_references as clocks
        run(clocks,clocks.previous.previous)
    else:
        run()
