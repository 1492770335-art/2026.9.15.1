"""Independently replay the complete early-book transfer layer on each full held-out day and prefix."""
import json
import os
import sys
import numpy as np
import torch
import experiment_joint_early_book_transfer_v1 as model

trace=model.stack.verifier.trace
OUT=model.OUT/'full_chain_verification'


def check(day):
    replay=trace.replay
    raw=replay.original.features(replay.original.DATA,day)
    with np.load(replay.CACHE/f'day_{day}.npz') as f:
        assert str(f['raw_sha256'])==replay.digest(raw)
        p,h,n=f['full'],f['head'],int(f['prefix_ticks'])
    factory=model.layer()
    full,head=factory(raw,p),factory(raw[:n],h)
    assert np.isfinite(full).all() and abs(full[:n]-head).max()<1e-4
    OUT.mkdir(exist_ok=True)
    path=OUT/f'day_{day}.npz'
    tmp=path.with_suffix(f'.{os.getpid()}.npz')
    np.savez(tmp,full=full,head=head,raw_hash=replay.digest(raw),pred_hash=replay.digest(full),head_hash=replay.digest(head))
    tmp.replace(path)
    print('replayed',day+1,flush=True)


def read(day,raw):
    with np.load(OUT/f'day_{day}.npz') as archive:
        q,h=archive['full'],archive['head']
        assert [str(archive[k]) for k in ('raw_hash','pred_hash','head_hash')]==[trace.replay.digest(v) for v in (raw,q,h)]
    return q,h,len(h)


def finish():
    reports={}
    rows=[]
    expected=json.loads((model.OUT/'reload_verification.json').read_text())
    _,parent=model.history.replay_with()
    for d in range(16,20):
        raw=trace.replay.original.features(trace.replay.original.DATA,d)
        q,h,n=read(d,raw)
        p,_,_=parent(d,raw)
        row=p.reshape(-1,3),q.reshape(-1,3),trace.replay.labels(d)
        rows.append(row)
        stat=trace.older.measure([row])
        daily=expected['daily'][str(d+1)]
        np.testing.assert_allclose(stat['candidate'],daily['corr'],atol=1e-10,rtol=0)
        assert abs(stat['score']-daily['score'])<1e-10
        reports[str(d+1)]=dict(score=stat['score'],gain=stat['gain'],prefix_error=float(abs(q[:n]-h).max()))
    pool=trace.older.measure(rows)
    assert pool['rows']==expected['pooled']['rows']==936250
    np.testing.assert_allclose(pool['candidate'],expected['pooled']['candidate'],atol=1e-10,rtol=0)
    (OUT/'composed_replay.json').write_text(json.dumps(dict(pooled=pool,daily=reports),indent=2))
    print(pool,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['finish']:
        finish()
    elif sys.argv[1:]:
        check(int(sys.argv[1]))
    else:
        for day in range(16,20):
            check(day)
        finish()
