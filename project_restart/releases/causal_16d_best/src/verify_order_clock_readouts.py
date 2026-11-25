"""Rebuild the raw-to-score order-clock layer and verify its parent and prefix predictions."""
import json
import os
import sys
import numpy as np
import torch
import y3_order_clock_references as history

model=history.incumbent
trace=history.stack.verifier.trace
OUT=model.OUT/'full_chain_verification'


def check(day):
    r=trace.replay
    raw=r.original.features(r.original.DATA,day)
    with np.load(r.CACHE/f'day_{day}.npz') as base:
        assert str(base['raw_sha256'])==r.digest(raw)
        p,h,n=base['full'],base['head'],int(base['prefix_ticks'])
    parent=model.parent.layer()
    q,f=parent(raw,p),parent(raw[:n],h)
    _,read=history.previous.replay_with()
    ref,head,_=read(day,raw)
    np.testing.assert_array_equal(q,ref)
    np.testing.assert_array_equal(f,head)
    update=model.correction()
    q[:,:,2]+=update(raw,q)
    f[:,:,2]+=update(raw[:n],f)
    assert np.isfinite(q).all() and np.max(abs(q[:n]-f))<1e-4
    OUT.mkdir(exist_ok=True)
    path=OUT/f'day_{day}.npz'
    tmp=path.with_suffix(f'.{os.getpid()}.npz')
    np.savez(tmp,full=q,head=f,raw_hash=r.digest(raw),full_hash=r.digest(q),head_hash=r.digest(f))
    tmp.replace(path)
    print('replayed day',day+1,flush=True)


def read(day,raw):
    with np.load(OUT/f'day_{day}.npz') as f:
        q,h=f['full'],f['head']
        assert [str(f[k]) for k in ('raw_hash','full_hash','head_hash')]==[trace.replay.digest(a) for a in (raw,q,h)]
    return q,h,len(h)


def finish():
    expected=json.loads((model.OUT/'reload_verification.json').read_text())
    _,old=history.previous.replay_with()
    rows=[]
    daily={}
    for d in range(16,20):
        raw=trace.replay.original.features(trace.replay.original.DATA,d)
        p=old(d,raw)[0]
        q,h,n=read(d,raw)
        row=p.reshape(-1,3),q.reshape(-1,3),trace.replay.labels(d)
        rows.append(row)
        s=trace.older.measure([row])
        assert abs(s['score']-expected['daily'][str(d+1)]['score'])<1e-10
        daily[str(d+1)]=dict(score=s['score'],gain=s['gain'],prefix_error=float(abs(q[:n]-h).max()))
    scores=trace.older.measure(rows)
    assert scores['rows']==expected['pooled']['rows']==936250
    np.testing.assert_allclose(scores['candidate'],expected['pooled']['candidate'],atol=1e-10,rtol=0)
    (OUT/'composed_replay.json').write_text(json.dumps(dict(pooled=scores,daily=daily),indent=2))
    print(scores,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['finish']:
        finish()
    elif sys.argv[1:]:
        check(int(sys.argv[1]))
    else:
        for d in range(16,20):
            check(d)
        finish()
