"""Verify the response to moderate returns on top of the frozen three-head clock forecast."""
import json
import sys
import numpy as np
import torch
import experiment_y3_order_clock_tail_slices_v1 as incumbent
from y3_same_deadline_references import rebuild
from y3_expanding_path_references import read_verified

previous=incumbent.history
stack,study,trial=previous.stack,previous.study,previous.trial
CACHE=incumbent.OUT/'historical_references'
FULL=incumbent.OUT/'full_chain_verification'


def references(d,raw):
    return read_verified(CACHE,d,raw)


def read(d,raw):
    with np.load(FULL/f'day_{d}.npz') as archive:
        q,h=archive['full'],archive['head']
        assert [str(archive[k]) for k in ('raw_hash','full_hash','head_hash')]==[previous.previous.digest(a) for a in (raw,q,h)]
    return q,h,len(h)


def verify():
    assert (previous.FULL/'composed_replay.json').is_file()
    FULL.mkdir(exist_ok=True)
    reference=json.loads((incumbent.OUT/'reload_verification.json').read_text())
    apply=incumbent.correction()
    with np.load(incumbent.OUT/'model.npz') as archive:
        st=dict(archive)
    trace=stack.verifier.trace
    def run(raw,p):
        cols=incumbent.signals(raw,np.arange(len(raw)),st['bank'],st['prior'])
        x=np.concatenate((trial.clock.center(p),cols),2)
        w=st['w'][:,int(st['pick'])]
        # Rebuild the learned readout directly from the three calibrated band responses.
        delta=np.sum(x.astype(float)*w,axis=2)
        actual=apply(raw,p)
        np.testing.assert_allclose(actual,delta,atol=2e-6,rtol=1e-5)
        q=p.copy()
        q[:,:,2]+=actual
        return q
    daily={}
    rows=[]
    for d in range(16,20):
        raw=trace.replay.original.features(trace.replay.original.DATA,d)
        p,h,n=previous.read(d,raw)
        q=run(raw,p)
        head=run(raw[:n],h)
        err=float(abs(head-q[:n]).max())
        assert err<1e-4 and np.isfinite(q).all()
        row=p.reshape(-1,3),q.reshape(-1,3),trace.replay.labels(d)
        rows.append(row)
        s=trace.older.measure([row])
        assert abs(s['score']-reference['daily'][str(d+1)]['score'])<1e-10
        daily[str(d+1)]=dict(score=s['score'],gain=s['gain'],prefix_error=err)
        np.savez(FULL/f'day_{d}.npz',full=q,head=head,raw_hash=previous.previous.digest(raw),
                 full_hash=previous.previous.digest(q),head_hash=previous.previous.digest(head))
    score=trace.older.measure(rows)
    assert score['rows']==936250
    np.testing.assert_allclose(score['candidate'],reference['pooled']['candidate'],atol=1e-10,rtol=0)
    (FULL/'composed_replay.json').write_text(json.dumps(dict(pooled=score,daily=daily),indent=2))
    print('verified forecast',score['score'],flush=True)


def replay_with():
    if (FULL/'composed_replay.json').is_file():
        return stack.verifier,read
    ver,old=previous.replay_with()
    apply=incumbent.correction()
    def load(d,raw):
        q,h,n=old(d,raw)
        q[:,:,2]+=apply(raw,q)
        h[:,:,2]+=apply(raw[:n],h)
        return q,h,n
    return ver,load


def outer(module):
    ver,load=replay_with()
    ver.outer(module,reference=incumbent,load_reference=load)


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['verify']:
        verify()
    else:
        rebuild(incumbent,previous.references,CACHE)
