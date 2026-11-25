"""Verify the stage readouts in one fresh end-to-end replay against their independently built component caches."""
import argparse
import json
import numpy as np
import torch
import experiment_joint_horizon_stage_transfer_v1 as transfer

stack=transfer.stack
trace=stack.verifier.trace


def run(models):
    earlier=stack.early.incumbent.layer()
    experts=[stack.module(name).correction() for name in stack.STAGES]
    reads={m:np.load(m.OUT/'model.npz')['w'] for m in models}
    def apply(raw,p):
        q,z=stack.revisions(raw,earlier(raw,p),experts)
        result={}
        for m,w in reads.items():
            result[m]=stack.adjusted(q,z,w) if m is stack else transfer.apply_readout(q,z,w)
        return q,result
    rows={m:[] for m in models}
    daily={m:{} for m in models}
    for day in range(16,20):
        raw=trace.replay.original.features(trace.replay.original.DATA,day)
        with np.load(trace.replay.CACHE/f'day_{day}.npz') as f:
            assert str(f['raw_sha256'])==trace.replay.digest(raw)
            p,h,n=f['full'],f['head'],int(f['prefix_ticks'])
        q,out=apply(raw,p)
        head,first=apply(raw[:n],h)
        y=trace.replay.labels(day)
        with np.load(stack.OUT/'components'/f'day_{day}.npz') as f:
            np.testing.assert_array_equal(q,f['full'])
            np.testing.assert_array_equal(head,f['head'])
        for m,ans in out.items():
            where=m.OUT/'full_chain_verification'
            where.mkdir(exist_ok=True)
            err=float(abs(ans[:n]-first[m]).max())
            assert np.isfinite(ans).all() and err<1e-4
            row=(q.reshape(-1,3),ans.reshape(-1,3),y)
            rows[m].append(row)
            daily[m][str(day+1)]=dict(score=trace.older.measure([row])['score'],prefix_error=err)
            np.savez(where/f'day_{day}.npz',full=ans,head=first[m],raw_hash=trace.replay.digest(raw),pred_hash=trace.replay.digest(ans))
        print('replayed',day+1,flush=True)
    for m in models:
        actual=trace.older.measure(rows[m])
        exp=json.loads((m.OUT/'reload_verification.json').read_text())
        assert actual['rows']==exp['pooled']['rows']==936250
        np.testing.assert_allclose(actual['candidate'],exp['pooled']['candidate'],rtol=0,atol=1e-10)
        for d,v in daily[m].items():
            assert abs(v['score']-exp['daily'][d]['score'])<1e-10
        (m.OUT/'full_chain_verification'/'composed_replay.json').write_text(json.dumps(dict(pooled=actual,daily=daily[m]),indent=2))
        print(m.OUT.name,actual['score'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--include-y3-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    run([stack,transfer] if args.include_y3_only else [transfer])
