"""Capture complete and prefix predictions while replaying the frozen price-change revision model."""
import json
import numpy as np
import torch
import os

import experiment_y3_quiet_forecast_revisions_v1 as model
import y3_tail_specialist_eval as scoring

CACHE=model.OUT/'verified_parent_trace'
digest=scoring.trace.replay.digest


def load(day,raw):
    manifest=json.loads((CACHE/'verified.json').read_text())
    assert manifest['model']==model.OUT.name and 16<=day<20
    with np.load(CACHE/f'day_{day}.npz') as f:
        q,h=f['full'],f['head']
        assert digest(raw)==f['raw_hash'].item() and digest(q)==f['full_hash'].item() and digest(h)==f['head_hash'].item()
        return q,h,int(f['n'])


def store(day,raw,full,prefix):
    tmp=CACHE/f'day_{day}.{os.getpid()}.npz'
    np.savez(tmp,full=full,head=prefix,n=len(prefix),raw_hash=digest(raw),full_hash=digest(full),head_hash=digest(prefix))
    tmp.replace(CACHE/f'day_{day}.npz')
    print('replayed day',day+1,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    CACHE.mkdir(exist_ok=True)
    scoring.verify_full_chain(model,on_day=store)
    r=json.loads((model.OUT/'full_chain_verification/composed_replay.json').read_text())
    (CACHE/'verified.json').write_text(json.dumps(dict(model=model.OUT.name,pooled=r['pooled']),indent=2))
