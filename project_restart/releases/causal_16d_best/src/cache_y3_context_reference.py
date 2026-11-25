"""Capture verified full-day and prefix forecasts while independently replaying the complete context model."""
import json
import os
import numpy as np
import torch
import experiment_y3_path_price_context_v1 as model
import y3_tail_specialist_eval as scoring

CACHE=model.OUT/'verified_parent_trace'
digest=scoring.trace.replay.digest


def store(day,raw,full,head):
    path=CACHE/f'day_{day}.npz'
    tmp=path.with_suffix(f'.{os.getpid()}.npz')
    np.savez(tmp,full=full,head=head,n=len(head),raw_hash=digest(raw),full_hash=digest(full),head_hash=digest(head))
    tmp.replace(path)


def load(day,raw):
    assert (CACHE/'verified.json').is_file() and 16<=day<20
    with np.load(CACHE/f'day_{day}.npz') as s:
        p,h=s['full'],s['head']
        assert digest(raw)==s['raw_hash'].item() and digest(p)==s['full_hash'].item() and digest(h)==s['head_hash'].item()
        return p,h,int(s['n'])


def main():
    CACHE.mkdir(exist_ok=True)
    scoring.verify_full_chain(model,on_day=store)
    report=json.loads((model.OUT/'full_chain_verification/composed_replay.json').read_text())
    (CACHE/'verified.json').write_text(json.dumps(dict(model=model.OUT.name,days=[17,18,19,20],score=report['pooled']['score']),indent=2))


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
