"""Build chronological forecasts of the retained risk/ordering model for screening subsequent Y3 changes."""
import json
import os
import numpy as np
import torch

import experiment_y3_risk_rank_partition_v1 as incumbent
import y3_pair_incumbent_validation as parent

CACHE=incumbent.OUT/'historical_references'
digest=parent.previous.previous.prior.digest


def prepare():
    report=json.loads((incumbent.OUT/'results.json').read_text())
    pick=report['selected']
    assert pick is not None
    CACHE.mkdir(exist_ok=True)
    hist=[]
    checked=[]
    for d in range(8,16):
        raw=parent.clock.book.rawday(d)
        y=parent.clock.labels(d).reshape(-1,50,3)
        z=incumbent.state(raw)
        refs=parent.refs(d,raw)
        if d>=10 and d%2==0:
            model=incumbent.fit(hist)
            summaries=[[],[]]
        records=[]
        components=[]
        for j,p in enumerate(refs):
            records.append(incumbent.record_with_state(z,p,y,d))
            if d>=10:
                terms=incumbent.changes_with_reference(z,model,p)
                delta=terms[pick]
                summaries[j].append([a.sum(0) for a in parent.previous.statistics(p,y,[delta])])
                p[:,:,2]+=delta
                updated=incumbent.changes_with_reference(z,model,p)
                components.append(np.stack(updated[:2],2))
        hist.append(incumbent.merge_records(records))
        if d<10:
            continue
        vals=np.stack(components)
        path=CACHE/f'day_{d:02d}.npz'
        tmp=path.with_suffix(f'.{os.getpid()}.npz')
        np.savez(tmp,p=refs,components=vals,raw_hash=digest(raw),pred_hash=digest(refs),
                 component_hash=digest(vals),cutoff=d-d%2)
        tmp.replace(path)
        if d%2:
            found=np.array([[parent.scoring.audit.pooled(s)[0] for s in np.sum(one,axis=0)] for one in summaries])
            expected=np.asarray(report['folds'][(d-11)//2])[:,:,pick]
            diff=float(abs(found-expected).max())
            assert diff<2e-8,(d,diff)
            checked.append(diff)
            print('verified through date',d+1,diff,flush=True)
    (CACHE/'verified.json').write_text(json.dumps(dict(pick=pick,dates=list(range(11,17)),errors=checked),indent=2))


def load(d,raw):
    assert 10<=d<16 and (CACHE/'verified.json').exists()
    with np.load(CACHE/f'day_{d:02d}.npz') as f:
        p,terms=f['p'],f['components']
        assert int(f['cutoff'])==d-d%2 and digest(raw)==f['raw_hash'].item()
        assert digest(p)==f['pred_hash'].item() and digest(terms)==f['component_hash'].item()
    return p,terms


def references(d,raw):
    return load(d,raw)[0] if d>=10 else parent.refs(d,raw)


def outer(module):
    old=parent.previous.previous
    parent.scoring.outer(module,reference=incumbent,preceding=(old.prior.prior.incumbent,
                         old.prior.incumbent,old.incumbent,parent.previous.parent,parent.incumbent))


def run(module):
    incumbent.screen.run(module,references=references,outer=outer)


def correction(module):
    return parent.previous.previous.prior.correction(module)


def layer(module):
    base,update=incumbent.layer(),module.correction()
    def predict(raw,p):
        q=base(raw,p)
        q[:,:,2]+=update(raw,q)
        return q
    return predict


if __name__=='__main__':
    torch.set_num_threads(2)
    prepare()
