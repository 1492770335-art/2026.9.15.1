"""Cache checked full-day and prefix forecasts of the frozen market-agreement model for subsequent trials."""
import json
import os
import tempfile
import numpy as np
import torch

import y3_overshoot_agreement_references as training
import y3_unfulfilled_references as history

model=training.incumbent
cached=history.rolling.cached
scoring=cached.scoring
CACHE=model.OUT/'verified_parent_trace'


def load(day,raw):
    if day not in range(16,20) or not (CACHE/'verified.json').is_file():
        raise ValueError('Missing verified market-agreement trace for the requested validation day.')
    with np.load(CACHE/f'day_{day}.npz') as item:
        p,h=item['full'],item['head']
        assert cached.digest(raw)==item['raw_hash'].item()
        assert cached.digest(p)==item['full_hash'].item() and cached.digest(h)==item['head_hash'].item()
        return p,h,int(item['n'])


def outer(module):
    scoring.outer(module,reference=model,load_reference=load)


def main():
    reference=json.loads((model.OUT/'reload_verification.json').read_text())
    stages=(history.rolling.incumbent,history.previous.parent.previous.incumbent,history.previous.parent.incumbent,
            history.previous.incumbent,history.incumbent,training.market,model)
    updates=[stage.correction() for stage in stages]
    CACHE.mkdir(exist_ok=True)
    rows=[]
    with tempfile.TemporaryDirectory(dir=CACHE) as directory:
        from pathlib import Path
        tmp=Path(directory)
        for day in range(16,20):
            raw=scoring.trace.replay.original.features(scoring.trace.replay.original.DATA,day)
            p,h,n=cached.load(day,raw)
            for update in updates:
                p[:,:,2]+=update(raw,p)
                h[:,:,2]+=update(raw[:n],h)
            baseline=p.reshape(-1,3)
            row=(baseline,baseline,scoring.trace.replay.labels(day))
            score=scoring.trace.older.measure([row])['score']
            assert np.isfinite(p).all() and np.max(abs(h-p[:n]))<1e-4
            assert abs(score-reference['daily'][str(day+1)]['score'])<1e-10
            rows.append(row)
            np.savez(tmp/f'day_{day}.npz',full=p,head=h,n=n,raw_hash=cached.digest(raw),
                     full_hash=cached.digest(p),head_hash=cached.digest(h))
        pooled=scoring.trace.older.measure(rows)
        assert pooled['rows']==936250
        np.testing.assert_allclose(pooled['candidate'],reference['pooled']['candidate'],rtol=0,atol=1e-10)
        marker=tmp/'verified.json'
        marker.write_text(json.dumps(dict(model=model.OUT.name,pooled=pooled),indent=2))
        for file in tmp.glob('*.npz'):os.replace(file,CACHE/file.name)
        os.replace(marker,CACHE/marker.name)
    print(model.OUT.name,pooled['score'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
