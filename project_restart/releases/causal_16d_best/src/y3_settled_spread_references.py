"""Load the settled-spread incumbent and rebuild its two training-period Y3 references."""
from functools import lru_cache
import json
import os
import tempfile

import numpy as np
import experiment_y3_settled_rank_graph_v1 as incumbent
from y3_touch_clock_references import digest

CACHE=incumbent.OUT/'historical_spreads'


@lru_cache(maxsize=1)
def reader():
    return incumbent.replay_with()[1]


def read(day,raw):
    return reader()(day,raw)


def training_day(day):
    raw,refs,y=incumbent.source.read_day(day)
    if day<8:
        return raw,refs,y
    assert day<16
    refs=np.asarray(refs)
    path=CACHE/f'day_{day:02d}.npz'
    key=(digest(raw),digest(refs))
    if path.exists():
        with np.load(path) as f:
            q=f['pred']
            assert np.array_equal(f['inputs'],key) and digest(q)==f['checksum'].item()
        return raw,q,y
    update=incumbent.correction()
    q=refs.copy()
    for p,v in zip(refs,q):
        v[:,:,2]+=update(raw,p)
    expected=json.loads((incumbent.OUT/'results.json').read_text())
    pick=expected['selected']
    gains=[np.corrcoef(v[:,:,2].ravel(),y[:,:,2].ravel())[0,1]-np.corrcoef(p[:,:,2].ravel(),y[:,:,2].ravel())[0,1]
           for p,v in zip(refs,q)]
    np.testing.assert_allclose(gains,np.array(expected['daily'])[day-8,:,pick],atol=1e-8,rtol=0)
    CACHE.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=CACHE,suffix='.npz',delete=False) as f:
        np.savez(f,pred=q,inputs=key,checksum=digest(q))
    os.replace(f.name,path)
    return raw,q,y


if __name__=='__main__':
    for day in range(8,16):
        training_day(day)
        print('verified rolling day',day+1,flush=True)
