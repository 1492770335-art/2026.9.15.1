"""Walk-forward backbone predictions for downstream experiments."""
import json
import numpy as np
import experiment_opening_expert_v1 as opening

OUT = opening.prior.ART/'opening_oof_cache_v1'


def load(day):
    return np.load(OUT/f'day_{day:02d}.npy',mmap_mode='r')


def main():
    OUT.mkdir(exist_ok=True)
    settings=opening.spec.settings()
    spectral=[];mixing=[]
    ref=json.loads((opening.OUT/'results.json').read_text())
    for tag,days,end in [('A',range(8,12),8),('B',range(12,16),12),('outer',range(16,20),16)]:
        if tag=='A':
            for d,p,y in opening.prior.predictions(tag,days):
                if d<10:
                    np.save(OUT/f'day_{d:02d}.npy',p.reshape(-1,50,3).astype(np.float32))
        for d,raw,old,diff,y,conf in opening.frames(tag,days,opening.spec.projector(range(end)),settings,spectral):
            if d in (12,14,16):
                mix=opening.fit(mixing,'opening_mix')
            pred=old if d<12 else opening.apply('opening_mix',mix,raw,old,diff,conf)
            if d>=12:
                expected=ref['daily'][str(d)]['opening_mix']['candidate']
                np.testing.assert_allclose(opening.prior.corr(pred,y),expected,rtol=0,atol=1e-8)
            np.save(OUT/f'day_{d:02d}.npy',pred.reshape(-1,50,3).astype(np.float32))
            if d<16:
                mixing.append(opening.stats(diff.astype(float),y[:,:2]-old[:,:2],opening.opening(len(raw))))
    print('Verified and cached all 12 rolling prediction days',flush=True)


if __name__=='__main__':
    main()
