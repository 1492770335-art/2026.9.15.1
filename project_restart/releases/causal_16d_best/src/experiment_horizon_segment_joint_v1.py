"""Combine the separately trained medium-horizon transition and far-market distribution heads."""
import json
import numpy as np
import experiment_horizon_composition_v1 as medium
import experiment_segment_distribution_head_v1 as segment
from experiment_causal_cross_horizon_ridge_v1 import ART

OUT=ART/'horizon_segment_joint_v1'


def load():
    np.testing.assert_array_equal(np.load(medium.OUT/'model.npz')['arms'],[1,-1])
    np.testing.assert_array_equal(np.load(segment.OUT/'gate.npy'),[True,False])
    return segment.load(parent=medium.load())


def main():
    OUT.mkdir(exist_ok=True)
    replay=segment.best.roll.gls.best.walk.replay
    r1=json.loads((medium.OUT/'reload_verification.json').read_text())
    r2=json.loads((segment.OUT/'raw_replay.json').read_text())
    expected=r1['pooled']['candidate'][:2]+r2['pooled']['candidate'][2:]
    for name in ('raw_replay','reload_verification'):
        res=replay(load())
        assert res['pooled']['rows']==r1['pooled']['rows']==r2['pooled']['rows']==936250
        np.testing.assert_allclose(res['pooled']['candidate'],expected,rtol=0,atol=1e-10)
        if name=='reload_verification':
            previous=json.loads((OUT/'raw_replay.json').read_text())
            np.testing.assert_allclose(previous['pooled']['score'],res['pooled']['score'],rtol=0,atol=1e-11)
            for d in previous['daily']:
                np.testing.assert_allclose(previous['daily'][d]['score'],res['daily'][int(d)]['score'],rtol=0,atol=1e-10)
        (OUT/f'{name}.json').write_text(json.dumps(res,indent=2))
        print(name,res['pooled']['candidate'],res['pooled']['score'],flush=True)


if __name__=='__main__':main()
