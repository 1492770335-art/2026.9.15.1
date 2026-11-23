"""Combine the separately selected near repricing and far response-template corrections."""
import json
import numpy as np
import experiment_repricing_payoff_heads_v1 as event
import experiment_response_template_training_v1 as response
from experiment_causal_cross_horizon_ridge_v1 import ART

OUT=ART/'repricing_response_joint_v1'


def load():
    np.testing.assert_array_equal(np.load(event.OUT/'mask.npy'),[[0,0,0],[1,0,0]])
    np.testing.assert_array_equal(np.load(response.OUT/'mask.npy'),[[0,0,0],[0,0,1]])
    return event.load(parent=response.load())


def main():
    OUT.mkdir(exist_ok=True)
    first=json.loads((event.OUT/'reload_verification.json').read_text())['pooled']
    last=json.loads((response.OUT/'reload_verification.json').read_text())['pooled']
    expected=first['candidate'][:2]+last['candidate'][2:]
    for name in ('raw_replay','reload_verification'):
        res=event.linear.best.online.best.roll.gls.best.walk.replay(load())
        assert res['pooled']['rows']==936250
        np.testing.assert_allclose(res['pooled']['candidate'],expected,rtol=0,atol=1e-10)
        if name=='reload_verification':
            old=json.loads((OUT/'raw_replay.json').read_text())
            for d,entry in res['daily'].items():
                np.testing.assert_allclose(old['daily'][str(d)]['score'],entry['score'],rtol=0,atol=1e-10)
        (OUT/f'{name}.json').write_text(json.dumps(res,indent=2))
        print(name,res['pooled']['score'],flush=True)


if __name__=='__main__':main()
