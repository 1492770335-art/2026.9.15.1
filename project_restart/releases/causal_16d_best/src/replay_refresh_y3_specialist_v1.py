"""Replay the post-hoc Y3-only combination from the fixed refresh-cohort experiment."""
import json
import numpy as np
import experiment_refresh_cohort_transfer_v1 as cohort
from experiment_book_transition_rewards_v1 import replay

OUT=cohort.ART/'refresh_cohort_y3_only_v1'


def load():
    with np.load(cohort.OUT/'model.npz') as st:
        geo=st['activity'],st['peers'],st['order']
        w=st['w'].copy()
    w[:,:,:2]=0
    return lambda raw,p:cohort.advance(p,raw,geo,w)


def main():
    OUT.mkdir(exist_ok=True)
    predict=load()
    def verify(raw,p):
        q=predict(raw,p)
        np.testing.assert_array_equal(q[:,:,:2],p[:,:,:2])
        return q
    frozen=replay(verify)
    joint=json.loads((cohort.OUT/'raw_replay.json').read_text())['pooled']
    if abs(frozen['pooled']['candidate'][2]-joint['candidate'][2])>2e-7:
        raise ValueError('target 3 differs from the original cohort model')
    frozen['selection']='Post-hoc target routing after examining the four development dates.'
    frozen['training_days']=[1,16]
    frozen['validation_days']=[17,20]
    (OUT/'raw_replay.json').write_text(json.dumps(frozen,indent=2))
    print(frozen['pooled'],flush=True)


if __name__=='__main__':
    main()
