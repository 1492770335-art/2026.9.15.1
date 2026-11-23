"""Compose the independently selected short spell and long reversal-payoff predictions."""
import json
import numpy as np
import experiment_pressure_spell_survival_v1 as spell
import experiment_pressure_reversal_payoff_v1 as reversal

OUT=spell.peer.ART/'pressure_payoff_joint_v1'


def load():
    with np.load(spell.OUT/'model.npz') as data:
        np.testing.assert_array_equal(data['mask'],[[1,0,0],[0,0,0]])
    np.testing.assert_array_equal(np.load(reversal.OUT/'keep.npy'),[[0,0,1],[0,0,0]])
    return reversal.load(parent=spell.load())


def main():
    OUT.mkdir(exist_ok=True)
    base=json.loads((spell.peer.PARENT.OUT/'reload_verification.json').read_text())
    near=json.loads((spell.OUT/'reload_verification.json').read_text())
    far=json.loads((reversal.OUT/'raw_replay.json').read_text())
    expected=[near['pooled']['candidate'][0],base['pooled']['candidate'][1],far['pooled']['candidate'][2]]
    for label in ('raw_replay','reload_verification'):
        result=spell.peer.walk.replay(load())
        assert result['pooled']['rows']==936250
        np.testing.assert_allclose(result['pooled']['candidate'],expected,rtol=0,atol=1e-10)
        for d,day in result['daily'].items():
            key=str(d)
            value=near['daily'][key]['score']+far['daily'][key]['score']-base['daily'][key]['score']
            np.testing.assert_allclose(day['score'],value,rtol=0,atol=1e-10)
        if label=='reload_verification':
            first=json.loads((OUT/'raw_replay.json').read_text())
            np.testing.assert_allclose(result['pooled']['score'],first['pooled']['score'],rtol=0,atol=1e-10)
        (OUT/f'{label}.json').write_text(json.dumps(result,indent=2))
        print(label,result['pooled']['score'],flush=True)


if __name__=='__main__':main()
