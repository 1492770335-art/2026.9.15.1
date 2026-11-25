"""Ablate groups of input channels with identical targets, constraints, and walk-forward splits."""
import json

import numpy as np

import experiment_y3_protected_path_decoder_v1 as model

VIEWS = {'incumbent forecasts': list(range(32,38)), 'neural paths': list(range(32)),
         'relative paths and incumbent': list(range(16))+list(range(32,38)),
         'market paths and incumbent': list(range(16,38)), 'all inputs': list(range(38))}


def main():
    grams = np.load(model.OUT/'training_moments.npz')['grams']
    arms = [model.MODES.index(name) for name in ('unconstrained endpoint','distant with protected first twenty')]
    report = {}
    for name,cols in VIEWS.items():
        ids = np.r_[np.arange(5),5+np.asarray(cols)]
        rows = grams[:,:,ids,:][:,:,:,ids]
        result = []
        for start in (4,8):
            w = model.fit(rows[:start])[arms]
            result.append([[model.changes(model.base.covariance(m),v) for v in w] for m in rows[start:start+4].sum(0)])
        report[name] = np.asarray(result).tolist()
        print(name,np.round(np.asarray(result)[:,:,1].sum(2),7).ravel())
    original = json.loads((model.OUT/'results.json').read_text())
    np.testing.assert_allclose(np.array(report['all inputs']), np.array(original['folds'])[:,:,arms],atol=1e-9,rtol=0)
    (model.OUT/'input_ablation.json').write_text(json.dumps(dict(arms=[model.MODES[j] for j in arms],
        windows=original['windows'],horizons=original['horizons'],views=report),indent=2))


if __name__=='__main__':
    main()
