"""Keep each rolling Y3 head frozen while shrinking its cross-stock exposure update."""
import json
import numpy as np
import torch

import experiment_y3_realized_recipient_exposures_v1 as model
import y3_recipient_exposure_references as accepted


def run():
    calibration = json.loads((model.OUT/'exposure_persistence.json').read_text())['days']
    names = ('fixed exposure', 'weight fitted to price reconstruction', 'full update')
    records, daily, folds = [], [], []
    for day in range(4,16):
        raw = model.trial.clock.book.rawday(day)
        y = model.trial.clock.labels(day).reshape(-1,50,3)
        train = np.arange(90+day%21, len(raw)-60,21)
        test = np.arange(day%4, len(raw)-60,4)
        ticks = np.union1d(train, test) if day >= 8 else train
        xtrain = np.searchsorted(ticks,train)
        refs = (model.history.references(day,raw) if day >= 8 else
                [model.study.screen.history.market.prior.path.early.load(day,raw)])
        if day >= 8:
            true = accepted.references(day,raw)
            if day % 2 == 0:
                weights = model.fit(records)['w'][:,7]
                paired = [[],[]]
            amount = calibration[day-8]['fitted_gain']
            assert calibration[day-8]['training_days'] == day-day%2
        data = []
        for j,p in enumerate(refs):
            x = model.historical_design(day)(raw,p,ticks)
            data.append((x[xtrain],model.study.error(p,y,train)))
            if day < 8:
                continue
            current = x[np.searchsorted(ticks,test)]
            fixed = current[:,:,19:21]
            dynamic = current[:,:,17:19]
            common = current[:,:,:3] @ weights[:3]
            changes = [common + (fixed + f*(dynamic-fixed)) @ weights[17:19] for f in (0,amount,1)]
            np.testing.assert_allclose(p[test,:,2] + changes[-1], true[j,test,:,2], atol=1.e-6, rtol=0)
            block = model.trial.scoring.audit.blockstats(p[test], y[test],changes).sum(0)
            paired[j].append(block)
            if j==0: daily.append(model.trial.scoring.audit.pooled(block).tolist())
        records.append(tuple(np.concatenate(v) for v in zip(*data)))
        if day >= 8 and day%2:
            folds.append([model.trial.scoring.audit.pooled(np.sum(f,0)).tolist() for f in paired])
    report = dict(arms=names, days=list(range(9,17)), daily=daily, folds=folds, mean=np.mean(folds,0).tolist())
    (model.OUT/'transfer_to_y3.json').write_text(json.dumps(report,indent=2))
    print(report['mean'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
