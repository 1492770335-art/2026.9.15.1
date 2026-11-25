"""Compare frozen and fine-tuned signals under exactly the same rolling readout on dates 9–16."""
import json
import numpy as np
import torch

import experiment_y3_scale_conditioned_path_v1 as net

trial=net.trial
NAMES=('frozen','direction','own','other','market')
CHANNELS=(0,1,2,4,6)


def main():
    report=json.loads((net.OUT/'results.json').read_text())
    pick=report['selected']
    assert pick==2
    hist=net.warm_history()
    daily=[]
    folds=[]
    all_days=[[],[]]
    for d in range(8,16):
        raw=trial.clock.book.rawday(d)
        y=trial.clock.labels(d).reshape(-1,50,3)
        z=net.historical_state(raw,d)
        if d%2==0:
            m=net.fit(hist)
            stats=[[],[]]
        records=[]
        for j,p in enumerate(net.history.references(d,raw)):
            changes=[]
            for channel in CHANNELS:
                altered=z.copy()
                altered[:,:,4:6]=z[:,:,2*channel:2*channel+2]
                changes.append(net.changes_with_reference(altered,m,p)[pick])
            g=trial.scoring.audit.blockstats(p,y,changes).sum(0)
            stats[j].append(g)
            all_days[j].append(g)
            if j==0:
                daily.append(trial.scoring.audit.pooled(g).tolist())
            records.append(net.record_with_state(z,p,y,d))
        hist.append([np.mean(v,0) for v in zip(*records)])
        if d%2:
            fold=[trial.scoring.audit.pooled(sum(g)).tolist() for g in stats]
            np.testing.assert_allclose(np.array(fold)[:,2],np.array(report['folds'][(d-9)//2])[:,0,pick],atol=1e-10,rtol=0)
            folds.append(fold)
    pooled=[trial.scoring.audit.pooled(sum(g)).tolist() for g in all_days]
    result=dict(channels=NAMES,compared_metric='incremental Y3 correlation',dates=list(range(9,17)),pooled=pooled,folds=folds,daily=daily)
    (net.OUT/'training_transfer_ablation.json').write_text(json.dumps(result,indent=2))
    print(pooled,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
