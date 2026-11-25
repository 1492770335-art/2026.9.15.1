"""Recreate chronological training predictions for the cross-portfolio monotone model."""
import json
import numpy as np
import torch

import experiment_y3_monotone_horizon_projection_v1 as incumbent
import y3_terminal_panel_trial as screen

history,cache,trial=screen.history,screen.cache,screen.trial
CACHE=incumbent.OUT/'historical_references'


def rebuild(model, reference_loader, directory):
    report=json.loads((model.OUT/'results.json').read_text());pick=report['selected']
    coeff=trial.curve.settled.historical_geometry()[1];ratio=coeff[2,1]/coeff[1,1]
    directory.mkdir(exist_ok=True)
    errors=[];rows=[]
    for d in range(4,16):
        raw=trial.clock.book.rawday(d);y=trial.clock.labels(d).reshape(-1,50,3)
        refs=reference_loader(d,raw) if d>=8 else [history.market.prior.path.early.load(d,raw)]
        if d>=8 and d%2==0:w=model.fit(rows);groups=[[],[]]
        rec=[]
        for j,p in enumerate(refs):
            z=model.features(raw,p)
            rec.append(screen.record(z,p,y,d))
            if d>=8:
                ds=screen.changes(z,p,w)[pick]
                t=np.arange(len(raw)-60)
                q=np.zeros_like(p[t]);truth=np.zeros_like(q)
                q[:,:,2]=trial.clock.center(p[t,:,2]-ratio*p[t,:,1])
                truth[:,:,2]=ratio*trial.clock.center(y[t+40,:,1])
                stat=trial.scoring.audit.blockstats
                groups[j].append([stat(p,y,[ds]).sum(0),stat(q,truth,[ds[t]]).sum(0)])
                p[:,:,2]+=ds
        rows.append([np.mean(a,0) for a in zip(*rec)])
        if d>=8:
            digest=cache.cached.digest
            np.savez(directory/f'day_{d:02d}.npz',forecasts=refs,cutoff=d-d%2,raw_hash=digest(raw),forecast_hash=digest(refs))
            if d%2:
                obs=np.array([[trial.scoring.audit.pooled(g)[0] for g in np.sum(c,axis=0)] for c in groups])
                ref=np.array(report['folds'][(d-9)//2])[:,:,pick]
                errors.append(float(np.max(abs(obs-ref))))
                assert errors[-1]<1e-9
    (directory/'verified.json').write_text(json.dumps(dict(fold_errors=errors)))


def prepare():
    rebuild(incumbent,history.references,CACHE)

def references(d,raw):
    return cache.history.rolling.read_verified(CACHE,d,raw)


def outer(mod):
    update=incumbent.correction()
    def load(day,raw):
        p,h,n=cache.load(day,raw)
        p[:,:,2]+=update(raw,p);h[:,:,2]+=update(raw[:n],h)
        return p,h,n
    cache.scoring.outer(mod,reference=incumbent,load_reference=load)


if __name__=='__main__':
    torch.set_num_threads(2);prepare()
