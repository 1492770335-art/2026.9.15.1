"""Propagate the learned 20-tick clock correction to its contribution in the 60-tick return."""
import json
import numpy as np

import experiment_periodic_book_response_v1 as trial

OUT = trial.ART/'periodic_near_carry_v1'


def carry(q, terms, mask):
    out=q.copy()
    for z,k in zip(terms,mask):
        out+=z*k
        out[:,:,2]+=trial.RATIO*z[:,:,1]*k[1]
    return out


def screen():
    history=[];folds=[];daily=[]
    shift,_=trial.clock_geometry()
    # Test the same predeclared near-band correction in both market and individual returns.
    for end in (8,10,12,14):
        history.extend(trial.statistics(d,shift) for d in range(len(history),end))
        weights=trial.weights(history)
        groups=[[],[]]
        for d in (end,end+1):
            raw=trial.book.rawday(d);t=np.arange(d%3,len(raw),3)
            y=trial.labels(d).reshape(-1,50,3)[t]
            terms=trial.changes(raw,t,shift,weights)
            near=trial.RATIO*(terms[0][:,:,1]+terms[1][:,:,1])
            for j, ref in enumerate((trial.incumbent.fi.peer.reference,trial.incumbent.parent.fixed.earlier.early.baseline)):
                p=ref(d).reshape(-1,50,3)[t]
                base=p[:,:,2]
                groups[j].append((base.ravel(),(base+near).ravel(),y[:,:,2].ravel()))
            def gains(rows):
                a,b,y=(np.concatenate(col) for col in zip(*rows))
                return float(np.corrcoef(b,y)[0,1]-np.corrcoef(a,y)[0,1])
            daily.append([gains(rows[-1:]) for rows in groups])
        folds.append([gains(rows) for rows in groups]);print('window',end+1,folds[-1],flush=True)
    passed=min(np.array(folds).ravel())>0 and (np.array(daily)[:,0]>0).sum()>=5
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=daily,retained=bool(passed)),indent=2))
    return passed


def load():
    parent=trial.incumbent.load()
    with np.load(trial.OUT/'model.npz') as f:
        w,shift,keep=(f['m'],f['r']),int(f['shift']),f['keep']
    if not keep[:,1].all():raise ValueError('Both 20-tick terms are required')
    def predict(raw,p):
        q=parent(raw,p)
        return carry(q,trial.changes(raw,np.arange(len(raw)),shift,w),keep)
    return predict


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if screen():
        reference=trial.OUT if (trial.OUT/'reload_verification.json').exists() else trial.incumbent.OUT
        trial.incumbent.parent.fixed.votes.follow.replay_candidate(OUT,load,baseline=reference)
