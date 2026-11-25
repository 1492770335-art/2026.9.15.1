"""At fixed fold weights, test the role of the nonlinear cross-portfolio assignment in training-period gains."""
import json
import numpy as np
import torch

import experiment_y3_monotone_horizon_projection_v1 as model

def main():
    screen,trial,history=model.screen,model.trial,model.screen.history
    saved=json.loads((model.OUT/'results.json').read_text());pick=saved['selected']
    assert model.ARMS[pick]=='out-of-basket projection'
    rows=[];all_stats=[[],[]]
    for d in range(4,16):
        raw=trial.clock.book.rawday(d);y=trial.clock.labels(d).reshape(-1,50,3)
        refs=history.references(d,raw) if d>=8 else [history.market.prior.path.early.load(d,raw)]
        if d>=8 and d%2==0:w=model.fit(rows)[:,pick:pick+1]
        rec=[]
        for j,p in enumerate(refs):
            z=model.features(raw,p);rec.append(screen.record(z,p,y,d))
            if d>=8:
                deltas=[]
                # Preserve the learned coefficients and change only the cross-portfolio curve being evaluated.
                for v in (z[:,:,2],np.zeros_like(z[:,:,2]),z[:,:,0],z[:,:,5],np.roll(z[:,:,2],17,axis=1)):
                    projected=z.copy();projected[:,:,2]=v
                    deltas.append(screen.changes(projected,p,w)[0])
                all_stats[j].append(trial.scoring.audit.blockstats(p,y,deltas).sum(0))
        rows.append([np.mean(a,axis=0) for a in zip(*rec)])
    result=dict(days=list(range(9,17)),views=['selected','controls only','linear replacement','wrong anchor','wrong recipient'],
                y3_gains=[trial.scoring.audit.pooled(np.sum(s,0)).tolist() for s in all_stats])
    (model.OUT/'training_assignment_audit.json').write_text(json.dumps(result,indent=2))
    print(result,flush=True)

if __name__=='__main__':
    torch.set_num_threads(2);main()
