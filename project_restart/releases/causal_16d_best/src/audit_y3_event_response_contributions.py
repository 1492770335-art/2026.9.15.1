"""Ablate immediate event neighbors in the fitted readout and compare both event layers to the frozen starting model."""
import json
import numpy as np
import torch
import experiment_y3_event_forecast_response_shape_v1 as model

screen,parent,trial=model.screen,model.parent,model.trial
NAMES=('full','on change only','without leading revision','without following revision','five-tick replacement')


def interventions(raw,p,w):
    z=model.features(raw,p)
    weighted=lambda value:screen.changes(value,p,w)[0]
    results=[weighted(z)]
    for cols in (slice(4,12),slice(4,8),slice(8,12)):
        q=z.copy();q[:,:,cols]=0
        results.append(weighted(q))
    q=z.copy()
    q[:,:,4:8]=z[:,:,20:24]
    q[:,:,8:12]=np.roll(z[:,:,20:24],17,axis=1)
    results.append(weighted(q))
    return results


def training():
    chosen=json.loads((model.OUT/'results.json').read_text())['selected']
    hist=[];grams=[[],[]]
    for d in range(4,16):
        raw=trial.clock.book.rawday(d);y=trial.clock.labels(d).reshape(-1,50,3)
        refs=parent.references(d,raw) if d>=8 else [screen.history.market.prior.path.early.load(d,raw)]
        if d>=8 and d%2==0:w=model.fit(hist)[:,chosen:chosen+1]
        rec=[]
        for j,p in enumerate(refs):
            rec.append(screen.record(model.features(raw,p),p,y,d))
            if d>=8:
                grams[j].append(trial.scoring.audit.blockstats(p,y,interventions(raw,p,w)).sum(0))
        hist.append([np.mean(a,0) for a in zip(*rec)])
    return [trial.scoring.audit.pooled(np.sum(g,0)).tolist() for g in grams]


def validation():
    base=parent.previous
    cached=base.cache
    scoring=cached.scoring
    start=base.incumbent.correction()
    first=parent.incumbent.correction();second=model.correction()
    rows=[];blocks=[];daily={}
    for day in range(16,20):
        raw=scoring.trace.replay.original.features(scoring.trace.replay.original.DATA,day)
        p,h,n=cached.load(day,raw)
        p[:,:,2]+=start(raw,p);h[:,:,2]+=start(raw[:n],h)
        q=p.copy()
        for correct in (first,second):
            q[:,:,2]+=correct(raw,q);h[:,:,2]+=correct(raw[:n],h)
        assert abs(q[:n]-h).max()<1e-4
        y=scoring.trace.replay.labels(day)
        row=(p.reshape(-1,3),q.reshape(-1,3),y)
        s=scoring.trace.older.measure([row])
        blocks.append(scoring.audit.blockstats(p,y.reshape(p.shape),[q[:,:,2].astype(float)-p[:,:,2]]))
        rows.append(row);daily[day+1]=s['gain']
    pooled=scoring.trace.older.measure(rows)
    old=json.loads((base.incumbent.OUT/'reload_verification.json').read_text())['pooled']
    new=json.loads((model.OUT/'reload_verification.json').read_text())['pooled']
    assert pooled['rows']==936250 and abs(pooled['score']-new['score'])<1e-10 and abs(pooled['baseline_score']-old['score'])<1e-10
    return dict(pooled=pooled,daily=daily,score_percentiles=(.5*np.asarray(scoring.audit.intervals(blocks))[:,0]).tolist())


if __name__=='__main__':
    torch.set_num_threads(2)
    report=dict(variants=NAMES,training_y3_gain=training(),combined_validation=validation())
    (model.OUT/'event_contributions.json').write_text(json.dumps(report,indent=2))
    print(report,flush=True)
