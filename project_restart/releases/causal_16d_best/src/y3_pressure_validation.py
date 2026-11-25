"""Walk-forward evaluation against the retained near/late order model on the two training references."""
import json

import numpy as np

import experiment_y3_pressure_horizon_controls_v1 as incumbent
import y3_tail_specialist_eval as scoring

clock, curve, parent = incumbent.clock, incumbent.curve, incumbent.parent


def screen(module):
    parent_stats, pressure_stats, history = [], [], []
    folds, daily, diagnostics = [], [], []
    for d in range(16):
        raw = clock.book.rawday(d)
        y = clock.labels(d).reshape(-1,50,3)
        pressure = incumbent.signature.observations(raw)
        state = module.state(raw,pressure)
        if d >= 8:
            if d % 2 == 0:
                parent_geometry = parent.geometry(parent_stats)
                prior = incumbent.fit(pressure_stats)
                model, diagnostic = module.fit(history)
                diagnostics.append(diagnostic)
                groups = [[],[]]
            changes = module.changes(state,model)
            if d % 2 == 0:
                short = raw[:367]
                short_state = module.state(short,incumbent.signature.observations(short))
                for a,b in zip(changes,module.changes(short_state,model)):
                    np.testing.assert_allclose(a[:len(short)],b,atol=5e-6,rtol=0)
            terms = parent.terms(parent.incumbent.state(raw)[1],*parent_geometry)
            earlier = terms[0]+parent.candidates(terms)[2]
            signals = incumbent.signature.predict(pressure,prior)
            for j,ref in enumerate(curve.REFS):
                p = ref(d).reshape(y.shape).copy()
                p[:,:,2] += earlier
                p[:,:,2] += incumbent.candidates(signals,p)[4]
                stats = [r.sum(0) for r in scoring.statistics(p,y,changes)]
                groups[j].append(stats)
                if j==0:daily.append([scoring.audit.pooled(r).tolist() for r in stats])
            if d%2:
                folds.append([[scoring.audit.pooled(r).tolist() for r in np.sum(group,0)] for group in groups])
                print(module.OUT.name,'ending',d+1,folds[-1][0],flush=True)
        history.append(module.observe(state,y,d))
        parent_stats.append(parent.incumbent.stats(raw,y,d))
        pressure_stats.append(incumbent.record(pressure,np.arange(d%5,len(raw)-60,5),y))
    pick=module.select(folds,daily) if hasattr(module,'select') else scoring.select(folds,daily)
    report=dict(selected=pick,folds=folds,daily=daily,diagnostics=diagnostics,arms=module.ARMS)
    (module.OUT/'results.json').write_text(json.dumps(report,indent=2))
    if pick is not None:
        model,_=module.fit(history)
        np.savez(module.OUT/'model.npz',pick=pick,**model)
    return pick is not None


def correction(module):
    with np.load(module.OUT/'model.npz') as s:
        pick=int(s['pick'])
        model={k:s[k] for k in s.files if k!='pick'}
    def predict(raw,p):
        state=module.state(raw,incumbent.signature.observations(raw))
        return module.changes(state,model)[pick]
    return predict


def layer(module):
    prior,delta=incumbent.layer(),module.correction()
    def apply(raw,p):
        q=prior(raw,p)
        q[:,:,2]+=delta(raw,q)
        return q
    return apply


def load(module):
    import experiment_periodic_near_carry_v1 as ancestor
    prior,update=ancestor.load(),layer(module)
    return lambda raw,p:update(raw,prior(raw,p))


def run(module):
    module.OUT.mkdir(exist_ok=True)
    if screen(module):scoring.outer(module,reference=incumbent)
