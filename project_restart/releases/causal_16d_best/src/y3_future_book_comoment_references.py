"""Reconstruct the rolling peer forecasts and independently evaluate conditional book covariances."""
import json
import sys
import numpy as np
import torch

import experiment_y3_future_book_comoments_v1 as incumbent
from y3_expanding_path_references import read_verified
from y3_touch_clock_references import digest, read_stage, verify_stage

previous=incumbent.history
trial,stack,study=previous.trial,previous.stack,previous.study
CACHE=incumbent.OUT/'historical_references'
FULL=incumbent.OUT/'full_chain_verification'


def references(d,raw):
    return read_verified(CACHE,d,raw)


def read(d,raw):
    return read_stage(FULL,d,raw)


def replay_with():
    return stack.verifier,read


def outer(module):
    stack.verifier.outer(module,reference=incumbent,load_reference=read)


def rebuild(model=incumbent, reference=previous, directory=CACHE):
    fit_weights=incumbent.model.weights
    measure=incumbent.model.stats
    observed=incumbent.model.observed
    result=json.loads((model.OUT/'results.json').read_text())
    pick=result['selected']
    directory.mkdir(exist_ok=True)
    rows,errors=[],[]
    for d in range(4,16):
        raw=trial.clock.book.rawday(d)
        y=trial.clock.labels(d).reshape(-1,50,3)
        refs=reference.references(d,raw) if d>=8 else [study.screen.history.market.prior.path.early.load(d,raw)]
        if d%2==0:
            st=model.params(d)
            if d>=8:
                w=fit_weights(rows,model.GROUPS)[pick]
                group=[]
        t=np.arange(d%5,len(raw)-60,5)
        t=t[(t+60<2401)|(t>=2401)]
        full_t=np.arange(len(raw)) if d>=8 else t
        advice=np.concatenate([model.advice(raw,chunk,st) for chunk in np.array_split(full_t,max(1,(len(full_t)+799)//800))])
        sample=advice[t] if d>=8 else advice
        rows.append([measure(p,y,t,sample) for p in refs])
        if d<8:
            continue
        grams=[]
        for p in refs:
            base=p[t,:,2].copy()
            p[:,:,2]+=p@w[:3]+advice@w[3:]
            far=observed.historical_geometry()[1][2,1]/observed.historical_geometry()[1][1,1]*(y[t+20,:,1]+y[t+40,:,1])
            a=np.stack((np.ones_like(base),base,p[t,:,2],far,y[t,:,2]),axis=2).reshape(-1,5).astype(float)
            grams.append(a.T@a)
        group.append(grams)
        if d%2:
            actual=[]
            for m in np.sum(group,0):
                c=stack.covariance(m)
                r=c[:2,-2:]/np.sqrt((c.diagonal()[:2]*c[-1,-1])[:,None])
                actual.append(r[1]-r[0])
            want=np.asarray(result['folds'])[(d-9)//2,:,pick]
            errors.append(float(np.max(abs(actual-want))))
            assert errors[-1]<1e-8,errors
        np.savez(directory/f'day_{d:02d}.npz',forecasts=refs,cutoff=d-d%2,raw_hash=digest(raw),forecast_hash=digest(refs))
    (directory/'verified.json').write_text(json.dumps(dict(fold_errors=errors)))


def verify():
    with np.load(incumbent.OUT/'model.npz') as f:
        w,S,pay=f['w'],f['full_state'],f['full_pay']
        assert int(f['pick'])==6
    def independent(raw,p):
        result=p.copy()
        for t in np.array_split(np.arange(len(p)),max(1,(len(p)+639)//640)):
            x=incumbent.observations(raw,t)[0]
            pred=x@S.astype(float)
            mean=pred[:,:,:16]@pay[:16]
            cross=pred[:,:,16:].reshape(len(t),50,8,8)
            cov=cross-pred[:,:,:8,None]*pred[:,:,None,8:16]
            interaction=np.einsum('tsij,ij->ts',cov,pay[16:].reshape(8,8))
            v=trial.clock.center(w[4]*mean+w[7]*interaction)
            result[t,:,2]+=p[t]@w[:3]+v
        return result
    verify_stage(incumbent,previous.read,independent)


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['verify']:
        verify()
    else:
        rebuild()
