"""Cache frozen historical-fold references for the fourfold-strength order filter and test its successors."""
import hashlib
import json
import os

import numpy as np
import torch

import experiment_y3_order_filter_strength_v1 as incumbent
import y3_pressure_validation as prior

clock,curve,scoring=prior.clock,prior.curve,prior.scoring
CACHE=incumbent.OUT/'historical_references'


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()


def prepare():
    hist,book,order=[],[],[]
    CACHE.mkdir(exist_ok=True)
    expected=json.loads((incumbent.OUT/'results.json').read_text())
    pick=expected['selected'];amp=expected['amplitudes'][pick]
    assert amp==4
    for d in range(16):
        raw=clock.book.rawday(d);y=clock.labels(d).reshape(-1,50,3)
        pre=prior.incumbent.signature.observations(raw);z=incumbent.source.state(raw,pre)
        if d>=8:
            if d%2==0:
                geo=prior.parent.geometry(hist);w=prior.incumbent.fit(book);coeff=incumbent.source.fit(order)[0]
                groups=[[],[]]
            terms=prior.parent.terms(prior.parent.incumbent.state(raw)[1],*geo)
            old=prior.incumbent.signature.predict(pre,w);delta=amp*incumbent.source.changes(z,coeff)[1]
            refs=[]
            for j,ref in enumerate(curve.REFS):
                p=ref(d).reshape(y.shape).copy()
                p[:,:,2]+=terms[0]+prior.parent.candidates(terms)[2]
                p[:,:,2]+=prior.incumbent.candidates(old,p)[4]
                groups[j].append([m.sum(0) for m in scoring.statistics(p,y,[delta])])
                p[:,:,2]+=delta;refs.append(p)
            pred=np.asarray(refs)
            path=CACHE/f'day_{d:02d}.npz'
            tmp=path.with_suffix(f'.{os.getpid()}.npz')
            np.savez(tmp,p=pred,hash=digest(pred),raw_hash=digest(raw),train_end=d-d%2)
            tmp.replace(path)
            if d%2:
                found=np.array([[scoring.audit.pooled(m)[0] for m in np.sum(g,axis=0)] for g in groups])
                wanted=np.asarray(expected['folds'][(d-9)//2])[:,:,pick]
                np.testing.assert_allclose(found,wanted,rtol=0,atol=2e-9)
                print('verified reference days',d,d+1,flush=True)
        hist.append(prior.parent.incumbent.stats(raw,y,d))
        book.append(prior.incumbent.record(pre,np.arange(d%5,len(y)-60,5),y));order.append(incumbent.source.observe(z,y,d))
    (CACHE/'verified.json').write_text(json.dumps(dict(amplitude=amp,dates=list(range(9,17)),source=incumbent.OUT.name)))


def refs(d,raw):
    assert 8<=d<16 and (CACHE/'verified.json').is_file()
    with np.load(CACHE/f'day_{d:02d}.npz') as f:
        p=f['p']
        assert f['hash'].item()==digest(p) and f['raw_hash'].item()==digest(raw) and int(f['train_end'])==d-d%2
        return p


def screen(module, references=refs):
    hist=[];folds=[];daily=[];diag=[]
    for d in range(16):
        raw=clock.book.rawday(d);y=clock.labels(d).reshape(-1,50,3)
        z=module.state(raw)
        if d>=8:
            if d%2==0:model,info=module.fit(hist);diag.append(info);groups=[[],[]]
            conditioned=getattr(module,'changes_with_reference',None)
            if conditioned is None:
                changes=module.changes(z,model)
                if d%2==0:
                    for a,b in zip(changes,module.changes(module.state(raw[:367]),model)):
                        np.testing.assert_allclose(a[:367],b,rtol=0,atol=5e-6)
            for j,p in enumerate(references(d,raw)):
                if conditioned is not None:
                    changes=conditioned(z,model,p)
                    if d%2==0:
                        for a,b in zip(changes,conditioned(module.state(raw[:367]),model,p[:367])):
                            np.testing.assert_allclose(a[:367],b,rtol=0,atol=5e-6)
                metrics=getattr(module,'statistics',scoring.statistics)
                moments=[s.sum(0) for s in metrics(p,y,changes)]
                groups[j].append(moments)
                if not j:daily.append([scoring.audit.pooled(m).tolist() for m in moments])
            if d%2:
                folds.append([[scoring.audit.pooled(m).tolist() for m in np.sum(g,axis=0)] for g in groups])
                print(module.OUT.name,'ending',d+1,folds[-1][0],flush=True)
        hist.append(module.observe(z,y,d))
    pick=getattr(module,'select',scoring.select)(folds,daily)
    (module.OUT/'results.json').write_text(json.dumps(dict(selected=pick,folds=folds,daily=daily,diagnostics=diag,arms=module.ARMS,secondary=getattr(module,'SECONDARY','late relative correlation')),indent=2))
    if pick is not None:
        model,_=module.fit(hist);np.savez(module.OUT/'model.npz',pick=pick,**model)
    return pick is not None


def correction(module):
    with np.load(module.OUT/'model.npz') as f:
        model={k:f[k] for k in f.files if k!='pick'};pick=int(f['pick'])
    if hasattr(module,'changes_with_reference'):
        return lambda raw,p:module.changes_with_reference(module.state(raw),model,p)[pick]
    return lambda raw,p:module.changes(module.state(raw),model)[pick]


def layer(module):
    previous,update=incumbent.layer(),module.correction()
    def predict(raw,p):
        q=previous(raw,p);q[:,:,2]+=update(raw,q)
        return q
    return predict


def load(module):
    import experiment_periodic_near_carry_v1 as ancestor
    old,update=ancestor.load(),module.layer()
    return lambda raw,p:update(raw,old(raw,p))


def run(module):
    module.OUT.mkdir(exist_ok=True)
    if not (CACHE/'verified.json').is_file():prepare()
    if screen(module):scoring.outer(module,reference=incumbent,preceding=(prior.incumbent,))


if __name__=='__main__':
    torch.set_num_threads(2);prepare()
