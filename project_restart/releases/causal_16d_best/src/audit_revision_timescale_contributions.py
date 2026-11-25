"""Attribute each target's change in Pearson correlation to temporal bands and model stages."""
import json
from math import comb

import numpy as np
import torch

import experiment_joint_revision_timescale_transfer_v1 as model

BANDS = ('new versus six-tick mean','six versus sixty-four tick mean','persistent component','short-long agreement')
STAGES = tuple(f'{a.incumbent.OUT.name} -> {b.incumbent.OUT.name}' for a,b in zip(model.CHECKPOINTS,model.CHECKPOINTS[1:]))


def coalition_scores(gram,groups):
    cov = model.stack.covariance(gram)
    n = len(groups)
    scores = []
    for bitset in range(1<<n):
        v = np.zeros(len(cov)-1)
        v[0] = 1.
        for j in range(n):
            if bitset & (1<<j):
                v[groups[j]] = 1.
        scores.append(v@cov[:-1,-1]/np.sqrt((v@cov[:-1,:-1]@v)*cov[-1,-1]))
    attribution = []
    for j in range(n):
        total = 0.
        for subset in range(1<<n):
            if subset & (1<<j):
                continue
            size = bin(subset).count('1')
            total += (scores[subset | (1<<j)]-scores[subset])/(n*comb(n-1,size))
        attribution.append(total)
    assert abs(sum(attribution)-(scores[-1]-scores[0]))<2.e-12
    return np.asarray(attribution)


def row_moments(raw,anchors,w,y):
    p = anchors[-1]
    features = model.design(anchors)[:,:,3:15]
    stats = []
    for k in range(3):
        q = p[:,:,k:k+1]*(1.+w[k,0])
        terms = features*w[k,4:16]
        row = np.concatenate((np.ones_like(q),q,terms,y[:,:,k:k+1]),axis=2).reshape(-1,15).astype(float)
        stats.append(row.T@row)
    return stats


def report(grams):
    bytime = np.stack([coalition_scores(m,[1+3*j+np.arange(3) for j in range(4)]) for m in grams])
    bystage = np.stack([coalition_scores(m,[1+j+np.arange(4)*3 for j in range(3)]) for m in grams])
    return dict(temporal_band=bytime.tolist(),model_stage=bystage.tolist(),IC_gain=bytime.sum(1).tolist())


def run():
    chosen = json.loads((model.OUT/'results.json').read_text())['selected']
    rows,hist = [],[]
    for day in range(8,16):
        if day>=10:
            if day%2==0:
                w = np.stack([model.fit(hist,k)[1][i] for k,i in enumerate(chosen)])
            raw = model.trial.clock.book.rawday(day)
            y = model.trial.clock.labels(day).reshape(-1,50,3)
            refs = [h.references(day,raw) for h in model.CHECKPOINTS]
            rows.append([row_moments(raw,anchors,w,y) for anchors in zip(*refs)])
        hist.append(model.records(day))
    with np.load(model.OUT/'model.npz') as f:
        w = f['w']
    holdout = []
    for day in range(16,20):
        raw = model.trial.clock.book.rawday(day)
        y = model.trial.clock.labels(day).reshape(-1,50,3)
        anchors = [h.read(day,raw)[0] for h in model.CHECKPOINTS]
        holdout.append(row_moments(raw,anchors,w,y))
    valid = report(np.sum(holdout,0))
    actual = json.loads((model.OUT/'reload_verification.json').read_text())['pooled']
    np.testing.assert_allclose(valid['IC_gain'],np.asarray(actual['candidate'])-actual['base'],atol=1e-8,rtol=0)
    data = dict(temporal_bands=BANDS,model_stages=STAGES,rows_are=['Y1','Y2','Y3'],
                method='Exact Shapley attribution of the frozen linear readout, retaining its incumbent coefficient.',
                training_references=[report(x) for x in np.sum(rows,0)],validation=valid,
                daily={day+17:report(g) for day,g in enumerate(holdout)})
    (model.OUT/'component_attribution.json').write_text(json.dumps(data,indent=2))
    print('temporal-band IC gains by target',np.asarray(valid['temporal_band']).round(7),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
