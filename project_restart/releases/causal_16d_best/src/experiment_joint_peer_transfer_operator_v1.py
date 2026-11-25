"""Estimate cross-stock transfers from out-of-fold forecast errors and check the transfer directions across days."""
import json
import sys
import numpy as np
import torch

import experiment_joint_factor_order_calibration_v1 as replay
import experiment_y3_risk_subspace_calibration_v1 as risk
import y3_sibling_spread_references as history

trial, parent, stack = history.trial, history.incumbent, history.stack
OUT = trial.clock.ART / 'joint_peer_transfer_operator_v1'
ARMS = ('unchanged', 'same risk direction', 'ridge cross direction', 'six transfer directions',
        'directions repeated across days', 'common to individual', 'individual to common',
        'rotated stock identities', 'individual to individual', 'across common directions')
U = risk.geometry
DIM = 150


def coordinates(panel, basis):
    return (basis.T @ panel).reshape(len(panel), DIM)


def moments(p, y, basis):
    z = np.concatenate((np.ones((len(p), 1)), coordinates(p, basis), coordinates(y, basis)), axis=1)
    return z.T @ z


def record(d):
    raw = trial.clock.book.rawday(d)
    y = trial.clock.labels(d).reshape(-1, 50, 3)
    basis = U()[0].astype(float)
    return [moments(p, y, basis) for p in history.references(d, raw)]


def unit(k):
    eye = np.zeros((50, 3, 50))
    eye[:, k] = np.eye(50)
    return eye.reshape(DIM, 50)


def terms(m, k):
    yy = 1 + DIM + np.arange(k, DIM, 3)
    return m[0,0], m[0, 1:1+DIM], m[0, yy], m[1:1+DIM, 1:1+DIM], m[1:1+DIM, yy], m[yy,yy].sum()


def correlation(m, w, k, slope=False):
    n, mx, my, gram, xy, sy = terms(m, k)
    ones = U()[0].astype(float).sum(0)
    sum_pred = mx @ w @ ones
    sum_y = my @ ones
    var = (w*(gram @ w)).sum() - sum_pred**2 / (50*n)
    cov = np.sum(w*xy) - sum_pred*sum_y / (50*n)
    if slope:
        return cov/max(var, 1.e-12)
    return cov/np.sqrt(var*(sy - sum_y**2/(50*n)))


def readouts(records, k):
    daily = np.mean(records, axis=1)
    base = unit(k)
    same = np.repeat(np.eye(50, dtype=bool), 3, axis=0)
    bank = []
    grads = []
    for m in daily:
        n, _, _, gram, xy, _ = terms(m, k)
        beta = correlation(m, base, k, slope=True)
        grads.append((xy / np.clip(beta, .35, 3) - gram @ base)/n)
        bank.append(gram/n)
    gram = np.mean(bank, axis=0)
    scale = np.sqrt(gram.diagonal().clip(.00001))
    g = gram/scale[:, None]/scale[None, :]
    grad = np.asarray(grads)/scale[None,:,None]
    avg = grad.mean(0)
    # Shrink uncertain entries before extracting a low-rank transfer operator.
    err = grad.std(0)/np.sqrt(len(grad))
    shrunk = avg*abs(avg)/(abs(avg)+err+1.e-7)
    ridge = np.linalg.inv(g + 1.8*np.eye(DIM))
    local = shrunk*same
    cross = shrunk*(~same)
    left, s, right = np.linalg.svd(cross, full_matrices=False)
    left, s, right = left[:, :6], s[:6], right[:6]
    measured = np.einsum('ia,dij,aj->da', left, grad*(~same), right)
    halves = [measured[::2].mean(0), measured[1::2].mean(0),
              measured[:len(grad)//2].mean(0), measured[len(grad)//2:].mean(0)]
    reliability = (np.min(halves, axis=0)/s.clip(1.e-7)).clip(0,1)
    def solve(rhs):
        return .35*(ridge @ rhs)/scale[:,None]
    def bounded(rhs, mask):
        return solve(rhs*mask)*mask
    source = np.repeat(np.arange(50), 3)[:, None]
    dest = np.arange(50)[None, :]
    low_high = (source < 17) & (dest >= 17)
    high_low = (source >= 17) & (dest < 17)
    reprod = (left*(s*reliability)) @ right
    compact = bounded(reprod, ~same)
    # Conjugation by a stock permutation preserves the size and rank of the map.
    basis = U()[0].astype(float)
    rotate = basis.T @ np.roll(basis, 17, axis=0)
    placebo = np.einsum('ij,jhk,lk->ihl', rotate, compact.reshape(50,3,50), rotate).reshape(DIM,50)
    return np.array([np.zeros_like(base), bounded(local,same), bounded(cross,~same),
                     bounded((left*s) @ right,~same), compact, bounded(cross,low_high),
                     bounded(cross,high_low), placebo, bounded(cross,(source>=17)&(dest>=17)&~same),
                     bounded(cross,(source<17)&(dest<17)&~same)])


def gains(m, matrices, k):
    base = unit(k)
    old = correlation(m, base, k)
    return np.array([correlation(m, base+w, k)-old for w in matrices])


def train(module=None):
    module = module or sys.modules[__name__]
    module.OUT.mkdir(exist_ok=True)
    make_record = getattr(module, 'record', record)
    records = np.asarray([make_record(d) for d in range(8,16)])
    picks, weights, reports = [], [], []
    for k in range(3):
        folds, daily = [], []
        for n in (2,4,6):
            models = module.readouts(records[:n],k)
            folds.append([gains(m, models, k) for m in records[n:n+2].sum(0)])
            daily.extend([[gains(m, models, k) for m in day] for day in records[n:n+2]])
        f, d = np.array(folds), np.array(daily)
        avg = f.mean(0)
        valid = [j for j in (2,3,4,5,6,8,9) if (avg[:,j]>0).all() and (avg[:,j]>avg[:,7]).all()
                 and (avg[:,j]>avg[:,1]).all() and (f[-1,:,j]>0).all() and (f[:,:,j]>0).sum(0).min()>=2
                 and (d[:,:,j]>0).sum(0).min()>=4]
        pick = max(valid, key=lambda j:avg[:,j].min()) if valid else 0
        picks.append(pick)
        weights.append(module.readouts(records,k)[pick])
        reports.append(dict(folds=f.tolist(), daily=d.tolist(), mean=avg.tolist()))
        print('target',k+1,module.ARMS[pick],np.round(avg,6),flush=True)
    (module.OUT/'results.json').write_text(json.dumps(dict(selected=picks, arms=module.ARMS, targets=reports)))
    if any(picks):
        np.savez(module.OUT/'model.npz', w=np.stack(weights,axis=2), basis=U()[0], selected=picks)
    return any(picks)


def adjustment(output=OUT):
    with np.load(output/'model.npz') as archive:
        w, basis = archive['w'], archive['basis'].astype(float)
    def apply(p):
        x = coordinates(p,basis)
        z = (x @ w.reshape(DIM,-1)).reshape(len(p),50,3)
        return p+(basis @ z).astype('float32')
    return apply


def layer():
    before, adjust = parent.layer(), adjustment()
    return lambda raw,initial: adjust(before(raw,initial))


def replay_with():
    adjust = adjustment()
    def read(d,raw):
        p,h,n = history.read(d,raw)
        return adjust(p),adjust(h),n
    return stack.verifier,read


if __name__ == '__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['replay'] or train():
        replay.replay(sys.modules[__name__])
