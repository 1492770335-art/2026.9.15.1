"""Fit a return-path readout under exact short-horizon Pearson-contribution constraints."""
import json
import sys

import numpy as np
import torch
from scipy.optimize import minimize

import experiment_y3_frequency_path_loss_v1 as path

parent,base,history = path.parent,path.base,path.history
OUT = parent.OUT.parent/'y3_protected_path_decoder_v1'
MODES = ('unconstrained endpoint','protected first twenty','protected two early intervals','protected on each training half',
         'distant only','distant with protected first twenty','distant with protected early intervals',
         'distant with protected first twenty on each half','endpoint protecting both bands on each half')


@torch.inference_mode()
def decoded(raw,models):
    x,phase = path.featurizer.features(raw,np.arange(len(raw)))
    result = []
    for start in range(0,len(x),96):
        v = torch.from_numpy(x[start:start+96]).cuda()
        ph = torch.from_numpy(phase[start:start+96]).cuda()
        outputs = [net(v,ph) for net in models]
        pieces = []
        for channel in (0,1):
            def endpoints(output):
                value = output[channel]
                return torch.stack((value[...,0],value[...,3]-value[...,0],value[...,7]-value[...,3],
                                    value[...,11]-value[...,7]),-1)
            point,shuffled,ordered,split = [endpoints(o) for o in outputs]
            signals = torch.cat((point,ordered-point,ordered-shuffled,split-ordered),-1)
            if channel == 1:
                signals = signals[:,None].expand(-1,50,-1)
            pieces.append(signals)
        result.append(torch.cat(pieces,-1).cpu().numpy())
    return np.concatenate(result)


def channels(z,p):
    mean = p.mean(1,keepdims=True)
    return np.concatenate((z,p-mean,np.broadcast_to(mean,p.shape)),axis=2)


def moment(p,y,z):
    m = 0.
    for start in range(0,len(z),256):
        sl = slice(start,start+256)
        a = y[sl,:,0]*path.SCALE[2]/path.SCALE[0]
        near = y[sl,:,1]*path.SCALE[2]/path.SCALE[1]
        parts = np.stack((a,near-a,y[sl,:,2]-near),2)
        bank = np.concatenate((np.ones_like(parts[:,:,:1]),p[sl,:,2:],parts,channels(z[sl],p[sl])),2)
        bank = bank.reshape(-1,bank.shape[-1]).astype(float)
        m += bank.T@bank
    return m


def changes(c,w):
    dev = np.sqrt(max(1e-14,c[0,0]+2*c[4:,0]@w+w@c[4:,4:]@w))
    ydev = np.sqrt(c[1:4,1:4].sum())
    return ((c[0,1:4]+w@c[4:,1:4])/dev-c[0,1:4]/np.sqrt(c[0,0]))/ydev


def fit(rows):
    grams = np.asarray(rows)
    cs = [base.covariance(v) for v in grams.sum(0)]
    pooled = np.mean(cs,axis=0)
    z = pooled[4:,4:]
    scales = np.sqrt(z.diagonal().clip(1e-8))
    eig,vec = np.linalg.eigh(z/np.outer(scales,scales))
    basis = (vec/np.sqrt(eig.clip(0)+.35))/scales[:,None]*np.sqrt(pooled[0,0])
    def convert(c):
        t = np.eye(len(c))
        t[4:,4:] = basis
        return t.T@c@t
    all_cov = [convert(c) for c in cs]
    halves = [convert(base.covariance(g)) for subset in np.array_split(grams,2) for g in subset.sum(0)]
    def combined(c,v,band):
        old = c[0,1:4]@band
        col = c[4:,1:4]@band
        ystd = np.sqrt(c[1:4,1:4].sum())
        zz = c[4:,4:]@v
        dev = np.sqrt(max(1e-14,c[0,0]+(2*c[4:,0]+zz)@v))
        num = old+col@v
        gain = (num/dev-old/np.sqrt(c[0,0]))/ystd
        grad = (col/dev-num*(c[4:,0]+zz)/dev**3)/ystd
        return gain,grad
    full = np.ones(3)
    weights = []
    for mode in range(len(MODES)):
        target_band = np.array([0.,0.,1.]) if 4<=mode<=7 else full
        def objective(v):
            scores,derivatives = zip(*(combined(c,v,target_band) for c in all_cov))
            return -np.mean(scores)+1.25*v@v,-np.mean(derivatives,axis=0)+2.5*v
        bands = (np.array([1.,1.,0]),) if mode in (1,3,5,7) else tuple(np.eye(3)[:2])
        constraints = []
        if mode==8:
            bands = (np.array([1.,1.,0]),np.array([0.,0.,1.]))
        if mode not in (0,4):
            for cov in (halves if mode in (3,7,8) else all_cov):
                for b in bands:
                    constraints.append(dict(type='ineq',fun=lambda v,c=cov,b=b:combined(c,v,b)[0],
                                            jac=lambda v,c=cov,b=b:combined(c,v,b)[1]))
        opt = minimize(objective,np.zeros(len(basis)),jac=True,method='SLSQP',constraints=constraints,
                       options={'maxiter':120,'ftol':1e-12})
        if not opt.success:
            raise RuntimeError(opt.message)
        for con in constraints:
            assert con['fun'](opt.x)>=-1e-9
        weights.append(basis@opt.x)
    return np.stack(weights)


def train():
    OUT.mkdir(exist_ok=True)
    rows,training = [],[]
    for day in range(16):
        if day in (4,8,12):
            nets = path.fit(training)
            print('decoder fit through day',day,flush=True)
        if day>=4:
            raw = history.trial.clock.book.rawday(day)
            y = history.trial.clock.labels(day).reshape(-1,50,3).astype(float)
            z = decoded(raw,nets)
            if day==4:
                np.testing.assert_allclose(z[:137],decoded(raw[:137],nets),atol=6e-6,rtol=0)
            ps = history.references(day,raw) if day>=8 else parent.parent.warm.read(day,raw)
            rows.append([moment(p,y,z) for p in ps])
        training.append(path.data(day))
    np.savez(OUT/'training_moments.npz',days=np.arange(4,16),grams=np.asarray(rows))
    pick = screen(rows)
    return save_selected(pick,training)


def save_selected(pick,training):
    if pick is not None:
        rows = np.load(OUT/'training_moments.npz')['grams']
        models = path.fit(training)
        params = {str(i)+'_'+k:v.cpu().numpy() for i,m in enumerate(models) for k,v in m.state_dict().items()}
        np.savez(OUT/'model.npz',w=fit(rows)[pick],n=training[0][0].shape[-1],**params)
    return pick is not None


def screen(rows):
    folds,daily = [],[]
    for first in (4,8):
        weights = fit(rows[:first])
        group = rows[first:first+4]
        def evaluate(pair):
            return [[changes(base.covariance(m),w) for w in weights] for m in pair]
        daily.extend(evaluate(pair) for pair in group)
        folds.append(evaluate(np.sum(group,0)))
    f,ds = np.asarray(folds),np.asarray(daily)
    total,near,late = f.sum(3),f[:,:,:,:2].sum(3),f[:,:,:,2]
    good = [j for j in range(1,len(MODES)) if j!=4 and total[:,:,j].min()>0 and near[:,:,j].min()>=0
            and late[:,:,j].min()>0 and (ds[:,:,j].sum(2)>0).sum(0).min()>=5]
    pick = max(good,key=lambda j:total[:,:,j].min()) if good else None
    strict = pick
    primary = MODES.index('distant with protected first twenty')
    if pick is None and total[:,:,primary].min()>0 and late[:,:,primary].min()>0:
        pick = primary
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,strict_both_bands=strict,
        primary='distant with protected first twenty',modes=MODES,horizons=[[1,5],[6,20],[21,60]],
        windows=[[9,12],[13,16]],folds=f.tolist(),daily=ds.tolist()),indent=2))
    print('selected',pick,'near and remote Y3 contribution',np.stack((near,late),-1).round(6).tolist(),flush=True)
    return pick


def correction():
    with np.load(OUT/'model.npz') as f:
        models = [path.Decoder(int(f['n'])).cuda().eval() for _ in path.MODES]
        for i,net in enumerate(models):
            net.load_state_dict({k:torch.from_numpy(f[str(i)+'_'+k]) for k in net.state_dict()})
        w = f['w']
    def update(raw,p):
        return (channels(decoded(raw,models),p)@w).astype('float32')
    return update


def layer():
    before,update = parent.layer(),correction()
    def predict(raw,initial):
        p = before(raw,initial).copy()
        p[:,:,2] += update(raw,p)
        return p
    return predict


def replay_with():
    update = correction()
    def read(day,raw):
        p,h,n = history.read(day,raw)
        p[:,:,2] += update(raw,p)
        h[:,:,2] += update(raw[:n],h)
        return p,h,n
    return history.stack.verifier,read


if __name__=='__main__':
    if sys.argv[1:]==['screen']:
        selected = screen(np.load(OUT/'training_moments.npz')['grams'])
        ready = selected is not None and save_selected(selected,[path.data(d) for d in range(16)])
    else:
        ready = sys.argv[1:]==['replay'] or train()
    if ready:
        base.replay.replay(sys.modules[__name__])
