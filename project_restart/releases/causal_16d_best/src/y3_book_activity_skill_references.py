"""Independently evaluate the activity-conditioned readout and reconstruct the rolling forecasts used downstream."""
import sys
import numpy as np
import torch
import experiment_y3_book_activity_skill_v1 as incumbent
from y3_touch_clock_references import read_stage,verify_stage
from y3_expanding_path_references import read_verified

previous = incumbent.history
trial,stack,study = previous.trial,previous.stack,previous.study
FULL = incumbent.OUT/'full_chain_verification'
CACHE = incumbent.OUT/'historical_references'


def read(d,raw):
    return read_stage(FULL,d,raw)


def references(d,raw):
    return read_verified(CACHE,d,raw)


def replay_with():
    return stack.verifier,read


def independent(raw,p,st):
    assert int(st['pick'])==6
    changed = abs(raw[:,:,0].astype(float)-.00136)>.022
    def average(span):
        y = np.empty(changed.shape,float)
        rho = np.exp(-1/span)
        for i,row in enumerate(changed):
            y[i] = row if i in (0,2401) else rho*y[i-1]+(1-rho)*row
        return y
    s = average(128)
    l = np.tanh(1.5*(average(8)-s)/np.sqrt(.02+s*(1-s)))
    z = trial.clock.center(p.astype(float))
    q = p.copy()
    risk = st['risk']
    H = risk+.8*np.diag(risk.diagonal().clip(.001))
    for side in (0,1):
        own,base,agree = incumbent.peers.views(z,side)
        peer = z[:,side::2]
        a = l[:,1-side::2]
        views = [incumbent.peers.geometry(peer,1+v*.85*l[:,side::2]) for v in (1,-1)]
        b = st['liquid']
        responses = []
        for k in (0,6,9):
            moment = base@b[:13,k:k+3]+(views[0]-views[1])@b[13:,k:k+3]
            coef = np.linalg.solve(H,moment.T).T
            norm = np.sqrt((coef*(coef@risk)).sum(1)).clip(1e-8)
            responses.append(np.einsum('nsi,ni->ns',own,coef)*(.55*np.minimum(1.,.18*np.sqrt(risk[2,2])/norm))[:,None])
        q[:,1-side::2,2] += (responses[0]+a*(responses[1]+agree*responses[2])).astype('float32')
    return q


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['verify']:
        with np.load(incumbent.OUT/'model.npz') as f:
            st = dict(f)
        verify_stage(incumbent,previous.read,lambda raw,p:independent(raw,p,st))
    elif sys.argv[1:]==['audit']:
        import audit_early_book_horizon_gain_v1 as horizons
        import audit_y3_release_gain_v1 as gain
        gain.run(incumbent,read,previous.incumbent,previous.read)
        gain.run(incumbent,read)
        horizons.run(sys.modules[__name__],previous)
    else:
        incumbent.train.rebuild(incumbent,CACHE)
