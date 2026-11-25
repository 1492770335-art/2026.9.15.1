"""Verify the selected Gaussian transport with an independent Cholesky/SVD construction."""
import sys
import numpy as np
import torch
import experiment_y3_horizon_covariance_transport_v1 as incumbent
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


def verify():
    with np.load(incumbent.OUT/'model.npz') as f:
        w = f['w']
    assert set(np.flatnonzero(w[1:])).issubset({0,1,2,6,7,8})
    def predict(raw,p):
        rel = trial.clock.center(p.astype(float))
        q = p.copy()
        adj = w[0]*p[:,:,2].astype(float)
        for side in (0,1):
            peer,own = rel[:,side::2],rel[:,1-side::2]
            g = np.einsum('nsi,nsj->nij',peer,peer)/25
            slow = np.empty_like(g)
            rho = np.exp(-1/96)
            for k in range(len(g)):
                slow[k] = g[k] if k in (0,2401) else rho*slow[k-1]+(1-rho)*g[k]
            a = .75*g+.25*slow+.003*np.eye(3)
            b = slow+.003*np.eye(3)
            x,y = [np.sqrt(m.diagonal(axis1=1,axis2=2)) for m in (a,b)]
            left,right = [np.linalg.cholesky(m/s[:,:,None]/s[:,None,:]) for m,s in ((a,x),(b,y))]
            u,_,v = np.linalg.svd(right.transpose(0,2,1)@left)
            transport = (right@(u@v))@np.linalg.inv(left)
            np.testing.assert_allclose(transport,transport.transpose(0,2,1),rtol=0,atol=1e-9)
            scale = own*(y/x-1)[:,None,:]
            rotated = (own/x[:,None,:])@(transport-np.eye(3))*y[:,None,:]
            adj[:,1-side::2] += scale@w[1:4]+rotated@w[7:10]
        q[:,:,2] += adj
        return q
    verify_stage(incumbent,previous.read,predict)


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['verify']:
        verify()
    elif sys.argv[1:]==['audit']:
        import audit_early_book_horizon_gain_v1 as horizon
        import audit_y3_release_gain_v1 as gain
        gain.run(incumbent,read,previous.incumbent,previous.read)
        gain.run(incumbent,read)
        horizon.run(sys.modules[__name__],previous)
    else:
        incumbent.study.rebuild(incumbent,CACHE)
