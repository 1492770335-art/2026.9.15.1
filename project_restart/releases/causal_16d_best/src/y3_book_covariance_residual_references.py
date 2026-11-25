"""Independently verify the curvature term and rebuild its rolling predictions."""
import sys
import numpy as np
import torch
import experiment_y3_book_covariance_residuals_v1 as incumbent
import y3_future_book_comoment_references as previous
from y3_touch_clock_references import read_stage,verify_stage
from y3_expanding_path_references import read_verified

trial,study,stack=previous.trial,previous.study,previous.stack
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


def verify():
    with np.load(incumbent.OUT/'model.npz') as f:
        means,pay,linear,w=f['means'],f['pay'],f['read'][:,1],f['w']
        assert int(f['pick'])==3 and np.count_nonzero(w[3:])==1
    def independent(raw,p):
        q=p.copy()
        for t in np.array_split(np.arange(len(p)),max(1,(len(p)+639)//640)):
            x=incumbent.base.observations(raw,t)[0].astype(float)
            a=x@means[:,:8];b=x@means[:,8:]
            quadratic=np.einsum('tsi,ij,tsj->ts',a,pay,b,optimize=True)
            curvature=trial.clock.center(x@linear-quadratic)
            q[t,:,2]+=p[t]@w[:3]+curvature*w[5]
        return q
    verify_stage(incumbent,previous.read,independent)


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['verify']:
        verify()
    else:
        previous.rebuild(incumbent,previous,CACHE)
