"""Verify the three-age readout and rebuild its strictly past-only training forecasts."""
import sys

import numpy as np
import torch

import experiment_y3_touch_delay_readout_v1 as incumbent
from y3_expanding_path_references import read_verified
from y3_same_deadline_references import rebuild

previous = incumbent.history
trial, study, stack = previous.trial, previous.study, previous.stack
CACHE = incumbent.OUT/'historical_references'
FULL = incumbent.OUT/'full_chain_verification'


def references(d, raw):
    return read_verified(CACHE,d,raw)


def read(d,raw):
    return previous.read_stage(FULL,d,raw)


def verify():
    with np.load(incumbent.OUT/'model.npz') as archive:
        st = dict(archive)
    pick = int(st['pick'])
    w = st['w'][:,pick]
    def predict(raw,p):
        z = incumbent.features(raw,np.arange(len(raw)),st,st['heads'])
        delta = np.einsum('tsi,i->ts',z.astype(float),w[3:])
        delta += np.einsum('tsi,i->ts',trial.clock.center(p).astype(float),w[:3])
        q = p.copy()
        q[:,:,2] += delta
        return q
    previous.verify_stage(incumbent, previous.read, predict)


def replay_with():
    return stack.verifier,read


def outer(module):
    stack.verifier.outer(module,reference=incumbent,load_reference=read)


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['verify']:
        verify()
    else:
        rebuild(incumbent,previous.references,CACHE)
