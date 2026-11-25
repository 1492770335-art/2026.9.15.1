"""Replay the order response in sibling stock portfolios and reconstruct earlier fold forecasts."""
import sys

import numpy as np
import torch

import experiment_y3_sibling_spread_network_v1 as incumbent
from y3_expanding_path_references import read_verified
from y3_same_deadline_references import rebuild
from y3_touch_clock_references import read_stage, verify_stage

previous = incumbent.history
trial, study, stack = previous.trial, previous.study, previous.stack
CACHE = incumbent.OUT/'historical_references'
FULL = incumbent.OUT/'full_chain_verification'


def references(day,raw):
    return read_verified(CACHE,day,raw)


def read(d,raw):
    return read_stage(FULL,d,raw)


def verify():
    with np.load(incumbent.OUT/'model.npz') as f:
        st = dict(f)
    w = st['w'][:,int(st['pick'])]
    nets = incumbent.networks(st)
    def predict(raw,p):
        cols = incumbent.features(raw,np.arange(len(raw)),st,nets)
        delta = np.einsum('tsc,c->ts',cols.astype(float),w[3:])
        delta += np.einsum('tsc,c->ts',trial.clock.center(p).astype(float),w[:3])
        q = p.copy()
        q[:,:,2] += delta
        return q
    verify_stage(incumbent,previous.read,predict)


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
