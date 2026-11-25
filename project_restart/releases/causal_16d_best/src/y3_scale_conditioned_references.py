"""Chronological references after the direction-and-magnitude forecast was fitted."""
import torch

import experiment_y3_scale_conditioned_path_v1 as incumbent
import y3_expanding_path_references as previous

CACHE=incumbent.OUT/'historical_references'


def prepare():
    previous.rebuild(incumbent, previous.references, CACHE)


def references(day,raw):
    return previous.read_verified(CACHE, day, raw)


def outer(module):
    cached=previous.cached
    before=previous.incumbent.correction()
    current=incumbent.correction()
    def load(day,raw):
        full,head,n=cached.load(day,raw)
        for f in (before,current):
            full[:,:,2]+=f(raw,full)
            head[:,:,2]+=f(raw[:n],head)
        return full,head,n
    cached.scoring.outer(module,reference=incumbent,load_reference=load)


def run(module):
    if not (CACHE/'verified.json').exists():
        prepare()
    previous.incumbent.history.previous.previous.previous.incumbent.screen.run(module,references,outer)


if __name__=='__main__':
    torch.set_num_threads(2)
    prepare()
