"""Chronological Y3 forecasts including the conditional price-surprise response."""
import torch

import experiment_y3_expected_price_paths_v1 as incumbent
import y3_consensus_references as parent
import y3_expanding_path_references as rolling

CACHE=incumbent.OUT/'historical_references'


def prepare():
    rolling.rebuild(incumbent,parent.references,CACHE)


def references(day,raw):
    return rolling.read_verified(CACHE,day,raw)


def outer(module):
    earlier=(rolling.incumbent,parent.previous.incumbent,parent.incumbent,incumbent)
    correctors=[m.correction() for m in earlier]
    def load(day,raw):
        p,h,n=rolling.cached.load(day,raw)
        for update in correctors:
            p[:,:,2]+=update(raw,p)
            h[:,:,2]+=update(raw[:n],h)
        return p,h,n
    rolling.cached.scoring.outer(module,reference=incumbent,load_reference=load)


def run(module):
    if not (CACHE/'verified.json').is_file():
        prepare()
    parent.screen.run(module,references,outer)


if __name__=='__main__':
    torch.set_num_threads(2)
    prepare()
