"""Reconstruct chronological Y3 forecasts after learning from repeat visits to the same quote."""
import torch

import experiment_y3_same_quote_memory_v1 as incumbent
import y3_same_deadline_references as previous

study, trial = incumbent.study, incumbent.trial
CACHE = incumbent.OUT/'historical_references'


def prepare():
    previous.rebuild(incumbent,previous.references,CACHE)


def references(day,raw):
    return previous.previous.previous.previous.cache.history.rolling.read_verified(CACHE,day,raw)


def outer(module):
    cache = previous.previous.previous.cache
    updates = (previous.previous.incumbent.correction(), previous.incumbent.correction(), incumbent.correction())
    def load(day, raw):
        p, h, n = cache.load(day,raw)
        for update in updates:
            p[:,:,2] += update(raw,p)
            h[:,:,2] += update(raw[:n],h)
        return p,h,n
    cache.scoring.outer(module,reference=incumbent,load_reference=load)


if __name__=='__main__':
    torch.set_num_threads(2)
    prepare()
