"""Rebuild forward-fold Y3 predictions after classifying estimated fills at the previous quote."""
import torch

import experiment_y3_inferred_print_impacts_v1 as incumbent
import y3_quote_token_references as previous
import y3_expanding_path_references as rolling
from y3_same_deadline_references import rebuild

trial, study = previous.trial, previous.study
CACHE = incumbent.OUT/'historical_references'


def prepare():
    rebuild(incumbent, previous.references, CACHE)


def references(day, raw):
    return rolling.read_verified(CACHE, day, raw)


def replay_with(extra=()):
    quote_memory = previous.previous
    deadline = quote_memory.previous
    cache = deadline.previous.previous.cache
    updates = tuple(m.correction() for m in (deadline.previous.incumbent, deadline.incumbent,
                    quote_memory.incumbent, previous.incumbent, incumbent, *extra))
    def load(day, raw):
        p, prefix, n = cache.load(day, raw)
        for fn in updates:
            p[:, :, 2] += fn(raw, p)
            prefix[:, :, 2] += fn(raw[:n], prefix)
        return p, prefix, n
    return cache.scoring, load


def outer(module):
    scoring, load = replay_with()
    scoring.outer(module, reference=incumbent, load_reference=load)


if __name__=='__main__':
    torch.set_num_threads(2)
    prepare()
