"""Reconstruct forward-fold reference predictions after the Y3 event-token readout."""
import torch

import experiment_y3_quote_event_tokens_v1 as incumbent
import y3_quote_memory_references as previous

study, trial = previous.study, previous.trial
CACHE = incumbent.OUT/'historical_references'


def prepare():
    previous.previous.rebuild(incumbent, previous.references, CACHE)


def references(day, raw):
    rolling = previous.previous.previous.previous.previous.cache.history.rolling
    return rolling.read_verified(CACHE, day, raw)


def outer(module):
    chain = previous.previous
    cache = chain.previous.previous.cache
    updates = (chain.previous.incumbent.correction(), chain.incumbent.correction(),
               previous.incumbent.correction(), incumbent.correction())
    def load(day, raw):
        p, h, n = cache.load(day, raw)
        for update in updates:
            p[:, :, 2] += update(raw, p)
            h[:, :, 2] += update(raw[:n], h)
        return p, h, n
    cache.scoring.outer(module, reference=incumbent, load_reference=load)


if __name__ == '__main__':
    torch.set_num_threads(2)
    prepare()
