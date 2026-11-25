"""Verified chronological references for the event-aligned revision readout."""
import torch
import cache_y3_price_revision_reference as cache
import y3_monotone_references as previous
from cache_y3_price_revision_reference import model as incumbent

screen,trial=previous.screen,previous.trial
CACHE=incumbent.OUT/'historical_references'


def prepare():
    previous.rebuild(incumbent,previous.references,CACHE)


def references(d,raw):
    return previous.cache.history.rolling.read_verified(CACHE,d,raw)


def outer(mod):
    if (cache.CACHE/'verified.json').is_file():
        cache.scoring.outer(mod,reference=incumbent,load_reference=cache.load)
        return
    updates=(previous.incumbent.correction(),incumbent.correction())
    def load(day,raw):
        p,h,n=previous.cache.load(day,raw)
        for update in updates:
            p[:,:,2]+=update(raw,p);h[:,:,2]+=update(raw[:n],h)
        return p,h,n
    cache.scoring.outer(mod,reference=incumbent,load_reference=load)


if __name__=='__main__':
    torch.set_num_threads(2);prepare()
