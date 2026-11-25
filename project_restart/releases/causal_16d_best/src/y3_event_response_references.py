"""Cross-fitted Y3 forecasts after the lead, contemporaneous, and follow-up revision readout."""
import torch
import y3_price_revision_references as previous
import experiment_y3_event_forecast_response_shape_v1 as incumbent

screen,trial=previous.screen,previous.trial
CACHE=incumbent.OUT/'historical_references'


def prepare():
    previous.previous.rebuild(incumbent,previous.references,CACHE)


def references(d,raw):
    return previous.previous.cache.history.rolling.read_verified(CACHE,d,raw)


def outer(mod):
    first=previous.incumbent.correction();second=incumbent.correction()
    mono=previous.previous.incumbent.correction()
    def load(day,raw):
        if (previous.cache.CACHE/'verified.json').is_file():
            p,h,n=previous.cache.load(day,raw)
        else:
            p,h,n=previous.previous.cache.load(day,raw)
            for update in (mono,first):
                p[:,:,2]+=update(raw,p);h[:,:,2]+=update(raw[:n],h)
        p[:,:,2]+=second(raw,p);h[:,:,2]+=second(raw[:n],h)
        return p,h,n
    previous.cache.scoring.outer(mod,reference=incumbent,load_reference=load)


if __name__=='__main__':
    torch.set_num_threads(2);prepare()
