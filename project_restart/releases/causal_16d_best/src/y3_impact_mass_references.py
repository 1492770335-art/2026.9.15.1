"""Rebuild forward-only training predictions after the event activity correction."""
import torch

import experiment_y3_impact_mass_forecast_v1 as incumbent
import y3_crossed_level_references as previous

study,trial=previous.study,previous.trial
CACHE=incumbent.OUT/'historical_references'


def references(day,raw):
    return previous.rolling.read_verified(CACHE,day,raw)


def replay_with(extra=()):
    return previous.replay_with((incumbent,*extra))


def outer(module):
    scoring,load=replay_with()
    scoring.outer(module,reference=incumbent,load_reference=load)


if __name__=='__main__':
    torch.set_num_threads(2)
    previous.rebuild(incumbent,previous.references,CACHE)
