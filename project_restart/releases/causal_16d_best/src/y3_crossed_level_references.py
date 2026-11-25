"""Replay chronological Y3 forecasts after adding the crossing-time book correction."""
import torch

import experiment_y3_crossed_level_book_v1 as incumbent
import y3_expanding_path_references as rolling
import y3_recipient_exposure_references as previous
from y3_same_deadline_references import rebuild

study,trial=previous.study,previous.trial
CACHE=incumbent.OUT/'historical_references'


def references(day,raw):
    return rolling.read_verified(CACHE,day,raw)


def replay_with(extra=()):
    return previous.replay_with((incumbent,*extra))


def outer(module):
    scoring,load=replay_with()
    scoring.outer(module,reference=incumbent,load_reference=load)


if __name__=='__main__':
    torch.set_num_threads(2)
    rebuild(incumbent,previous.references,CACHE)
