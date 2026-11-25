"""Read rolling forecasts after the activity and price-level corrections have both been applied."""
import torch
import experiment_y3_clock_and_crossing_depth_v1 as incumbent
import y3_impact_mass_references as previous
from y3_same_deadline_references import rebuild
from y3_expanding_path_references import read_verified

study,trial=previous.study,previous.trial
CACHE=incumbent.OUT/'historical_references'


def references(day,raw):
    return read_verified(CACHE,day,raw)


def replay_with(extra=()):
    return previous.replay_with((incumbent,*extra))


def outer(module):
    scoring,load=replay_with()
    scoring.outer(module,reference=incumbent,load_reference=load)


if __name__=='__main__':
    torch.set_num_threads(2)
    rebuild(incumbent,previous.references,CACHE)
