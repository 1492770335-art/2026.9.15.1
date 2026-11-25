"""Reconstruct chronological forecasts of the Y3 same-side execution run model."""
import torch

import experiment_y3_trade_spell_survival_v1 as incumbent
import y3_inferred_print_references as previous
import y3_expanding_path_references as rolling
from y3_same_deadline_references import rebuild

study, trial = previous.study, previous.trial
CACHE = incumbent.OUT/'historical_references'


def references(day, raw):
    return rolling.read_verified(CACHE, day, raw)


def outer(module):
    scoring, load = previous.replay_with((incumbent,))
    scoring.outer(module, reference=incumbent, load_reference=load)


if __name__ == '__main__':
    torch.set_num_threads(2)
    rebuild(incumbent, previous.references, CACHE)
