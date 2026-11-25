"""Rebuild first-16-day rolling references and apply all three verified path updates on outer dates."""
import torch
import experiment_y3_path_magnitude_consensus_v1 as incumbent
import experiment_y3_rank_spacing_calibration_v1 as screen
import y3_scale_conditioned_references as previous

CACHE = incumbent.OUT / 'historical_references'


def prepare():
    previous.previous.rebuild(incumbent, previous.references, CACHE)


def references(day, raw):
    return previous.previous.read_verified(CACHE, day, raw)


def outer(module):
    cached = previous.previous.cached
    corrections = [m.correction() for m in (previous.previous.incumbent, previous.incumbent, incumbent)]

    def load(day, raw):
        full, head, n = cached.load(day, raw)
        for correction in corrections:
            full[:, :, 2] += correction(raw, full)
            head[:, :, 2] += correction(raw[:n], head)
        return full, head, n

    cached.scoring.outer(module, reference=incumbent, load_reference=load)


def run(module):
    if not (CACHE / 'verified.json').exists():
        prepare()
    screen.run(module, references, outer)


if __name__ == '__main__':
    torch.set_num_threads(2)
    prepare()
