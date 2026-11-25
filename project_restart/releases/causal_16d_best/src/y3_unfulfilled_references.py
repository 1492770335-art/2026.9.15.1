"""Roll the three price-error readout forward on training dates before testing new Y3 specialists."""
import torch
import experiment_y3_unfulfilled_price_moves_v1 as incumbent
import y3_surprise_references as previous

rolling = previous.rolling
CACHE = incumbent.OUT / 'historical_references'


def prepare():
    rolling.rebuild(incumbent, previous.references, CACHE)


def references(day, raw):
    return rolling.read_verified(CACHE, day, raw)


def outer(module):
    stages = (rolling.incumbent, previous.parent.previous.incumbent, previous.parent.incumbent,
              previous.incumbent, incumbent)
    updates = [stage.correction() for stage in stages]

    def load(day, raw):
        full, prefix, n = rolling.cached.load(day, raw)
        for update in updates:
            full[:, :, 2] += update(raw, full)
            prefix[:, :, 2] += update(raw[:n], prefix)
        return full, prefix, n

    rolling.cached.scoring.outer(module, reference=incumbent, load_reference=load)


def run(module):
    if not (CACHE / 'verified.json').exists():
        prepare()
    previous.parent.screen.run(module, references, outer)


if __name__ == '__main__':
    torch.set_num_threads(2)
    prepare()
