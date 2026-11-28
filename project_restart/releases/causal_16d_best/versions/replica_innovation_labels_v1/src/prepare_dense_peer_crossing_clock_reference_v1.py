import numpy as np

import experiment_dense_peer_crossing_clock_v1 as model

parent = model.ref

OUT = model.REFERENCE.parent/'dense_peer_crossing_clock_verified_reference_v1'

def read_p(day):
    return parent.read_p(day) if day<6 else np.load(OUT/f'rolling_{day:02d}.npy')
