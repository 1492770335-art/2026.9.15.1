import numpy as np

import experiment_dense_staggered_book_events_v1 as model

parent = model.risk.ref

OUT = model.REFERENCE.parent / 'dense_staggered_book_events_verified_reference_v1'

def read_p(day):
    return parent.read_p(day) if day < 6 else np.load(OUT / f'rolling_{day:02d}.npy')
