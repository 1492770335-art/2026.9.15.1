import numpy as np

import experiment_dense_market_unpriced_covariance_v1 as model

OUT = model.REFERENCE.parent / 'dense_market_unpriced_verified_reference_v1'

def read_p(day):
    if day < 6:
        return model.ref.read_p(day)
    return np.load(OUT / f'rolling_{day:02d}.npy')
