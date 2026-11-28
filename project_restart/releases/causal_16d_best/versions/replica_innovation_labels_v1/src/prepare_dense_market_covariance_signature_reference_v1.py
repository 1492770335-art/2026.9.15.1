import numpy as np

import experiment_dense_market_covariance_signature_v1 as model

OUT=model.REFERENCE.parent/'dense_market_covariance_signature_verified_reference_v1'

def read_p(day):
    return model.read_p(day) if day<6 else np.load(OUT/f'rolling_{day:02d}.npy')
