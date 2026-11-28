import numpy as np

import experiment_dense_analogue_block_payoff_v1 as model

import prepare_dense_disjoint_settled_error_reference_v1 as parent

OUT=model.REFERENCE.parent/'dense_analogue_block_verified_reference_v1'

def read_p(day):
    return parent.read_p(day) if day<6 else np.load(OUT/f'rolling_{day:02d}.npy')
