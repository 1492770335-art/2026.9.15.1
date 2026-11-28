import numpy as np

import prepare_dense_receiver_priced_links_reference_v1 as parent

import experiment_dense_revisited_price_pressure_v1 as model

OUT = model.REFERENCE.parent/'dense_revisited_pressure_verified_reference_v1'

def read_p(day):
    return parent.read_p(day) if day<6 else np.load(OUT/f'rolling_{day:02d}.npy')
