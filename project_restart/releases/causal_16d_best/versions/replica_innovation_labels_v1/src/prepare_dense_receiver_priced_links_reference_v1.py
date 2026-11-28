import numpy as np

import prepare_dense_horizon_graph_reference_v1 as parent

import experiment_dense_receiver_priced_book_links_v1 as model

OUT = model.REFERENCE.parent / 'dense_receiver_priced_links_verified_reference_v1'

def read_p(day):
    return parent.read_p(day) if day<6 else np.load(OUT/f'rolling_{day:02d}.npy')
