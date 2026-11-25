"""Select the strength of the common paired-column level on training dates with a paired fold comparison."""
import sys

import numpy as np
import torch

import experiment_y3_paired_gross_flow_v1 as gross
import experiment_y3_order_response_filter_v1 as source
import experiment_y3_order_filter_strength_v1 as previous
import y3_filter_validation as trial

OUT=trial.clock.ART/'y3_paired_level_strength_v1'
ARMS=['base amplitude','double','quadruple','eightfold']


def state(raw):
    p,a,_=gross.components(raw)
    return gross.compact_filters(p,a)


observe=source.observe
fit=source.fit


def changes(z,model):
    q=np.concatenate([x.astype('float32')@model['w'][:,2] for x in np.array_split(z,max(1,(len(z)+127)//128))])
    return [.32*a*q for a in (1,2,4,8)]


select=previous.select_amplitude
def correction():return trial.correction(sys.modules[__name__])
def layer():return trial.layer(sys.modules[__name__])
def load():return trial.load(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2);trial.run(sys.modules[__name__])
