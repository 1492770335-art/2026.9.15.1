"""Test the previously selected within-jump location contrast after conditioning on predicted activity."""
import sys
import torch
import experiment_y3_implied_crossing_depth_v1 as depth
import y3_impact_mass_references as history

parent,study,trial=history.incumbent,history.study,history.trial
OUT=trial.clock.ART/'y3_clock_and_crossing_depth_v1'
ARMS=depth.ARMS
CANDIDATES=(9,)
CONTROLS=depth.CONTROLS
historical_design,fit,changes=depth.historical_design,depth.fit,depth.changes


def correction():
    return depth.load_correction(OUT)


def layer():
    return study.layer(sys.modules[__name__],parent)


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
