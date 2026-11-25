"""Replay the frozen field-amplitude model for a full day and its verification prefix."""
from functools import lru_cache
import experiment_joint_field_state_amplitudes_v1 as incumbent


@lru_cache(maxsize=1)
def reader():
    return incumbent.replay_with()[1]


def read(day, raw):
    return reader()(day, raw)
