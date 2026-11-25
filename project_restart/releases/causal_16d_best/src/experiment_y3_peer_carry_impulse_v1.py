"""Test whether a new versus already persistent peer forecast has a different terminal response."""
import sys
import numpy as np
import torch

import experiment_y3_peer_carry_context_v1 as context

parent,trial,history,stack=context.parent,context.trial,context.history,context.stack
OUT=trial.clock.ART/'y3_peer_carry_impulse_v1'
ARMS=('unchanged','contemporaneous carry','fast impulse','intermediate impulse','slow impulse',
      'distributed lag','distributed lag for later returns','other stocks histories','past innovations only',
      'persistent signed pressure')


def features(raw,p,z):
    fast,mid,slow=[context.smooth(z,h) for h in (5,24,96)]
    size=np.sqrt(context.smooth(z*z,96))+.004
    def view(a,b,c):
        # The three differences separate newly arrived, intermediate and older revisions.
        persistence=z*np.tanh(z/size)*np.tanh(c/size)
        return np.stack((z,z-a,a-b,b-c,persistence),axis=2)
    return view(fast,mid,slow),view(*(np.roll(a,17,axis=1) for a in (fast,mid,slow)))


def layer():
    return context.layer(sys.modules[__name__])


def replay_with():
    return context.replay_with(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2)
    mod=sys.modules[__name__]
    if sys.argv[1:]==['replay'] or context.train(mod):
        context.carry.peer.replay.replay(mod)
