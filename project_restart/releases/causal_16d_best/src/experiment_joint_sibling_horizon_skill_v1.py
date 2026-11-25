"""Apply the increment calibration to the short targets and the conditional skill estimate to Y3."""
import json
import sys
import numpy as np
import torch
import experiment_joint_increment_geometry_v1 as increments
import experiment_y3_conditional_forecast_skill_v1 as skill
import y3_horizon_transport_references as history

trial,stack,parent = history.trial,history.stack,history.incumbent
OUT = trial.clock.ART/'joint_sibling_horizon_skill_v1'


def assemble():
    OUT.mkdir(exist_ok=True)
    short = json.loads((increments.OUT/'results.json').read_text())['selected'][:2]
    long = json.loads((skill.OUT/'results.json').read_text())['selected']
    w = np.zeros((2,16))
    if any(short):
        with np.load(increments.OUT/'model.npz') as f:
            w = f['w'][:2].copy()
    st = {}
    if long:
        with np.load(skill.OUT/'model.npz') as f:
            st = {k:f[k] for k in f.files if k!='pick'}
    np.savez(OUT/'model.npz',w=w,pick=long,**st)
    (OUT/'results.json').write_text(json.dumps(dict(selected=[*short,long],training_days=[1,16],
                                                    references=[increments.OUT.name,skill.OUT.name])))
    return any(short) or long!=0


def update(raw,p,w,st,pick):
    q = p.copy()
    if w.any():
        q[:,:,:2] += (p[:,:,:2]*w[:,0]+increments.design(p,raw)@w[:,1:].T).astype('float32')
    if pick:
        q[:,:,2] += skill.signals(p,st)[:,:,pick]
    return q


def adjustment():
    with np.load(OUT/'model.npz') as f:
        st,w,pick = dict(f),f['w'],int(f['pick'])
    return lambda raw,p:update(raw,p,w,st,pick)


def layer():
    prev,apply = parent.layer(),adjustment()
    return lambda raw,p:apply(raw,prev(raw,p))


def replay_with():
    apply = adjustment()
    def read(d,raw):
        p,h,n = history.read(d,raw)
        return apply(raw,p),apply(raw[:n],h),n
    return stack.verifier,read


if __name__=='__main__':
    torch.set_num_threads(2)
    if not sys.argv[1:]:
        increments.train()
        skill.train()
    if sys.argv[1:]==['replay'] or assemble():
        increments.replay.replay(sys.modules[__name__])
