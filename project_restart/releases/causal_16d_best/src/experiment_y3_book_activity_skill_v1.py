"""Estimate horizon reliability conditional on which peers are repricing and which receivers remain quiet."""
import sys

import numpy as np
import torch
import experiment_joint_factor_order_calibration_v1 as replay
import experiment_y3_horizon_agreement_skill_v1 as peers
import y3_contextual_pricing_training as train
import y3_horizon_agreement_skill_references as history

parent,trial = history.incumbent,history.trial
OUT = trial.clock.ART/'y3_book_activity_skill_v1'
ARMS = ('unchanged','forecast-only agreement','active versus quiet peers','receiver activity','matched liquidity states',
        'whole liquidity readout','liquidity times horizon agreement','sender and receiver disagreement','reassigned activity')
CANDIDATES = (2,3,4,5,6,7)
CONTROLS = (1,8)


def activity(raw):
    r = raw[:,:,0].astype(float)-.00136
    changed = (abs(r)>.022).astype(float)
    fast = peers.study.ema(changed,8)
    slow = peers.study.ema(changed,128)
    return np.tanh(1.5*(fast-slow)/np.sqrt(.02+slow*(1-slow)))


def views(z,gate,side):
    own,base,agreement = peers.views(z,side)
    src = z[:,side::2]
    high,low = [peers.geometry(src,1+sgn*.85*gate[:,side::2]) for sgn in (1,-1)]
    return own,np.column_stack((base,high-low)),agreement,gate[:,1-side::2]


def record(raw,p,y):
    z = trial.clock.center(p.astype(float))
    yy = trial.clock.center(y[:,:,2].astype(float))
    slope = np.clip((z[:,:,2]*yy).sum()/(z[:,:,2]**2).sum(),.4,2.5)
    err = (yy/slope-z[:,:,2]).clip(-4,4)
    gate = activity(raw)
    rows = []
    for side in (0,1):
        own,s,a,l = views(z,gate,side)
        v = own*err[:,1-side::2,None]
        target = np.column_stack([v.mean(1),*((v*m[:,:,None]).mean(1) for m in (a,l,a*l))])
        rows.append((s.T@s/len(s),s.T@target/len(s),(own.transpose(0,2,1)@own).mean(0)/25))
    return tuple(np.mean(v,axis=0) for v in zip(*rows))


def fit(rows):
    gs,bs,rs = [np.stack(v) for v in zip(*rows)]
    out = dict(risk=rs.mean(0))
    for name,n in (('base',13),('liquid',19)):
        g = gs.mean(0)[:n,:n]
        r = bs[:,:n].mean(0)
        r *= abs(r)/(abs(r)+bs[:,:n].std(0)/np.sqrt(len(rows))+1e-8)
        out[name] = np.linalg.solve(g+1.1*np.diag(g.diagonal().clip(.0002)),r)
    return out


def coeff_projection(own,risk):
    inv = np.linalg.inv(risk+.8*np.diag(risk.diagonal().clip(.001)))
    def apply(mu):
        w = mu@inv
        sd = np.sqrt((w*(w@risk)).sum(1)).clip(1e-8)
        w *= (.55*np.minimum(1.,.18*np.sqrt(risk[2,2])/sd))[:,None]
        return (own*w[:,None]).sum(2)
    return apply


def signals(raw,p,st):
    z = trial.clock.center(p.astype(float))
    gate = activity(raw)
    out = np.zeros((*p.shape[:2],len(ARMS)),np.float32)
    for side in (0,1):
        own,s,a,l = views(z,gate,side)
        apply = coeff_projection(own,st['risk'])
        b = st['liquid']
        control = apply(s[:,:13]@st['base'][:,:3])+a*apply(s[:,:13]@st['base'][:,3:6])
        peer = apply(s[:,13:]@b[13:,:3])
        receiver = l*apply(s[:,:13]@b[:13,6:9])
        interaction = l*apply(s[:,13:]@b[13:,6:9])
        full = apply(s@b[:,:3])+l*apply(s@b[:,6:9])
        joint = a*l*apply(s@b[:,9:])
        out[:,1-side::2,1:] = np.stack((control,peer,receiver,peer+receiver+interaction,full,full+joint,
                                        peer-interaction,np.roll(full,7,axis=1)),axis=2)
    return out


def correction():
    return train.correction(sys.modules[__name__])


def replay_with():
    return train.replay_with(sys.modules[__name__])


def layer():
    return history.study.layer(sys.modules[__name__],parent)


if __name__=='__main__':
    torch.set_num_threads(2)
    mod = sys.modules[__name__]
    if sys.argv[1:]==['replay'] or train.train(mod):
        replay.replay(mod)
