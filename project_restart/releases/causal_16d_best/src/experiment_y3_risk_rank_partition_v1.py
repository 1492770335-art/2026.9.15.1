"""Combine forecast calibration by future variance with a separate neural model for local stock ordering."""
import sys
import numpy as np
import torch

import experiment_y3_heteroskedastic_pair_network_v1 as pair
import experiment_y3_semivariance_pretraining_v1 as energy
import experiment_y3_score_neighborhood_heads_v1 as local
import experiment_y3_rank_spacing_calibration_v1 as screen
import y3_pair_incumbent_validation as trial

OUT=trial.clock.ART/'y3_risk_rank_partition_v1'
ARMS=['energy only','rank only','additive half','separate ranks and scale','protect small rank adjustment',
      'equal with risk-specialized ranks','misassigned ordering','half separated','conservative additive']


def state(raw):
    z=energy.state(raw)
    return z,pair.state(raw)


def record_with_state(z,p,y,d):
    return energy.record_with_state(z[0],p,y,d),pair.record_with_state(z[1],p,y,d)


def merge_records(records):
    a,b=zip(*records)
    return [np.mean(values,0) for values in zip(*a)],pair.merge_records(b)


def fit(hist):
    a,b=zip(*hist)
    values={}
    for name,part in [('energy',energy.fit(a)),('pair',pair.fit(b))]:
        values.update({name+'_'+k:v for k,v in part.items()})
    return values


def changes_with_reference(z,m,p):
    e=energy.changes_with_reference(z[0],{k[7:]:v for k,v in m.items() if k.startswith('energy_')},p)[2]
    bank=pair.changes_with_reference(z[1],{k[5:]:v for k,v in m.items() if k.startswith('pair_')},p)
    rank=bank[4]-bank[5]
    # Projecting into local score neighborhoods isolates ordering from the size calibration already supplied by energy.
    def coarse(order):
        small=local.residual(e[:,:,None],p[:,:,2],order,5)[:,:,0]
        return e-small
    broad=coarse(p[:,:,2])
    wrong=coarse(np.roll(p[:,:,2],17,axis=1))
    return [e,rank,.5*(e+rank),.5*(broad+rank),.6*broad+.25*rank,.5*(broad+bank[4]),.5*(wrong+rank),
            .25*(broad+rank),.65*e+.20*rank]


def select(f,d):
    good=[]
    for i in (2,3,4,5,7,8):
        v=f[:,:,0,i]
        if ((v>0).all() and (d[:,0,i]>0).sum()>=4 and
            all((v-f[:,:,0,k]).max(0).min()>0 for k in (0,1)) and
            (i in (2,8) or (v-f[:,:,0,6]).mean(0).min()>0)):
            good.append(i)
    return int(max(good,key=lambda i:f[:,0,0,i].mean())) if good else None


def correction():
    return trial.previous.previous.prior.correction(sys.modules[__name__])


def layer():
    return trial.layer(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2)
    screen.run(sys.modules[__name__])
