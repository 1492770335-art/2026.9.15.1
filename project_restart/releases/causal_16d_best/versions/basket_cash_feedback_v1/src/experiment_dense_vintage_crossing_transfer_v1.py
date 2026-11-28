"""Transfer the near-horizon payoff of matched revision crossings to the 60-tick head in physical return units."""
import sys
import numpy as np
import prepare_dense_vintage_crossing_reference_v1 as best

cross=best.model
strip,rel,onset,REFERENCE,read_p=cross.strip,cross.rel,cross.onset,best.OUT,best.read_p
OUT=REFERENCE.parent/'dense_vintage_crossing_transfer_v1'
ARMS=('near_payoff','shared_near_and_long_direction','near_without_conflicting_long')
train=cross.train


def signals(raw,grid,p,bank):
    vals=cross.signals(raw,grid,p,bank)
    _,coef,_=strip.physical()
    near=vals[:,:,2,0]*coef[2]/coef[1]
    long=vals[:,:,2,1]
    same=near*long>0
    out=np.zeros((*p.shape[:2],3,2))
    out[:,:,:,1]=np.stack((near,np.where(same,np.sign(near)*np.minimum(abs(near),abs(long)),0.),
                           np.where(same|(abs(long)<.5*abs(near)),near,0.)),axis=-1)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    strip.evaluation.rolling(sys.modules[__name__])
