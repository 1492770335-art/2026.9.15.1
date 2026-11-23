"""Track executions and quote departures at individually identified past bid and ask prices."""
import numpy as np
import experiment_execution_volume_clock_v1 as clock
from experiment_causal_cross_horizon_ridge_v1 import ART

OUT=ART/'price_level_execution_memory_v1'
OFFSETS=np.array([0,1,2,4,-1])
STOCK=np.arange(50)[:,None]


def side_memory(move,execution,orientation):
    # Prices are relative to this side's opening quote. Positive oriented motion goes outwards.
    price=(orientation*move).astype(np.int64).cumsum(0)
    shape=(50,512)
    tag=np.full(shape,np.iinfo(np.int64).min)
    touched=np.zeros(shape,np.int64)
    state=np.zeros(shape+(4,),dtype=np.float64)
    out=np.empty((len(move),50,len(OFFSETS),4),np.float32)
    names=np.arange(50)
    for t,here in enumerate(price):
        old=price[max(t-1,0)]
        slot=old%512
        span=t-touched[names,slot]
        v=state[names,slot]*(np.exp(-span/200)*(tag[names,slot]==old))[:,None]
        signed=here-old
        # Executed amounts when the price stayed and departed are stored in separate channels.
        v[:,0]+=np.where(signed==0,execution[t],0)
        v[:,1]+=np.where(signed!=0,execution[t],0)
        v[:,2]+=1
        v[:,3]+=np.sign(signed)
        tag[names,slot]=old;touched[names,slot]=t;state[names,slot]=v
        levels=here[:,None]+OFFSETS;slots=levels%512
        decay=np.exp(-(t-touched[STOCK,slots])/200)*(tag[STOCK,slots]==levels)
        result=state[STOCK,slots]*decay[:,:,None]
        out[t,:,:,0]=np.log1p(result[:,:,0])
        out[t,:,:,1]=np.log1p(result[:,:,1])
        out[t,:,:,2]=np.log1p(result[:,:,2])
        out[t,:,:,3]=np.arcsinh(result[:,:,3])
    return out


class Day:
    def __init__(self,raw,g):
        buy,sell,_,bid,ask,spread,q,_=clock.decode(raw,g)
        upper=side_memory(ask,buy/g['rate'],1)
        lower=side_memory(bid,sell/g['rate'],-1)
        asym=upper-lower
        rate=upper[:,:,:,0]-upper[:,:,:,2]-lower[:,:,:,0]+lower[:,:,:,2]
        imbalance=(buy-sell)/(buy+sell+g['rate'])
        near=np.tanh((ask+bid)/3)
        pressure=np.tanh(.65*(raw[:,:,41].astype(float)-raw[:,:,42].astype(float)))
        n=len(raw);flat=asym.reshape(n,50,-1);symmetric=(upper+lower).reshape(n,50,-1)*.5
        self.x=np.concatenate((flat,rate,flat*np.log1p(spread)[:,:,None],flat*np.log(q)[:,:,None],
                               symmetric*imbalance[:,:,None],symmetric*near[:,:,None],
                               symmetric*pressure[:,:,None],rate*pressure[:,:,None]),axis=2).astype('float32')

    def design(self,t):
        v=self.x[t];m=v.mean(1)
        return np.column_stack((m,m[:,:20]*np.exp(-t[:,None]/340))).astype('float32'),v-m[:,None]


if __name__=='__main__':
    import sys
    if '--replay' in sys.argv:
        import replay_verified_forecast_parent_v1 as replay
        replay.main(clock.OUT,OUT,clock.frozen(OUT,Day))
    else:
        clock.evaluate(OUT,clock.OUT,Day,clock.geometry(),{})
