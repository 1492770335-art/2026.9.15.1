"""Filter persistent buy/sell flow in event time and estimate how much of it survives each forecast horizon."""
import numpy as np
import experiment_execution_location_markouts_v1 as location
import experiment_execution_volume_clock_v1 as clock
from experiment_causal_cross_horizon_ridge_v1 import ART

OUT=ART/'persistent_execution_state_v1'
DURATION=np.array([2.5,12.,65.,320.])
BIAS=np.array([-.78,-.32,0.,.32,.78])
MASS=np.array([.12,.21,.34,.21,.12])
PRIOR=np.array([.36,.30,.22,.12])[:,None]*MASS[None,:]


def filter_steps(sign,volume):
    n=len(sign);states=np.broadcast_to(PRIOR,(50,4,5)).copy()
    propensity=np.empty((n,50,4),np.float32);horizon=np.empty((n,50,3),np.float32)
    prob=np.empty((n,50,3),np.float32);quality=np.zeros(4);count=0
    rate=np.full((50,1),.4);ew=np.zeros(50)
    persistence=1-1/DURATION[None,:,None]
    for t in range(n):
        here=volume[t]>0
        prior=states*persistence+(states*(1-persistence)).sum((1,2))[:,None,None]*PRIOR
        # At an observed execution, the label for this diagnostic is the soft fraction on the positive side.
        next_sign=(prior*BIAS).sum((1,2));s=sign[t]
        if here.any():
            frac=(1+s[here])*.5
            for j,pred in enumerate((np.zeros(50),ew,next_sign)):
                p=(1+pred[here].clip(-.997,.997))*.5
                quality[j]+=(-frac*np.log(p)-(1-frac)*np.log1p(-p)).sum()
            count+=here.sum()
        ew=np.where(here,.85*ew+.15*s,ew)
        # Fractional Bernoulli likelihood allows both directions within a single reported frame.
        weight=np.minimum(2,.56+.44*np.log1p(volume[t]))
        likelihood=(.5*(1+BIAS))**(((1+s)*.5*weight)[:,None])*(.5*(1-BIAS))**(((1-s)*.5*weight)[:,None])
        post=prior*likelihood[:,None,:];post/=post.sum((1,2),keepdims=True)
        states=np.where(here[:,None,None],post,states)
        rate+=.022*(here[:,None]-rate)
        component=(states*BIAS).sum(2);propensity[t]=component
        # After a regime replacement the prior is sign symmetric, so its expected signed contribution is zero.
        for j,h in enumerate((5,20,60)):
            decay=1-rate/DURATION
            horizon[t,:,j]=(component*(decay*(1-decay**h)/np.maximum(1e-6,1-decay))/h).sum(1)
        prob[t,:,0]=(states*abs(BIAS)).sum((1,2));prob[t,:,1]=states[:,2:].sum((1,2))
        prob[t,:,2]=s-next_sign
    return propensity,horizon,prob,quality[:3]/max(1,count)


class Day:
    def __init__(self,raw,g):
        r,q,old,curr,prev,s,rest,units,excess,volume=location.values(raw,g)
        part,exp,conf,self.loss=filter_steps(s,volume)
        t=np.arange(len(raw))+1
        gap=exp[:,:,2]-exp[:,:,0]
        risk=np.sqrt((7+clock.flow.trailing(r*r,72))/(7+np.minimum(t,72))[:,None])
        move=np.tanh(r);reach=np.tanh(excess)
        common=exp.mean(1,keepdims=True);mood=np.broadcast_to(common,exp.shape)
        pred=exp[:,:,2]
        pieces=[part,exp,mood,np.stack((s,conf[:,:,2],gap,pred*(volume==0)),axis=2)]
        for mod in (np.log(q),np.log1p(old),np.tanh(units),np.tanh(risk-1),conf[:,:,0],conf[:,:,1],
                    np.tanh(abs(excess)),s*move,s*np.tanh(curr),s*reach,np.sign(pred)*s):
            pieces.extend((part*mod[:,:,None],exp*mod[:,:,None]))
        flow=conf[:,:,2]*units
        for h in (8,38,160):
            avg=clock.flow.trailing(flow,h)/np.sqrt(np.minimum(t,h)[:,None])
            pieces.append(np.stack((np.arcsinh(avg),np.tanh(avg)*pred,avg*conf[:,:,1],avg*np.tanh(risk)),axis=2))
        self.x=np.concatenate(pieces,axis=2).astype('float32')

    def design(self,t):
        v=self.x[t];m=v.mean(1)
        return np.column_stack((m,m[:,:16]*np.exp(-t[:,None]/330))).astype('float32'),v-m[:,None]


if __name__=='__main__':
    import sys
    if '--replay' in sys.argv:
        import replay_verified_forecast_parent_v1 as replay
        replay.main(location.OUT,OUT,clock.frozen(OUT,Day))
    else:
        g=location.geometry();audit={}
        for d in range(4,8):
            *_,s,rest,units,excess,volume=location.values(clock.pair.rawday(d),g)
            audit[d+1]=dict(zip(('constant','event_ewma','state_prior'),filter_steps(s,volume)[3].tolist()))
        clock.evaluate(OUT,location.OUT,Day,g,{'next_execution_cross_entropy':audit})
