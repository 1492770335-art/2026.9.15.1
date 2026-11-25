"""Compare forecast revisions immediately before, at, and after the stock's recent price changes."""
import sys
import numpy as np
import torch
import y3_price_revision_references as parent

screen,trial=parent.screen,parent.trial
OUT=trial.clock.ART/'y3_event_forecast_response_shape_v1'
ARMS=('on change','lead','follow-through','response asymmetry','quiet lead and follow-through',
      'five-tick displacement','complete response','isolated change','half complete response')
CANDIDATES=(1,2,3,4,6,7,8)
CONTROLS=(0,5)


def lag(v,h):
    out=np.zeros_like(v)
    if len(v)>h:out[h:]=v[:-h]
    return out


def features(raw,p):
    c=trial.curve.settled.historical_geometry()[1][:,1]
    near=trial.clock.center(p[:,:,1])*c[2]/c[1]
    tail=trial.clock.center(p[:,:,2])-near
    levels=np.stack((near,tail),2)
    delta=np.diff(levels,axis=0,prepend=levels[:1])
    event=raw[:,:,0]!=np.float16(.00136)
    def attach(v,mask):return v*mask[:,:,None]
    now=attach(delta,event)
    lead=attach(lag(delta,1),event)
    follow=attach(delta,lag(event,1))
    before=attach(lag(delta,1),event&~lag(event,1))
    after=attach(delta,lag(event,1)&~event)
    fake=attach(lag(delta,5),event)
    isolated=attach(lag(delta,1),lag(event,1)&~lag(event,2)&~event)
    # The follow-up and isolation channels enter the state only when their next tick has already arrived.
    banks=np.concatenate((now,lead,follow,before,after,fake,isolated),axis=2)
    cs=np.concatenate((np.zeros_like(banks[:1],dtype=float),banks.cumsum(0,dtype=float)))
    t=np.arange(len(raw))[:,None]+1;stock=np.arange(50)[None,:]
    parts=[]
    for n in (3,12):
        start=parent.incumbent.boundaries(event,n)
        current=(cs[t,stock]-cs[start,stock]).reshape(len(raw),50,7,2)
        # Follow-up channels are one tick later than the event they refer to.
        later=np.minimum(t,start+1)
        current[:,:,[2,4,6],:]-=(cs[later,stock]-cs[start,stock]).reshape(len(raw),50,7,2)[:,:,[2,4,6],:]
        parts.append(current)
    z=np.stack(parts,3).reshape(len(raw),50,7,4)
    return trial.clock.center(np.concatenate((z.reshape(len(raw),50,28),z[:,:,2]-z[:,:,1]),2)).astype('float32')


def fit(rows):
    on,pre,post,quiet1,quiet2,wrong,isolated,diff=[np.arange(3+i*4,7+i*4) for i in range(8)]
    joint=np.r_[pre,post,on]
    return screen.projected_readouts(rows,[(on,3,.4),(pre,7,.4),(post,7,.4),(diff,7,.4),
          (np.r_[quiet1,quiet2],7,.4),(wrong,7,.4),(joint,3,.4),(isolated,7,.4),(joint,3,.2)])


def correction():return screen.correction(sys.modules[__name__])
def layer():return screen.layer(sys.modules[__name__],parent.incumbent)

if __name__=='__main__':
    torch.set_num_threads(2)
    screen.run(sys.modules[__name__],parent.references,parent.outer)
