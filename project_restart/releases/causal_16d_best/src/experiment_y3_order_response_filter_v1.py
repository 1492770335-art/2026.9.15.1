"""Factor the spacing and elapsed age of ordered book shocks with causal exponential filters."""
import sys

import numpy as np
import torch
from scipy.signal import lfilter

import y3_pressure_validation as trial

OUT=trial.clock.ART/'y3_order_response_filter_v1'
ARMS=['matched time scales','different time scales late','different time scales near','joint horizons','complete filter']


def smooth(x,span):
    rho=np.float32(np.exp(-1./span))
    # The first sample initializes the filter separately for each stock and date.
    result,_=lfilter(np.array([1-rho],dtype='float32'),np.array([1,-rho],dtype='float32'),x,axis=0,
                     zi=(rho*x[:1]).astype('float32'))
    return result


def state(raw,pre):
    return filters(np.diff(pre,axis=0).astype('float32'))


def filters(z):
    z=z-smooth(z,120)
    p,a=z[:,:,:8],z[:,:,8:]
    n=36+32+3*32+3*32
    out=np.empty((*z.shape[:2],n),dtype='float16')
    col=0
    def write(x):
        nonlocal col
        block=x.reshape(*z.shape[:2],-1)
        out[:,:,col:col+block.shape[2]]=trial.clock.center(block)
        col+=block.shape[2]
    for span in (4,16,64):write(smooth(z,span))
    coincident=(p[:,:,:,None]*a[:,:,None,:]).reshape(*z.shape[:2],32)
    write(smooth(coincident,24))
    spans=(4,16,64)
    matches=[];contrasts=[]
    for k,separation in enumerate(spans):
        past=smooth(z,separation)
        past=np.concatenate((np.zeros_like(past[:1]),past[:-1]))
        impulse=past[:,:,:8,None]*a[:,:,None,:]-p[:,:,:,None]*past[:,:,None,8:]
        impulse=impulse.reshape(*z.shape[:2],32)
        matched=smooth(impulse,separation)
        # Compare the same ordering at two different elapsed-age scales.
        other=smooth(impulse,spans[(k+1)%3])
        matches.append(matched.astype('float16'))
        contrasts.append((other-matched).astype('float16'))
    for v in matches+contrasts:write(v.astype('float32'))
    assert col==n
    return out


def observe(z,y,d):
    t=np.arange(d%5,len(y)-60,5);c=trial.curve.settled.historical_geometry()[1]
    g=b=None
    for ids in np.array_split(t,max(1,(len(t)+95)//96)):
        a=z[ids].reshape(-1,z.shape[2]).astype('float32')
        r=np.stack((trial.scoring.late(y[ids]),trial.clock.center(y[ids,:,1])*(c[2,1]/c[1,1])),2).clip(-3,3).reshape(-1,2)
        gg=a.T@a;bb=a.T@r
        if g is None:g=gg.astype(float);b=bb.astype(float)
        else:g+=gg;b+=bb
    return g/(len(t)*50),b/(len(t)*50)


def fit(hist):
    g,b=[np.stack(v) for v in zip(*hist)];avg=g.mean(0)
    weights=[]
    for start,stop in ((68,164),(164,260)):
        gram=avg[:stop,:stop]
        basis=np.vstack((-trial.curve.solve(gram[:start,:start],gram[:start,start:],.20),np.eye(stop-start)))
        day=np.moveaxis(b[:,:stop],2,1)@basis
        mean=day.mean(0).T;se=day.std(0).T/np.sqrt(len(day))
        w=basis@trial.curve.solve(basis.T@gram@basis,mean*abs(mean)/(abs(mean)+se+.000005),.85)
        result=np.zeros((len(avg),2));result[:stop]=w;weights.append(result)
    return dict(w=np.column_stack(weights).astype('float32')),dict(days=len(hist))


def changes(z,model):
    out=np.concatenate([row.astype('float32')@model['w'] for row in np.array_split(z,max(1,(len(z)+127)//128))])
    a,b,c,d=np.moveaxis(out,2,0)
    return [.13*(a+b),.15*c,.13*d,.12*(c+d),.10*(a+b+c+d)]


def correction():return trial.correction(sys.modules[__name__])
def layer():return trial.layer(sys.modules[__name__])
def load():return trial.load(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2);trial.run(sys.modules[__name__])
