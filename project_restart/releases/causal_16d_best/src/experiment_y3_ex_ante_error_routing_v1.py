"""Use forecast-time estimates of directional and size risk to calibrate a frozen Y3 prediction on separate dates."""
import sys
import numpy as np
from scipy.special import expit
import torch

import experiment_y3_order_response_filter_v1 as filters
import experiment_y3_peer_arrival_alignment_v1 as book
import experiment_y3_rank_spacing_calibration_v1 as screen
import y3_pair_incumbent_validation as trial

OUT = trial.clock.ART/'y3_ex_ante_error_routing_v1'
ARMS = ['score size only', 'direction risk', 'large-error risk', 'joint risk', 'stock-reassigned risk',
        'joint beyond score size', 'direction beyond score size', 'large error beyond score size']


def state(raw):
    x = book.signals(raw)
    vol = np.sqrt(.02+filters.smooth(raw[:,:,0].astype('float32')**2,60))
    active = filters.smooth((abs(raw[:,:,0])>.04).astype('float32'),60)
    flag = raw[:,:,83:].astype('float32')-1
    changes = (flag!=np.concatenate((flag[:1],flag[:-1]))).mean(2).astype('float32')
    return np.concatenate((x,filters.smooth(x,30),np.log(vol)[:,:,None],active[:,:,None],
                           filters.smooth(changes,45)[:,:,None]),2)


def design(z,p):
    q = trial.clock.center(p)
    scale = np.sqrt((q*q).mean(1,keepdims=True)).clip(.04)
    conf = np.tanh(abs(q[:,:,2:3])/scale[:,:,2:3])
    sign = np.sign(q[:,:,2:3])
    delta = q-filters.smooth(q,35)
    age = np.broadcast_to(np.exp(-np.arange(len(p),dtype='float32')[:,None,None]/420),sign.shape)
    features = np.concatenate((np.ones_like(sign),conf,conf*conf,abs(q[:,:,2:3]),q*sign/scale,
                delta*sign/scale,np.tanh(z[:,:,:16])*sign,z[:,:,16:],abs(z[:,:,:8]-z[:,:,8:16]),age),2)
    return features.astype('float32'),q[:,:,2],conf[:,:,0]


def record_with_state(z,p,y,d):
    t = np.arange(61+d%9,len(p)-60,9)
    x,score,_ = design(z,p)
    truth = trial.clock.center(y[t,:,2])
    score = score[t]
    slope = (score*truth).sum(dtype=float)/max(1e-6,(score*score).sum(dtype=float))
    v = truth/np.clip(slope,.4,2.)-score
    targets = np.stack((score*truth<=0,abs(v)>.8),2).astype('float32')
    return [a.astype('float32') for a in (x[t],score,v,targets)]


def merge_records(records):
    return [np.concatenate(parts) for parts in zip(*records)]


def logistic(x,y):
    x = x.reshape(-1,x.shape[-1]).astype(float)
    y = y.reshape(-1,y.shape[-1]).astype(float)
    penalty = .075*(x*x).mean(0).clip(.0003)
    penalty[0] = .00001
    w = np.zeros((x.shape[-1],y.shape[-1]))
    for _ in range(5):
        p = expit(x@w)
        for j in range(w.shape[1]):
            grad = x.T@(p[:,j]-y[:,j])/len(x)+penalty*w[:,j]
            h = x.T@(x*(p[:,j]*(1-p[:,j]))[:,None])/len(x)
            w[:,j] -= np.linalg.solve(h+np.diag(penalty),grad)
    return w.astype('float32')


def risks(x,w):
    return expit(x@w)


def signals(pred,base,q):
    diff = pred-base
    reassigned = np.roll(diff,17,axis=1)
    size = np.sign(q)*np.minimum(np.abs(q),.65)**.5
    # Isolate conditional slope risk at fixed score magnitude via a risk model with the same magnitude covariates.
    v = np.stack((q,q*base[:,:,0],q*base[:,:,1],size*diff[:,:,0],size*diff[:,:,1],
                  q*diff[:,:,0],q*diff[:,:,1],size*reassigned[:,:,0],size*reassigned[:,:,1],
                  q*reassigned[:,:,0],q*reassigned[:,:,1]),2)
    return trial.clock.center(v)


def fit(hist):
    # Estimate a risk model and the mapping of its risk to signed error on disjoint dates.
    ws,betas=[],[]
    for i in (0,1):
        x,q,v,target = [np.concatenate(a) for a in zip(*hist[i::2])]
        w=logistic(x,target)
        wb=logistic(x[:,:,:4],target)
        ws.append((w,wb))
        rows=[]
        for x,q,v,target in hist[1-i::2]:
            a=signals(risks(x,w),risks(x[:,:,:4],wb),q).reshape(q.size,-1)
            rows.append((a.T@a/len(a),a.T@v.ravel()/len(a)))
        g,b=[np.stack(a) for a in zip(*rows)]
        def solve(indices):
            idx=np.array(indices)
            cov=b[:,idx].mean(0)
            cov=cov*abs(cov)/(abs(cov)+b[:,idx].std(0)/np.sqrt(len(b))+1e-6)
            coef=np.zeros(g.shape[1])
            coef[idx]=trial.curve.solve(g.mean(0)[np.ix_(idx,idx)],cov,.8).clip(-2,2)
            return coef
        size=solve([0,1,2])
        direction=solve([0,1,2,3,5])
        magnitude=solve([0,1,2,4,6])
        joint=solve(list(range(7)))
        wrong=solve([0,1,2,7,8,9,10])
        betas.append(np.column_stack((size,direction,magnitude,joint,wrong,joint-size,direction-size,magnitude-size)))
    return dict(w=np.stack([w for w,_ in ws]),wb=np.stack([w for _,w in ws]),
                coef=(.4*np.stack(betas)).astype('float32'))


def changes_with_reference(z,m,p):
    x,q,_=design(z,p)
    change=0
    for w,wb,v in zip(m['w'],m['wb'],m['coef']):
        change+=.5*(signals(risks(x,w),risks(x[:,:,:4],wb),q)@v)
    return list(np.moveaxis(change,2,0))


def select(f,d):
    good=[]
    for i in (1,2,3,5,6,7):
        v=f[:,:,0,i]
        if ((v>0).all() and (d[:,0,i]>0).sum()>=4 and
            all((v-f[:,:,0,j]).mean(0).min()>0 for j in (0,4))):
            good.append(i)
    return int(max(good,key=lambda i:f[:,0,0,i].mean())) if good else None


def correction():
    return trial.previous.previous.prior.correction(sys.modules[__name__])


def layer():
    return trial.layer(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2)
    screen.run(sys.modules[__name__])
