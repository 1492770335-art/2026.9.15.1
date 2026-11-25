"""Map peer forecast covariances to recent geometry before adjusting each receiver's three forecasts."""
import sys
import numpy as np
import torch
import y3_horizon_geometry_training as study
import y3_dynamic_horizon_geometry_references as history

parent,trial = history.incumbent,history.trial
OUT = trial.clock.ART/'y3_horizon_covariance_transport_v1'
ARMS = ('unchanged','coordinate volatility only','symmetric covariance transport','correlation rotation',
        'short to long innovations','long to short innovations','long forecast component','conditional geometry change',
        'reassigned stock geometry','within and cross horizon transport','difference of triangular orders')
GROUPS = ((),(0,1,2),(3,4,5),(6,7,8),(9,10,11),(12,13,14),(5,),(7,8,11),(15,16,17),
          (0,1,2,6,7,8),(18,19,20))
CANDIDATES = (2,3,4,5,6,7,9,10)
CONTROLS = (1,8)


def root(a,power):
    v,u = np.linalg.eigh(a)
    return (u*v.clip(1e-9)[:,None,:]**power)@u.transpose(0,2,1)


def transport(a,b):
    r = root(a,.5)
    inv = root(a,-.5)
    return inv@root(r@b@r,.5)@inv


def design(p,raw):
    z = trial.clock.center(p.astype(float))
    out = np.empty((*p.shape[:2],21),np.float32)
    eye = np.eye(3)
    for side in (0,1):
        x,v = z[:,side::2],z[:,1-side::2]
        gm = x.transpose(0,2,1)@x/25
        slow = study.ema(gm,96)+.003*eye
        fast = .75*(gm+.003*eye)+.25*slow
        a = np.sqrt(fast.diagonal(axis1=1,axis2=2))
        b = np.sqrt(slow.diagonal(axis1=1,axis2=2))
        diagonal = v*((b/a)-1)[:,None]
        whole = v@(transport(fast,slow)-eye)
        corr = transport(fast/a[:,None,:]/a[:,:,None],slow/b[:,None,:]/b[:,:,None])
        rotation = ((v/a[:,None])@(corr-eye))*b[:,None]
        def triangle(order):
            f,s = [np.linalg.cholesky(m[:,order][:,:,order]) for m in (fast,slow)]
            t = np.linalg.solve(f.transpose(0,2,1),s.transpose(0,2,1))-eye
            return (v[:,:,order]@t)[:,:,np.argsort(order)]
        forward = triangle([0,1,2])
        reverse = triangle([2,1,0])
        out[:,1-side::2] = np.concatenate((diagonal,whole,rotation,forward,reverse,
                                          np.roll(whole,9,axis=1),forward-reverse),axis=2)
    return out


def correction():
    return study.correction(sys.modules[__name__])


def replay_with():
    return study.replay_with(sys.modules[__name__])


def layer():
    return history.study.layer(sys.modules[__name__],parent)


if __name__=='__main__':
    torch.set_num_threads(2)
    mod = sys.modules[__name__]
    if sys.argv[1:]==['replay'] or study.train(mod):
        parent.replay.replay(mod)
