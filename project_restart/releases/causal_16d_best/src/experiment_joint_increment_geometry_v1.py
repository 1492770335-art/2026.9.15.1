"""Transport the implied short, middle and long return increments, fitting the three official targets separately."""
import json
import sys
import numpy as np
import torch
import experiment_joint_factor_order_calibration_v1 as replay
import experiment_y3_settled_factor_confidence_v1 as observed
import experiment_y3_horizon_covariance_transport_v1 as geo
import y3_horizon_geometry_training as study
import y3_horizon_transport_references as history

parent,trial,stack = history.incumbent,history.trial,history.stack
OUT = trial.clock.ART/'joint_increment_geometry_v1'
ARMS = ('unchanged','cumulative horizons','increment dispersion','increment rotation','separate scale and rotation',
        'two delayed increments','late increment only','increment and cumulative contrast','permuted maturity increments')
GROUPS = ((),(0,1,2),(3,4,5),(6,7,8),tuple(range(3,9)),(4,5,7,8),(5,8),(9,10,11),(12,13,14))
CONTROLS = (1,8)
CANDIDATES = tuple(range(2,8))


def transport(z,side):
    peer,own = z[:,side::2],z[:,1-side::2]
    g = peer.transpose(0,2,1)@peer/peer.shape[1]
    slow = study.ema(g,96)
    fast = .75*g+.25*slow
    a,b = [m+.002*np.eye(3) for m in (fast,slow)]
    s,t = [np.sqrt(x.diagonal(axis1=1,axis2=2)) for x in (a,b)]
    delta = own*(t/s-1)[:,None]
    T = geo.transport(a/s[:,None,:]/s[:,:,None],b/t[:,None,:]/t[:,:,None])
    rotate = ((own/s[:,None])@(T-np.eye(3)))*t[:,None]
    return delta,rotate


def design(p,raw):
    z = trial.clock.center(p.astype(float))
    coeff = observed.historical_geometry()[1][:,1]
    scale = coeff[2]/coeff
    def increments(order):
        cash = z[:,:,order]*scale[order]
        cash[:,:,1:] -= cash[:,:,:-1].copy()
        return cash/scale[order]
    legs = increments([0,1,2])
    bad = increments([1,2,0])
    v = np.empty((*p.shape[:2],15),np.float32)
    for side in (0,1):
        plain = sum(transport(z,side))
        ds,dr = transport(legs,side)
        wrong = sum(transport(bad,side))
        # Convert the transported increments back to cumulative returns for a same-coordinate contrast.
        cum = ((ds+dr)*scale).cumsum(2)/scale
        v[:,1-side::2] = np.concatenate((plain,ds,dr,cum-plain,wrong),axis=2)
    return v


def record(d):
    raw = trial.clock.book.rawday(d)
    y = trial.clock.labels(d).reshape(-1,50,3)
    out = []
    for p in history.references(d,raw):
        z = design(p,raw)
        v = np.concatenate((np.ones_like(p[:,:,:1]),p,z,y),axis=2)
        v = v.reshape(-1,v.shape[-1]).astype(float)
        out.append(v.T@v)
    return out


def target_rows(rows,k):
    idx = np.r_[0,1+k,np.arange(4,19),19+k]
    return [[m[np.ix_(idx,idx)] for m in day] for day in rows]


def train():
    OUT.mkdir(exist_ok=True)
    mod = sys.modules[__name__]
    rows = [record(d) for d in range(8,16)]
    picks,bank,reports = [],[],[]
    for k in range(3):
        ds = target_rows(rows,k)
        f,daily = [],[]
        for n in (2,4,6):
            w = study.weights(mod,ds[:n])
            gain = study.history.incumbent.gain
            f.append([gain(m,w) for m in np.sum(ds[n:n+2],axis=0)])
            daily.extend([[gain(m,w) for m in day] for day in ds[n:n+2]])
        f,dd = np.asarray(f),np.asarray(daily)
        avg = f.mean(0)
        keep = [i for i in CANDIDATES if (avg[:,i]>0).all() and (f[-1,:,i]>0).all() and
                (f[:,:,i]>0).sum(0).min()>=2 and (dd[:,:,i]>0).sum(0).min()>=4 and
                all((avg[:,i]>avg[:,j]).all() for j in CONTROLS)]
        pick = max(keep,key=lambda j:avg[:,j].min()) if keep else 0
        picks.append(pick)
        bank.append(study.weights(mod,ds)[pick])
        reports.append(dict(mean=avg.tolist(),folds=f.tolist(),daily=dd.tolist()))
        print(k+1,ARMS[pick],avg[:,pick],flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(selected=picks,targets=reports,arms=ARMS)))
    if any(picks):
        np.savez(OUT/'model.npz',w=bank)
    return any(picks)


def adjustment():
    with np.load(OUT/'model.npz') as f:
        w = f['w']
    return lambda raw,p:p+(p*w[:,0]+design(p,raw)@w[:,1:].T).astype('float32')


def replay_with():
    apply = adjustment()
    def read(d,raw):
        p,h,n = history.read(d,raw)
        return apply(raw,p),apply(raw[:n],h),n
    return stack.verifier,read


def layer():
    old,apply = parent.layer(),adjustment()
    return lambda raw,p:apply(raw,old(raw,p))


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['replay'] or train():
        replay.replay(sys.modules[__name__])
