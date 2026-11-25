"""Jointly refit forecast revisions in the orthogonal stock sectors they actually affect."""
import json
import numpy as np
import torch

import joint_intraday_risk_references as history
import experiment_y3_risk_subspace_calibration_v1 as risk

parent,trial,stack=history.incumbent,history.trial,history.stack
live=history.previous
joint_factor=live.previous
factor=joint_factor.previous
touch= factor.previous
early=touch.previous.previous
ANCHORS=(early,touch,joint_factor,history)
OUT=trial.clock.ART/'joint_factor_stage_stack_v1'
ARMS=('unchanged','same-horizon shared slopes','all horizons shared slopes','group-specific own horizon',
      'group-specific cross-horizon','date-shrunk group-specific','individual directions only',
      'correlation groups with permuted stock identities','separate common and relative slopes')
BASE=3
WIDTH=9
GROUPS=tuple((a,b) for _,a,b in risk.GROUPS)


def regressors(anchors,u):
    p=anchors[-1]
    z=np.concatenate([b-a for a,b in zip(anchors,anchors[1:])],2)
    modes=np.einsum('tsc,sk->tkc',z,u,optimize=True)
    pieces=[np.einsum('tkc,sk->tsc',modes[:,a:b],u[:,a:b],optimize=True) for a,b in GROUPS]
    return np.concatenate((p,*pieces),2)


def record(d):
    raw=trial.clock.book.rawday(d)
    forecasts=[h.references(d,raw) for h in ANCHORS]
    y=trial.clock.labels(d).reshape(-1,50,3)
    bases=risk.geometry()
    result=[]
    for u in bases:
        rows=[]
        for anchors in zip(*forecasts):
            v=np.concatenate((np.ones_like(y[:,:,:1]),regressors(anchors,u),y),2).reshape(-1,43).astype(float)
            rows.append(v.T@v)
        result.append(rows)
    return np.array(result)


def fit(records,k):
    # Each date and reference model contributes a covariance; the holdout never enters the readout fit.
    days=np.asarray(records).mean(2)
    def solve_basis(rows,A,shrink):
        c=stack.covariance(rows.sum(0))
        g=c[BASE:-3,BASE:-3]-np.outer(c[BASE:-3,k],c[k,BASE:-3])/c[k,k]
        def grad(m):
            q=stack.covariance(m)
            return (q[BASE:-3,-3+k]*q[k,k]/q[k,-3+k]-q[BASE:-3,k])@A
        rhs=grad(rows.sum(0))
        if shrink:
            parts=np.stack([grad(m) for m in rows])
            rhs*=abs(rhs)/(abs(rhs)+parts.std(0)*np.sqrt(len(rows))+1e-10)
        gram=A.T@g@A
        w=np.linalg.solve(gram+.85*np.diag(gram.diagonal().clip(.0000002)),rhs)
        return (A@w.clip(-2,2))*.6
    shared=np.tile(np.eye(WIDTH),(4,1))
    individual=np.eye(4*WIDTH)
    own=np.arange(k,WIDTH,3)
    split=np.zeros((4*WIDTH,2*WIDTH))
    split[:WIDTH,:WIDTH]=np.eye(WIDTH)
    split[WIDTH:,WIDTH:]=np.tile(np.eye(WIDTH),(3,1))
    layouts=((shared[:,own],True),(shared,True),(individual[:,np.r_[own,own+9,own+18,own+27]],True),
             (individual,False),(individual,True),(individual[:,-WIDTH:],True),(individual,True),(split,True))
    return np.stack([np.zeros(4*WIDTH)]+[solve_basis(days[:,int(i==6)],A,shrink)
                                            for i,(A,shrink) in enumerate(layouts)])


def gains(rows,weights,k):
    scores=[]
    for j,w in enumerate(weights):
        c=stack.covariance(rows[int(j==7)])
        a=np.r_[np.eye(3)[k],w]
        scores.append(a@c[:-3,-3+k]/np.sqrt((a@c[:-3,:-3]@a)*c[-3+k,-3+k]))
    return np.array(scores)-scores[0]


def train():
    OUT.mkdir(exist_ok=True)
    records=[record(d) for d in range(8,16)]
    picks,readouts,reports=[],[],[]
    for k in range(3):
        folds,daily=[],[]
        for n in (2,4,6):
            ws=fit(records[:n],k)
            folds.append([gains(v,ws,k) for v in np.sum(records[n:n+2],0).transpose(1,0,2,3)])
            daily.extend([[gains(v,ws,k) for v in day.transpose(1,0,2,3)] for day in records[n:n+2]])
        f,dd=np.asarray(folds),np.asarray(daily)
        avg=f.mean(0)
        eligible=[j for j in (1,2,3,4,5,6,8) if avg[:,j].min()>0 and (f[:,:,j]>0).sum(0).min()>=2 and
                  f[-1,:,j].min()>0 and (dd[:,:,j]>0).sum(0).min()>=4 and
                  (j<=2 or (avg[:,j]-avg[:,7]).min()>0)]
        pick=max(eligible,key=lambda j:avg[:,j].min()) if eligible else 0
        picks.append(pick);readouts.append(fit(records,k)[pick])
        reports.append(dict(folds=f.tolist(),daily=dd.tolist(),mean=avg.tolist()))
        print('target',k+1,ARMS[pick],avg[:,pick],flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(selected=picks,arms=ARMS,targets=reports),indent=2))
    if any(picks):
        np.savez(OUT/'model.npz',w=np.column_stack(readouts),basis=risk.geometry()[0],selected=picks)
    return any(picks)


def adjustment():
    with np.load(OUT/'model.npz') as archive: st=dict(archive)
    def apply(anchors):
        x=regressors(anchors,st['basis'])
        return anchors[-1]+(x[:,:,3:]@st['w']).astype('float32')
    return apply


def layer():
    first=early.incumbent.layer()
    touch_read=touch.incumbent.readout()
    corrections=[(h.incumbent.correction(),j.incumbent.adjustment()) for h,j in ((factor,joint_factor),(live,history))]
    adapt=adjustment()
    def apply(raw,initial):
        p0=first(raw,initial)
        anchors=[p0,touch_read(raw,p0)[1]]
        for step,adjust in corrections:
            prev=anchors[-1]
            p=prev.copy()
            p[:,:,2]+=step(raw,prev)
            anchors.append(adjust(raw,prev,p))
        return adapt(anchors)
    return apply


def replay_with():
    adjust=adjustment()
    def read(d,raw):
        p,h,n=zip(*(ref.read(d,raw) for ref in ANCHORS))
        assert len(set(n))==1
        return adjust(p),adjust(h),n[0]
    return stack.verifier,read


if __name__=='__main__':
    import sys
    import experiment_joint_factor_order_calibration_v1 as calibration
    torch.set_num_threads(2)
    if sys.argv[1:]==['replay'] or train():
        calibration.replay(sys.modules[__name__])
