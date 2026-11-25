"""Transfer innovations and persistence of earlier Y3 specialist revisions to each of the three horizons."""
import json
import sys
import numpy as np
import torch

import experiment_joint_factor_order_calibration_v1 as replay
import y3_book_activity_skill_references as history
import joint_factor_stage_references as original
import y3_future_book_comoment_references as book
import joint_interval_covariance_scale_references as scale
import y3_horizon_geometry_training as study

trial,parent,stack = history.trial,history.incumbent,history.stack
CHECKPOINTS = (original,book,scale,history)
OUT = trial.clock.ART/'joint_revision_timescale_transfer_v1'
ARMS = ('unchanged','current revisions','fast revision innovation','slow persistent revision','short and long changes',
        'short-long disagreement pricing','agreement across specialist families','older revision control','all temporal bands')
GROUPS = ((),(0,1,2),(3,4,5),(9,10,11),tuple(range(3,12)),tuple(range(3,15)),(15,16,17),(18,19,20),
          tuple(range(18)))
CANDIDATES = (2,3,4,5,6,8)
CONTROLS = (1,7)


def design(anchors):
    p = anchors[-1]
    shifts = np.stack([b[:,:,2]-a[:,:,2] for a,b in zip(anchors,anchors[1:])],axis=2).astype(float)
    fast,slow = [study.ema(shifts,span) for span in (6,64)]
    z = trial.clock.center(p.astype(float))
    norm = np.sqrt((z*z).mean(1,keepdims=True)+.003)
    coords = z/norm
    gate = np.tanh(coords[:,:,2]*(coords[:,:,0]+coords[:,:,1])*.5)
    agree = np.sign(shifts)*np.sign(shifts.sum(2,keepdims=True)-shifts)
    t = np.arange(len(p))
    lag = np.maximum(t-48,np.where(t>=2401,2401,0))
    return np.concatenate((shifts,shifts-fast,fast-slow,slow,shifts*gate[:,:,None],
                           shifts*agree,shifts[lag]),axis=2).astype('float32')


def records(day):
    raw = trial.clock.book.rawday(day)
    y = trial.clock.labels(day).reshape(-1,50,3)
    panels = [h.references(day,raw) for h in CHECKPOINTS]
    stats = []
    for anchors in zip(*panels):
        z = design(anchors)
        v = np.concatenate((np.ones_like(y[:,:,:1]),anchors[-1],z,y),axis=2)
        v = v.reshape(-1,v.shape[-1]).astype(float)
        stats.append(v.T@v)
    return stats


def fit(records,k):
    idx = np.r_[0,1+k,np.arange(4,25),25+k]
    rows = [[m[np.ix_(idx,idx)] for m in d] for d in records]
    return rows,study.weights(sys.modules[__name__],rows)


def train():
    OUT.mkdir(exist_ok=True)
    rows = [records(day) for day in range(8,16)]
    picks,heads,reports = [],[],[]
    for k in range(3):
        folds,days = [],[]
        for n in (2,4,6):
            _,w = fit(rows[:n],k)
            idx = np.r_[0,1+k,np.arange(4,25),25+k]
            test = [[m[np.ix_(idx,idx)] for m in d] for d in rows[n:n+2]]
            gain = study.history.incumbent.gain
            folds.append([gain(m,w) for m in np.sum(test,axis=0)])
            days.extend([[gain(m,w) for m in d] for d in test])
        f,d = np.array(folds),np.array(days)
        avg = f.mean(0)
        good = [i for i in CANDIDATES if (avg[:,i]>0).all() and (f[-1,:,i]>0).all() and
                (f[:,:,i]>0).sum(0).min()>=2 and (d[:,:,i]>0).sum(0).min()>=4 and
                all((avg[:,i]>avg[:,j]).all() for j in CONTROLS)]
        pick = max(good,key=lambda i:avg[:,i].min()) if good else 0
        picks.append(pick)
        heads.append(fit(rows,k)[1][pick])
        reports.append(dict(mean=avg.tolist(),daily=d.tolist(),folds=f.tolist()))
        print('Y',k+1,ARMS[pick],avg[:,pick],flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(selected=picks,arms=ARMS,targets=reports)))
    if any(picks):
        np.savez(OUT/'model.npz',w=heads)
    return any(picks)


def adjustment():
    with np.load(OUT/'model.npz') as f:
        w = f['w']
    def apply(anchors):
        p = anchors[-1]
        return p+(p*w[:,0]+design(anchors)@w[:,1:].T).astype('float32')
    return apply


def replay_with():
    apply = adjustment()
    def read(d,raw):
        states = [h.read(d,raw) for h in CHECKPOINTS]
        assert len({s[2] for s in states})==1
        return apply([s[0] for s in states]),apply([s[1] for s in states]),states[0][2]
    return stack.verifier,read


def snapshot_layers():
    # Reuse each already computed intermediate; none of the four snapshots needs a second backbone pass.
    import experiment_y3_sibling_spread_network_v1 as spread
    import experiment_joint_separated_peer_transfer_v1 as separated
    import experiment_y3_peer_carry_impulse_v1 as impulse
    import experiment_y3_book_covariance_residuals_v1 as covariance
    import experiment_joint_book_comoment_calibration_v1 as comoment
    import experiment_y3_interval_covariance_bridge_v1 as bridge
    import experiment_y3_dynamic_horizon_geometry_v1 as geometry
    import experiment_y3_horizon_covariance_transport_v1 as transport
    import experiment_joint_sibling_horizon_skill_v1 as siblings
    import experiment_y3_horizon_agreement_skill_v1 as agreement

    def long_step(mod):
        correction = mod.correction()
        def advance(raw,p):
            q = p.copy()
            q[:,:,2] += correction(raw,p)
            return q
        return advance

    step_spread = long_step(spread)
    separate = separated.adjustment()
    carry = impulse.context.adjustment(impulse)
    step_book = long_step(book.incumbent)
    step_cov = long_step(covariance)
    calibrate_book = comoment.adjustment()
    step_bridge = long_step(bridge)
    calibrate_scale = scale.incumbent.adjustment()
    tail = (long_step(geometry),long_step(transport),siblings.adjustment(),long_step(agreement),long_step(parent))

    def snapshots(raw,start):
        p = step_spread(raw,start)
        book_state = step_book(raw,carry(raw,p,separate(p)))
        p = calibrate_book(raw,book_state,step_cov(raw,book_state))
        scale_state = calibrate_scale(p,step_bridge(raw,p))
        final = scale_state
        for advance in tail:
            final = advance(raw,final)
        return start,book_state,scale_state,final
    return snapshots


def layer():
    start,snapshots,apply = original.incumbent.layer(),snapshot_layers(),adjustment()
    return lambda raw,p:apply(snapshots(raw,start(raw,p)))


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['replay'] or train():
        replay.replay(sys.modules[__name__])
