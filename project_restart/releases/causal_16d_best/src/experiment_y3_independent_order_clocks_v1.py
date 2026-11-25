"""Learn which pressure-to-activity delays matter separately from the age of the resulting event."""
from functools import lru_cache
import sys
import numpy as np
import torch
import experiment_y3_order_response_filter_v1 as filters
import experiment_y3_same_deadline_event_curve_v1 as linear
import joint_early_book_references as history

study,trial,parent=history.study,history.trial,history.incumbent
signature=parent.pressure.signature
OUT=trial.clock.ART/'y3_independent_order_clocks_v1'
ARMS=('simultaneous shock','matched clocks','independent clocks','new clock combinations','late window only',
      'middle window only','relative middle versus terminal','tied separations','tied event ages','reordered ages',
      'all window readouts together')
CANDIDATES=(2,3,4,5,6,7,8,10)
CONTROLS=(0,1,9)
SPANS=(4,16,64)
WIDTH=12


def design_bank(raw,t):
    v=signature.impact.arrival.signals(raw).astype('float32')
    blocks=[]
    # Reset the two filters at the session boundary.
    for start,end in ((0,min(len(raw),2401)),(2401,len(raw))):
        selected=t[(t>=start)&(t<end)]-start
        if not len(selected):
            continue
        p=np.tanh(v[start:end][:,:,[0,2,3,6]])
        a=np.tanh(np.stack((abs(v[start:end,:,0]),abs(v[start:end,:,1]),
                            abs(v[start:end,:,3:5]).mean(2)),2))
        z=np.concatenate((p,a),2)
        z-=filters.smooth(z,128)
        p,a=z[:,:,:4],z[:,:,4:]
        def product(pressure,activity):
            return (pressure[:,:,:,None]*activity[:,:,None,:]).reshape(len(z),50,WIDTH)
        groups=[filters.smooth(product(p,a),16)[selected]]
        for span in SPANS:
            old=filters.smooth(z,span)
            lag=np.concatenate((np.zeros_like(old[:1]),old[:-1]))
            first=product(lag[:,:,:4],a)-product(p,lag[:,:,4:])
            for age in SPANS:
                groups.append(filters.smooth(first,age)[selected])
        blocks.append(trial.clock.center(np.concatenate(groups,2)))
    return np.concatenate(blocks)


@lru_cache(maxsize=16)
def record(d):
    raw=trial.clock.book.rawday(d)
    t=np.arange(90+d%7,len(raw)-60,7)
    t=t[(t+60<2401)|(t>=2401)]
    x=design_bank(raw,t).reshape(len(t)*50,-1)
    y=trial.clock.labels(d).reshape(-1,50,3)
    targets=np.stack([y[t+h,:,1] for h in (0,20,40)],2)
    targets=trial.clock.center(targets).clip(-4,4).reshape(len(x),3)
    return (x.T@x/len(x)).astype(float),(x.T@targets/len(x)).astype(float)


@lru_cache(maxsize=7)
def parameters(end):
    gs,bs=[np.stack(v) for v in zip(*[record(d) for d in range(end)])]
    g=gs.mean(0)
    identity=np.eye(WIDTH)
    # Each basis controls the sharing of weights across the two filter axes.
    def build(groups):
        a=np.zeros((10*WIDTH,len(groups)*WIDTH))
        for i,ids in enumerate(groups):
            for j in ids:
                a[j*WIDTH:(j+1)*WIDTH,i*WIDTH:(i+1)*WIDTH]=identity/len(ids)
        return a
    diag=[1,5,9]
    layouts=([[0]],[[i] for i in diag],[[i] for i in range(1,10)],
             [[1+3*s+j for s in range(3)] for j in range(3)],[[1+3*s+j for j in range(3)] for s in range(3)])
    heads=[]
    for groups in layouts:
        a=build(groups)
        bg=a.T@g@a
        gradients=a.T@bs
        mean=gradients.mean(0)
        mean*=abs(mean)/(abs(mean)+gradients.std(0)/np.sqrt(end)+1e-6)
        heads.append(a@np.linalg.solve(bg+.9*np.diag(bg.diagonal().clip(.00001)),mean))
    sim,tied,whole,sep,ages=heads
    perm=np.r_[0,np.arange(1,10).reshape(3,3)[:,::-1].ravel()]
    bad=whole.reshape(10,WIDTH,3)[perm].reshape(10*WIDTH,3)
    return np.column_stack((sim.sum(1),tied.sum(1),whole.sum(1),(whole-tied).sum(1),
                            whole[:,2],whole[:,1],whole[:,2]-whole[:,1],sep.sum(1),ages.sum(1),bad.sum(1),whole)).astype('float32')


def historical_design(day):
    w=parameters(max(4,day-day%2))
    cache={}
    def build(raw,p,t):
        key=len(raw),t.tobytes()
        if key not in cache:
            cache[key]=design_bank(raw,t)@w
        return np.concatenate((trial.clock.center(p[t]),cache[key]),2)
    return build


def fit(rows):
    sets=[np.array([3+i]) for i in range(10)]+[np.arange(13,16)]
    return dict(**linear.gram(rows,[(s,3,.35) for s in sets]),clocks=parameters(len(rows)+4))


changes=linear.changes


def correction():
    with np.load(OUT/'model.npz') as f:
        st=dict(f)
    def apply(raw,p):
        x=design_bank(raw,np.arange(len(raw)))@st['clocks']
        return changes(np.concatenate((trial.clock.center(p),x),2),st)[int(st['pick'])]
    return apply


def layer():
    return study.layer(sys.modules[__name__],parent)


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
