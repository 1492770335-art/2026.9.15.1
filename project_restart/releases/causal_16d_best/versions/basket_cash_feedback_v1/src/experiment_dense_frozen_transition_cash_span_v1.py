"""Calibrate future transition-reward forecasts from a frozen cross-stock book encoder on independent dates."""
import sys
import numpy as np
import torch
import experiment_dense_transition_cash_student_v1 as student
import experiment_dense_matched_book_readout_v1 as book

strip,rel,onset,REFERENCE,read_p=student.strip,student.rel,student.onset,student.REFERENCE,student.read_p
OUT=REFERENCE.parent/'dense_frozen_transition_cash_span_v1'
ARMS=('forecastable_symmetric_component','three_component_span','beyond_total_cash')
W=99


def hidden(raw,t):
    # Fixed-size final batch makes a prefix use identical GPU kernels.
    ext=(-len(t))%64
    ticks=np.pad(t,(0,ext),mode='edge') if ext else t
    return book.hidden(raw,ticks)[0][:len(t)].astype(float)


def collect():
    OUT.mkdir(exist_ok=True)
    for d in range(16):
        raw=onset.settled.load(d,'x')
        with np.load(student.OUT/f'day_{d:02d}.npz') as f:
            t=f['t'];ok=f['ok'];cash=f['y'].reshape(len(t),50,8)
        x=hidden(raw,t)
        p=strip.center(read_p(d)[t].astype(float)) if d>=4 else np.zeros((*x.shape[:2],3))
        y=strip.center(onset.settled.load(d,'y')[t].astype(float))[...,1:]
        z=np.concatenate((p,x,cash,y),axis=-1)
        z-=((z*ok[...,None]).sum(1)/ok.sum(1)[:,None].clip(1))[:,None]
        row=z[ok]
        np.save(OUT/f'gram_{d:02d}.npy',row.T@row/len(row))


def ridge(g,r,rate):
    return np.linalg.solve(g+rate*np.diag(g.diagonal().clip(1e-7))+1e-9*np.eye(len(g)),r)


def axes(rows,inc,h):
    start=W+4*h
    g=rows.mean(0)
    co=ridge(g[3:W,3:W],g[3:W,start:start+4],.5)
    v=np.zeros((W,4))
    v[3:]=co
    v[:3]=-ridge(inc[:3,:3],inc[:3,3:W]@co,.02)
    scores=v[:,:3]
    direct=v.sum(1)
    separate=scores-direct[:,None]*((direct@inc[:W,:W]@scores)/max(direct@inc[:W,:W]@direct,1e-8))
    return (scores[:,2:],scores,separate)


def train(end):
    gs=np.stack([np.load(OUT/f'gram_{d:02d}.npy') for d in range(end)])
    weights=np.empty((W,3,2))
    for h in range(2):
        stats=[[] for _ in range(3)]
        for d in range(4,end):
            sel=np.arange(end)!=d
            ax=axes(gs[sel],gs[sel&(np.arange(end)>=4)].mean(0),h)
            g=gs[d]
            target=W+8+h
            beta=g[h+1,target]/max(g[h+1,h+1],1e-8)
            for j,v in enumerate(ax):
                gram=v.T@g[:W,:W]@v
                pay=v.T@(g[:W,target]/max(beta,.10)-g[:W,h+1])
                stats[j].append((gram,pay))
        final=axes(gs,gs[4:].mean(0),h)
        for j,ls in enumerate(stats):
            gram,pay=[np.stack(v) for v in zip(*ls)]
            m=pay.mean(0)
            stable=m**3/(m*m+pay.var(0)/len(pay)+1e-10)
            weights[:,j,h]=final[j]@ridge(gram.mean(0),stable,.7)
    return weights


def signals(raw,grid,p,bank):
    x=hidden(raw,np.arange(len(p)))
    v=np.concatenate((strip.center(p.astype(float)),x),axis=-1)
    return (v@bank.reshape(W,6)).reshape(*p.shape[:2],3,2)


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    collect()
    strip.evaluation.rolling(sys.modules[__name__])
