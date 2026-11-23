"""Distinguish changes in stock ranking from rescaling existing multi-horizon forecasts."""
import json
import numpy as np
import experiment_phase_conditional_blend_v1 as phase_gate
import experiment_variance_ratio_calibration_v1 as benchmark

inn=phase_gate.inn
OUT=inn.prior.ART/'orthogonal_expert_innovations_v1'

def design(p,side,ref):
    m=ref.mean(1,keepdims=True)
    x=ref.astype(float)-m
    d=side.astype(float)-ref
    dm=d.mean(1,keepdims=True);dr=d-dm
    gram=x.transpose(0,2,1)@x/50
    rhs=x.transpose(0,2,1)@dr/50
    reg=.09*gram.diagonal(axis1=1,axis2=2).clip(.012,None)+.002
    coef=np.linalg.solve(gram+np.eye(3)[None]*reg[:,None,:],rhs)
    explained=x@coef
    novel=dr-explained
    amplitude=np.sqrt((x*x).mean(1,keepdims=True)+.01)
    rival=side-side.mean(1,keepdims=True)
    agreement=(x*rival).mean(1,keepdims=True)/(amplitude*np.sqrt((rival*rival).mean(1,keepdims=True)+.01))
    # Consensus also distinguishes a disagreeing market forecast from simple amplitude changes.
    consensus=np.tanh(m*side.mean(1,keepdims=True)/(amplitude**2+.025)*3)
    open_weight=np.exp(-np.arange(len(p))[:,None,None]/380)
    intensity=np.sqrt((novel*novel).mean(1,keepdims=True))/(.1+amplitude)
    parts=[explained,novel,novel*(agreement-.65),novel*(intensity-.5).clip(-1,1),
           explained*open_weight,novel*open_weight,dm*consensus,dm*(agreement-.65),
           dm*(intensity-.5).clip(-1,1)]
    return [np.broadcast_to(a,p.shape) for a in parts]

def fit(hist):
    grams,bs=(np.mean(part,0) for part in zip(*hist))
    coef=[]
    for j,(g,b) in enumerate(zip(grams,bs)):
        reg=1.7*np.diag(g.diagonal().clip(8e-6,None))
        mean=np.linalg.solve(g+reg,b)
        daily=np.stack([np.linalg.solve(a[j]+reg,z[j]) for a,z in hist])
        stability=np.abs(np.sign(daily).mean(0))
        coef.append((.5*mean+.5*np.median(daily,axis=0))*stability)
    return (.55*np.stack(coef)).clip(-.16,.18)

def apply(p,side,ref,coef):
    return (p+sum(a*w for a,w in zip(design(p,side,ref),coef.T))).astype('float32')

def main():
    OUT.mkdir(exist_ok=True);hist=[];report={}
    for d in range(12,20):
        if d in (14,16):
            if d==16 and report['16']['gain']<=0:break
            coef=fit(hist);rows=[]
            if d==16:np.save(OUT/'coef.npy',coef)
        p=benchmark.baseline(d);ref=np.load(phase_gate.phase.ref.OUT/f'pred_{d}.npy')
        side=np.load(phase_gate.phase.OUT/f'side_{d}.npy');y=inn.prior.labels(d).reshape(p.shape)
        z=design(p,side,ref)
        if d>=14:
            q=(p+sum(a*w for a,w in zip(z,coef.T))).astype('float32')
            if d==16:np.testing.assert_allclose(q[:201],apply(p[:201],side[:201],ref[:201],coef),atol=1e-6)
            np.save(OUT/f'pred_{d}.npy',q);rows.append(tuple(a.reshape(-1,3) for a in (p,q,y)))
        if d<16:hist.append(phase_gate.moments(p,y,z))
        if d in (15,19):
            report[str(d+1)]=inn.prior.measure(rows)
            (OUT/'results.json').write_text(json.dumps(report,indent=2))
            print(d+1,report[str(d+1)],flush=True)
if __name__=='__main__':main()
