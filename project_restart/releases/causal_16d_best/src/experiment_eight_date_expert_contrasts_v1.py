"""Compare neural expert disagreements across eight earlier rolling validation dates."""
import json
import numpy as np
from experiment_causal_cross_horizon_ridge_v1 import ART, corr, labels, measure

OUT=ART/'eight_date_expert_contrasts_v1'
SOURCES=('phase_future_book_v1','symmetric_pressure_network_v1','pairwise_spread_decoder_v1','shared_bar_gru_v1')


def baseline(day):
    if day>=14:
        return np.load(ART/'dynamic_factor_return_net_v1'/f'pred_{day}.npy')
    if day>=12:
        return np.load(ART/'intraday_relative_calibration_v1'/f'pred_{day}.npy')
    return np.load(ART/'opening_oof_cache_v1'/f'day_{day:02d}.npy')


def basis(p, experts):
    control=experts[0]
    x=np.concatenate([v-v.mean(1,keepdims=True) for v in (p,control)],axis=2).astype(float)
    gram=x.transpose(0,2,1)@x/50
    diag=.055*gram.diagonal(axis1=1,axis2=2)+.001
    g=gram+np.eye(6)*diag[:,None,:]
    rs=[]
    for source in experts[1:]:
        diff=(source-control).astype(float)
        diff-=diff.mean(1,keepdims=True)
        coef=np.linalg.solve(g,x.transpose(0,2,1)@diff/50)
        rs.append(diff-x@coef)
    ms=[]
    agreement=np.tanh(p.mean(1)*control.mean(1)/.09)
    for source in (experts[1],experts[3]):
        delta=(source-control).mean(1)
        ms.extend([delta,delta*agreement])
    return np.column_stack(ms),np.concatenate(rs,axis=2).astype('float32')


def moments(p,y,features):
    m,r=features
    a=p.reshape(-1,3).astype(float)
    b=y.reshape(-1,3).astype(float)
    a-=a.mean(0)
    b-=b.mean(0)
    slope=((a*b).sum(0)/(a*a).sum(0)).clip(.4,2.5)
    error=y/slope-p
    shared=error.mean(1)
    r=r.reshape(-1,r.shape[-1]).astype(float)
    m=m.astype(float)
    return ((m.T@m/len(m),m.T@shared/len(m)),
            (r.T@r/len(r),r.T@(error-shared[:,None]).reshape(-1,3)/len(r)))


def fit(history):
    weights=[]
    for strength,part in zip((3.8,1.8),zip(*history)):
        g,b=(np.mean(v,0) for v in zip(*part))
        reg=strength*np.diag(g.diagonal().clip(1e-5,None))
        daily=np.stack([np.linalg.solve(a+reg,c) for a,c in part])
        w=.5*np.linalg.solve(g+reg,b)+.5*np.median(daily,0)
        weights.append((.65*w*np.abs(np.sign(daily).mean(0))).clip(-.16,.16))
    return weights


def main():
    OUT.mkdir(exist_ok=True)
    hist=[];report={};daily=[]
    for d in range(8,20):
        if d in (10,12,14,16):
            w=fit(hist);rows=[]
            if d==16:
                gains=np.stack(daily)
                eligible=((gains>0).sum(0)>=4)&(gains[-2:].mean(0)>0)&(gains.mean(0)>0)
                report['eligible']=eligible.tolist()
                if not eligible.any():break
                w=[a*gate for a,gate in zip(w,eligible)]
                np.savez(OUT/'weights.npz',market=w[0],relative=w[1])
        p=baseline(d)
        source=[np.load(ART/name/f'side_{d}.npy') for name in SOURCES]
        m,r=basis(p,source)
        if d>=10:
            terms=[(m@w[0])[:,None], r@w[1]]
            q=(p+sum(terms)).astype('float32')
            if d in (10,16):
                mh,rh=basis(p[:151],[v[:151] for v in source])
                np.testing.assert_allclose(q[:151],p[:151]+(mh@w[0])[:,None]+rh@w[1],atol=2e-6)
            if d>=16:np.save(OUT/f'pred_{d}.npy',q)
        y=labels(d).reshape(p.shape)
        if 10<=d<16:
            before=corr(p.reshape(-1,3),y.reshape(-1,3))
            daily.append([corr((p+c).reshape(-1,3),y.reshape(-1,3))-before for c in terms])
        if d<16:hist.append(moments(p,y,(m,r)))
        if d>=10:rows.append(tuple(v.reshape(-1,3) for v in (p,q,y)))
        if d in (11,13,15,19):
            report[str(d+1)]=measure(rows);print(d+1,report[str(d+1)],flush=True)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()

