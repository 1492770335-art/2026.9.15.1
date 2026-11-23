"""Decode predicted future quote states into returns with a training-only structural decoder."""
import json
import numpy as np
import torch
import experiment_phase_future_book_v1 as phase
import experiment_consensus_horizon_portfolios_v1 as blend
import experiment_pairwise_book_tree_v1 as book
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'auxiliary_quote_decoder_v1'
H=3
DIM=len(phase.OBS)


def net(end):
    st=torch.load(phase.OUT/f'net_{end}.pth',map_location='cpu',weights_only=True)
    model=phase.BookNet(st['current.0.weight'].shape[1]).cuda().eval()
    model.load_state_dict(st)
    return model


@torch.inference_mode()
def forecast(raw,t,model):
    aux=np.empty((len(t),50,3,DIM),np.float32)
    for start in range(0,len(t),900):
        ids=t[start:start+900]
        x,v=phase.spline.features(raw,ids)
        for k in range(0,len(ids),56):
            sl=slice(k,k+56)
            age=np.exp(-ids[sl,None]/400).astype('float32')
            q=model(torch.from_numpy(x[sl]).cuda().float(),torch.from_numpy((.45+.55*v[sl]).clip(.55,3)).cuda(),torch.from_numpy(age).cuda())[1]
            aux[start+k:start+k+len(q)]=q.cpu().numpy().reshape(-1,50,3,DIM)
    return aux


def designs(raw,t,aux,j):
    x0=np.arcsinh(raw[t][:,:,phase.OBS].astype('float32'))
    # Predict the change between two consecutive forward averages for the later bands.
    change=aux[:,:,j]-(aux[:,:,j-1] if j else 0)
    ab=abs(x0)
    z=np.concatenate((change,np.tanh(1.5*change),change*np.tanh(ab-.6)),axis=2)
    current=np.concatenate((x0,np.tanh(x0)),axis=2)
    z=np.concatenate((current,z),axis=2)
    m=z.mean(1)
    return m,z-m[:,None]


def moments(raw,t,aux,y):
    result=[]
    truth=np.diff(y/blend.UNITS,axis=2,prepend=np.zeros_like(y[:,:,:1]))*blend.UNITS
    for j in range(3):
        m,r=designs(raw,t,aux,j)
        market=truth[:,:,j].mean(1)
        for x,out in ((m,market),(r,truth[:,:,j]-market[:,None])):
            a=x.reshape(-1,x.shape[-1]).astype(float);b=out.ravel().astype(float).clip(-5,5)
            a-=a.mean(0);b-=b.mean()
            result.append((a.T@a/len(a),a.T@b/len(a)))
    return result


def structural(hist):
    weights=[]
    for band in zip(*hist):
        g,b=[np.mean(x,axis=0) for x in zip(*band)]
        ridge=.18*np.diag(g.diagonal().clip(.0003,None))
        weights.append(np.linalg.solve(g+ridge,b).astype('float32'))
    return weights


def components(raw,t,aux,coef,p):
    local=[];shared=[]
    for j in range(3):
        m,r=designs(raw,t,aux,j)
        # The decoder distinguishes information from the initial quote and the forecast path.
        split=2*DIM
        for x,w,buf in ((m,coef[2*j],shared),(r,coef[2*j+1],local)):
            buf.extend((x[...,:split]@w[:split],x[...,split:]@w[split:]))
    def stack(features,ref):
        a=np.stack(features,axis=-1)
        return np.concatenate((a,ref),axis=-1).astype('float32')
    mp=p.mean(1)
    return stack(shared,mp),stack(local,p-mp[:,None])


def cal_moments(p,y,components):
    return phase_stats(p,y,components)


def phase_stats(p,y,components):
    from experiment_eight_date_expert_contrasts_v1 import moments as stats
    return stats(p,y,components)


def readout(history):
    from experiment_eight_date_expert_contrasts_v1 import fit
    return np.stack(fit(history)).astype('float32')


def apply(p,z,w):
    return (p+(z[0]@w[0])[:,None]+z[1]@w[1]).astype('float32')


def main():
    OUT.mkdir(exist_ok=True);hist=[];cal=[];priors=[];report={'folds':[]};daily=[]
    for end in (8,10,12,14):
        for d in range(len(hist),end):
            raw=book.rawday(d);t=np.arange(d%8,len(raw)-61,8)
            aux=phase.auxiliary(raw,t).reshape(len(t),50,3,DIM).astype('float32')
            hist.append(moments(raw,t,aux,labels(d).reshape(-1,50,3)[t]))
        coef=structural(hist);model=net(8 if end<=10 else end)
        w=readout(cal) if cal else None;c=blend.fixed_weight(priors);rows=[]
        for d in range(end,end+2):
            p,y,fs,prior=blend.training_day(d,priors);priors.append(prior)
            p=blend.parent.fixed_forecast(p,fs,c)
            raw=book.rawday(d);t=np.arange(d%3,len(raw),3)
            predicted=forecast(raw,t,model)
            if d==end:
                head=t[t<193]
                np.testing.assert_allclose(predicted[:len(head)],forecast(raw[:193],head,model),atol=5e-5,rtol=0)
            p=p[t];y=y[t];z=components(raw,t,predicted,coef,p)
            if w is not None:
                q=apply(p,z,w)
                row=tuple(x.reshape(-1,3) for x in (p,q,y));rows.append(row)
                score=measure([row]);daily.append(np.subtract(score['candidate'],score['base']))
            cal.append(cal_moments(p,y,z))
        if rows:
            s=measure(rows);report['folds'].append(s);print(end+1,s,flush=True)
    gain=np.stack([np.subtract(f['candidate'],f['base']) for f in report['folds']])
    gate=(gain.min(0)>0)&((np.stack(daily)>0).sum(0)>=4)
    report['retained']=gate.tolist();path=OUT/'results.json';path.write_text(json.dumps(report,indent=2))
    if not gate.any():return
    for d in range(14,16):
        raw=book.rawday(d);t=np.arange(d%8,len(raw)-61,8)
        aux=phase.auxiliary(raw,t).reshape(len(t),50,3,DIM).astype('float32')
        hist.append(moments(raw,t,aux,labels(d).reshape(-1,50,3)[t]))
    coef=structural(hist);w=readout(cal)*gate;model=net(16)
    np.savez(OUT/'weights.npz',structural=coef,readout=w)
    from experiment_predictive_book_modes_v1 import validation
    from experiment_replicated_expert_consensus_v1 import sources
    models=sources();fixed=np.load(blend.PARENT/'weights.npy')[0]
    def advance(raw,p):
        p=blend.parent.fixed_forecast(p,[fn(raw) for fn in models],fixed)
        t=np.arange(len(p));return apply(p,components(raw,t,forecast(raw,t,model),coef,p),w)
    result=validation(advance)
    ref=json.loads((blend.PARENT/'raw_replay.json').read_text())
    result['gain_over_fixed']=result['pooled']['score']-ref['pooled']['score']
    report['development']=result;path.write_text(json.dumps(report,indent=2))
    print(result,flush=True)


if __name__=='__main__':main()
