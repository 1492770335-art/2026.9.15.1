"""Rebuild the frozen first-16-day predictor using only feature frames and saved parameters."""
import json
import hashlib
import os
from pathlib import Path
import numpy as np
import polars as pl
import torch
import package_opening_expert_v1 as opening
import experiment_multiscale_return_net_v1 as v1
import experiment_causal_path_tcn_v1 as path
import experiment_relative_price_endpoint_v1 as endpoint
import experiment_factor_exposure_return_v1 as exposure
import experiment_pooled_price_impact_v1 as impact
import experiment_separate_market_relative_v1 as separate
import experiment_stock_calibration_v1 as stock
import experiment_endpoint_innovation_v1 as innovation
import experiment_relative_residual_tree_v1 as tree
import experiment_term_structure_net_v1 as curve
import experiment_source_alpha_impact_v1 as source
import experiment_matured_market_innovations_v1 as matured
import experiment_curve_market_components_v1 as market
import experiment_rolling_quantile_residual_v1 as ranks
import experiment_curve_quantile_market_v1 as final

DATA=Path(__file__).resolve().parent/'project/data/public.parquet'

stage_observer=None


def observe(name,value):
    if stage_observer is not None:stage_observer(name,value)


def selection(module,key):
    return np.array(json.loads((module.OUT/'results.json').read_text())['16'][key])

def neural(model,folder,filename):
    model.load_state_dict(torch.load(folder/filename,weights_only=True,map_location='cpu'))
    return model.eval().cuda()

def params(folder):
    with np.load(folder/'model.npz') as pack:
        return {k:pack[k] for k in pack.files}

def time_features(raw,fn,lookback,decay):
    out=None
    for start in range(0,len(raw),480):
        stop=min(start+480,len(raw));prefix=max(0,start-lookback)
        block=fn(raw[prefix:stop])[start-prefix:]
        if out is None:out=np.empty((len(raw),*block.shape[1:]),np.float16)
        out[start:stop]=block
        out[start:stop,:,-1]=np.exp(-np.arange(start,stop)/decay).astype(np.float16)[:,None]
    return out

@torch.inference_mode()
def net_batches(model,x,interpret):
    out=[]
    for i in range(0,len(x),64):
        z=torch.from_numpy(x[i:i+64]).cuda().float()
        out.append(interpret(model(z)).cpu().numpy())
    return np.concatenate(out)

def raw_curve(raw,feat):
    state=torch.load(curve.OUT/'net_16.pth',weights_only=True,map_location='cpu')
    model=curve.CurveNet(state['local.0.weight'].shape[1]);model.load_state_dict(state);model.eval().cuda()
    pred=np.empty((len(raw),50,3),np.float32)
    with torch.inference_mode():
        for i in range(0,len(raw),160):
            t=np.arange(i,min(len(raw),i+160));z,v=curve.feature_rows(raw,feat,t)
            r,m=model(torch.from_numpy(z).cuda().float(),(.4+.6*torch.from_numpy(v).cuda()).clamp(.5,3))
            pred[t]=(r+model.beta[None,:,None]*m[:,None,:])[:,:,curve.J].cpu().numpy()*(np.sqrt(curve.H[curve.J])*v1.SCALE)
    return pred

def build(raw,opening_pred):
    feat=time_features(raw,v1.features,256,256.)
    net=neural(v1.ReturnNet(713),v1.OUT,'net_16_10.pth')
    p=v1.blend(opening_pred,net_batches(net,feat,v1.decode),selection(v1,'weights')).astype(np.float32)
    tcn=neural(path.PathNet(),path.OUT,'net_16_14.pth')
    subset=np.concatenate((feat[:,:,:100],feat[:,:,300:400],feat[:,:,500:600],feat[:,:,700:]),axis=2)
    a=separate.relative(v1.blend(p,path.forward_blocks(tcn,subset),selection(path,'weights')).astype(np.float32)-p)
    del subset
    curve_pred=raw_curve(raw,feat);del feat
    with np.load(endpoint.OUT/'stats_16.npz') as stats:
        z=time_features(raw,lambda x:endpoint.features(x,(stats['mid'],stats['scale'])),240,256.)
    net=neural(endpoint.RelativeNet(z.shape[-1]),endpoint.OUT,'net_16_14.pth')
    b=separate.relative(v1.blend(p,net_batches(net,z,lambda x:x[0]),selection(endpoint,'weights')).astype(np.float32)-p)
    del z
    ex=params(exposure.OUT)
    s=exposure.features(raw,p,(ex['beta'],ex['scale'],ex['proj']))@ex['coef']
    w=selection(separate,'weights')
    im=params(impact.OUT);z=impact.states(raw)
    n=((z@im['local'])+(impact.build(z,im['proj'])@im['cross']).reshape(p.shape))*im['vol'][None,:,None]
    pool=v1.blend(p,n.astype(np.float32),im['blend']);del z,n
    p=(pool.astype(np.float32)+w[0]*a+w[1]*b+w[2]*s.astype(np.float32)).astype(np.float32).astype(float)
    sc=params(stock.OUT)
    h=np.einsum('tsf,sfk->tsk',stock.covariates(p,raw),sc['coef'],optimize=True)
    p=(p+sc['strength'][0]*separate.relative(h)+sc['strength'][1]*sc['beta_diff'][None]*p.mean(1,keepdims=True)).astype(np.float32).astype(float)
    observe('stock_calibration_v1',p)
    inn=params(innovation.OUT);m,z=innovation.designs(raw,p,inn['graph'])
    p=(p+inn['weights'][0]*(m@inn['cm'])[:,None]+inn['weights'][1]*(z@inn['cr'])).astype(np.float32).astype(float);del z
    observe('endpoint_innovation_v1',p)
    t=np.arange(len(raw))
    models=[tree.lgb.Booster(model_file=str(tree.OUT/f'target_{j}.txt')) for j in range(3)]
    tr=np.empty_like(p)
    for i in range(0,len(t),192):
        ti=t[i:i+192];tr[ti]=tree.sides(raw,p,models,ti)
    p=(p+selection(tree,'weight')*tr).astype(np.float32);del tr
    result=v1.blend(p,curve_pred,selection(curve,'w')).astype(np.float32).astype(float)
    w=np.load(market.OUT/'weights.npy')
    assert not w[2].any()
    h=matured.adapt(raw,p.astype(float));mw=np.load(matured.OUT/'weights.npy')
    result+=w[0]*np.einsum('tkj,kj->tk',h,mw)[:,None]
    src=params(source.OUT);z=source.states(raw);loc=z@src['local']
    flow=(loc+(source.design(z,loc,src['proj'],src['source'])@src['coef']).reshape(p.shape))*src['vol'][None,:,None]
    result+=w[1]*src['blend'][0]*(flow.astype(np.float32).mean(1)-pool.mean(1))[:,None]
    del z,flow,loc
    qr=params(ranks.OUT)
    for i in range(0,len(t),192):
        ti=t[i:i+192];_,z=ranks.features(raw,ti,qr['proj'])
        result[ti]+=separate.relative(z@qr['cr'])*qr['weights'][1]
    return result

def summary(s):
    n,px,yy,pp,ys,py=s
    corr=(py-px*yy/n)/np.sqrt((pp-px**2/n)*(ys-yy**2/n))
    return {'rows':int(n[0]),'corr':corr.tolist(),'score':float(corr@np.array([.2,.3,.5]))}

def main(predictor=build, output_dir=final.OUT, reference_score=None, deep_prefix=False):
    torch.set_num_threads(2)
    dest=v1.cache.opening.OUT/'commit_opening_expert'
    ev=opening.Evaluator(str(DATA),str(dest),'cuda:0')
    weights=np.array(json.loads((dest/'conf/Model_0.json').read_text())['opening_mix'])
    totals=np.zeros((6,3));record={'daily':{}}
    for day in range(16,20):
        scan=pl.scan_parquet(DATA).filter(pl.col('date')==day).sort('time','code')
        frame=scan.select('time','code',*[f'f{i:03d}' for i in range(100)]).collect()
        ticks=frame['time'].to_numpy().reshape(-1,50);codes=frame['code'].to_numpy().reshape(-1,50)
        assert (ticks==np.arange(len(ticks))[:,None]).all() and (codes==np.arange(50)[None,:]).all()
        raw=frame.select(pl.exclude('time','code')).to_numpy().astype(np.float16).reshape(-1,50,100)
        del frame
        (_,op),stream_err=opening.replay_features(ev,raw,weights,80 if day==16 else 12)
        op=op.reshape(-1,50,3)
        pred=predictor(raw,op)
        prefix=(1337,2741,457,3199)[day-16] if deep_prefix else (173 if day==16 else 83)
        (_,ow),_=opening.replay_features(ev,raw[:prefix],weights,2)
        head=predictor(raw[:prefix],ow.reshape(prefix,50,3))
        prefix_err=float(abs(head-pred[:prefix]).max())
        assert np.isfinite(pred).all() and prefix_err < (1e-4 if deep_prefix else .02),(day,prefix_err)
        if deep_prefix and os.environ.get('SAVE_RAW_PREDICTION_TRACE') == '1':
            directory=output_dir/'raw_trace'
            directory.mkdir(exist_ok=True)
            digest=hashlib.sha256(np.ascontiguousarray(raw).view(np.uint8)).hexdigest()
            np.savez(directory/f'day_{day}.npz',full=pred,head=head,raw_sha256=digest,
                     prefix_ticks=prefix,opening_stream_max_error=stream_err)
        # Labels are read only after the full and truncated forecasts are produced.
        y=scan.select('y0','y1','y2').collect().to_numpy().astype(float)
        p=pred.reshape(-1,3).astype(np.float64,copy=False);ref=np.load(output_dir/f'pred_{day}.npy').reshape(-1,3)
        day_s=np.stack((np.full(3,len(y)),p.sum(0),y.sum(0),(p*p).sum(0),(y*y).sum(0),(p*y).sum(0)))
        entry=dict(**summary(day_s),prefix_ticks=prefix,opening_stream_max_error=stream_err,final_prefix_max_error=prefix_err,
                   cache_rmse=float(np.mean((p-ref)**2)**.5),cache_max_error=float(np.max(abs(p-ref))))
        record['daily'][day+1]=entry;totals+=day_s;print(day+1,entry,flush=True)
    record['pooled']=summary(totals)
    cached=reference_score if reference_score is not None else json.loads((output_dir/'results.json').read_text())['outer']['score']
    assert record['pooled']['rows']==936250
    assert abs(cached-record['pooled']['score'])<2e-6,(cached,record['pooled'])
    (output_dir/'raw_replay.json').write_text(json.dumps(record,indent=2))
    print('independent raw replay',record['pooled'],flush=True)
if __name__=='__main__':main()
