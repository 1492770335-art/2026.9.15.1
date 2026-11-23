"""Estimate incremental returns after recurring sequences of observed categorical book-state changes."""
import json
import numpy as np
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,corr,measure
import experiment_pairwise_book_tree_v1 as pair
import experiment_delayed_tail_band_skill_v1 as ref
import experiment_repricing_adjusted_book_flow_v1 as flow

OUT=ART/'categorical_suffix_responses_v1'
LEVELS=(27,729,19683,2916)


def groups():
    co=[]
    for d in range(8):
        x=pair.rawday(d)[:,:,83:].astype(np.int8)
        moves=np.abs(np.diff(x,axis=0)).reshape(-1,17).astype('float32')
        co.append(np.corrcoef(moves.T))
    a=np.mean(co,axis=0);out=[]
    for j in (0,1,3,4,6,8,9,11,12,14,16):
        order=np.argsort(-a[j]);neigh=[int(k) for k in order if k!=j and a[j,k]<.93]
        trip=tuple(sorted((j,neigh[0],neigh[1])))
        if trip not in out:out.append(trip)
    return np.array(out)


def contexts(raw,triplets):
    t=np.arange(len(raw),dtype=np.int32)[:,None];stock=np.arange(50)
    for trip in triplets:
        f=raw[:,:,83+trip].astype(np.int32);c=f[:,:,0]+3*f[:,:,1]+9*f[:,:,2]
        prev=np.vstack((np.full((1,50),13,dtype=np.int32),c[:-1]))
        changed=c!=prev
        last=np.maximum.accumulate(np.where(changed,t,0),axis=0)
        previous=prev[last,stock]
        old=previous[(last-1).clip(0),stock]
        age=t-last
        code=previous*27+c
        delay=(age>0).astype(np.int32)+(age>3)+(age>12)
        yield c,code,old*729+code,code*4+delay


def scales(raw):
    r=raw[:,:,0].astype(float)-.00136
    return (.45+.55*np.sqrt((8+flow.trailing(r*r,96))/(8+np.minimum(np.arange(len(raw))+1,96))[:,None])).clip(.55,3.0)


def collect(raw,trip,p,y):
    v=scales(raw)
    a=p.reshape(-1,3).astype(float);a-=a.mean(0)
    b=y.reshape(-1,3).astype(float);b-=b.mean(0)
    slope=((a*b).sum(0)/(a*a).sum(0)).clip(.4,2.5)
    e=y/slope-p;em=e.mean(1,keepdims=True);rs=(e-em)/v[:,:,None]
    bs=em/np.mean(v,axis=1)[:,None,None]
    u=np.concatenate((np.broadcast_to(bs,rs.shape),rs),axis=2)
    u-=u.mean((0,1));u=u.clip(-4,4).reshape(-1,6)
    all_tables=[]
    for keys in contexts(raw,trip):
        group=[]
        for code,n in zip(keys,LEVELS):
            idx=code.ravel()
            cnt=np.bincount(idx,minlength=n)
            totals=np.column_stack((cnt,*[np.bincount(idx,weights=u[:,j],minlength=n) for j in range(6)]))
            group.append(totals.astype('float32'))
        all_tables.append(group)
    return all_tables


def lookup(history):
    out=[]
    for group in zip(*history):
        tables=[]
        for level,all_days in enumerate(zip(*group)):
            size=LEVELS[level];codes=np.arange(size)
            total=np.sum(all_days,axis=0)
            if level==0:prior=np.zeros((size,6))
            elif level==1:prior=tables[0][codes%27]
            elif level==2:prior=tables[1][codes%729]
            else:prior=tables[1][codes//4]
            shrink=(2100,1450,950,1050)[level]
            raw=(total[:,1:]-prior*total[:,:1])/(total[:,:1]+shrink)
            # Count the dates supporting each adjustment, shrinking rare one-date patterns.
            support=np.stack([np.sign(z[:,1:]-prior*z[:,:1])*np.minimum(z[:,:1]/40,1) for z in all_days])
            strength=np.minimum(1,np.abs(support.sum(0))/max(2,.46*len(all_days)))
            tables.append((prior+strength*raw).astype('float32'))
        out.append(tables)
    return out


def predict(raw,trip,table):
    n=len(raw);m=np.zeros((n,1,3),np.float32);r=np.zeros((n,50,3),np.float32)
    v=scales(raw).astype('float32')
    for keys,tab in zip(contexts(raw,trip),table):
        c,k2,k3,k4=keys
        # Remove much of the current-state response already visible to the instantaneous predictor.
        z=.45*tab[2][k3]+.35*tab[3][k4]+.20*tab[1][k2]-.7*tab[0][c]
        m+=z[:,:,:3].mean(1,keepdims=True);r+=z[:,:,3:]*v[:,:,None]
    factor=.82/len(trip)
    return m*(factor*v.mean(1)[:,None,None]),(r-r.mean(1,keepdims=True))*factor


def main():
    OUT.mkdir(exist_ok=True);trip=groups();hist=[];daily=[];report={}
    for d in range(8,20):
        if d in (10,12,14,16):
            table=lookup(hist);rows=[]
            if d==16:
                gains=np.stack(daily);gate=((gains>0).sum(0)>=4)&(gains.mean(0)>0)&(gains[-2:].mean(0)>0)
                report['eligible']=gate.tolist()
                if not gate.any():break
                pars={f'{i}_{j}':a for i,group in enumerate(table) for j,a in enumerate(group)}
                np.savez(OUT/'model.npz',trip=trip,gate=gate,**pars)
        p=np.load(ART/'field_tokens_skew_means_v1'/f'pred_{d}.npy') if d>=16 else ref.base(d)
        raw=pair.rawday(d)
        if d>=10:
            a,b=predict(raw,trip,table)
            if d in (10,16):
                ca,cb=predict(raw[:221],trip,table)
                np.testing.assert_allclose(a[:221],ca,atol=1e-6);np.testing.assert_allclose(b[:221],cb,atol=1e-6)
            if d>=16:a*=gate[0];b*=gate[1]
            q=(p+a+b).astype('float32')
            if d>=16:np.save(OUT/f'pred_{d}.npy',q)
        y=labels(d).reshape(p.shape)
        if d<16:hist.append(collect(raw,trip,p,y))
        if d>=10:
            row=tuple(v.reshape(-1,3) for v in (p,q,y));rows.append(row)
            if d<16:
                old=corr(row[0],row[2])
                daily.append([corr((p+part).reshape(-1,3),row[2])-old for part in (a,b)])
        if d in (11,13,15,19):
            report[str(d+1)]=measure(rows);print(d+1,report[str(d+1)],flush=True)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))


def refinement():
    import experiment_field_tokens_skew_means_v1 as base
    _,first=base.refinement()
    with np.load(OUT/'model.npz') as st:
        trip=st['trip'].copy();gate=st['gate'].copy()
        tables=[[st[f'{i}_{j}'].copy() for j in range(len(LEVELS))] for i in range(len(trip))]
    def refine(raw,p):
        q=first(raw,p);a,b=predict(raw,trip,tables)
        return (q+a*gate[0]+b*gate[1]).astype('float32')
    return OUT,refine


if __name__=='__main__':
    import sys
    if '--replay' in sys.argv:
        import replay_paired_book_trees_v1 as replay
        replay.main(refinement())
    else:main()
