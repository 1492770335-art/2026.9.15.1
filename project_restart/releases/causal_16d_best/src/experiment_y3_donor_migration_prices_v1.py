"""Price book-group transitions from contemporaneous ensemble forecasts on disjoint donor stocks."""
import json
import sys

import numpy as np
import torch

import experiment_y3_book_group_migration_v1 as migration

history,parent,base,rank = migration.history,migration.parent,migration.base,migration.rank
OUT = parent.OUT.parent/'y3_donor_migration_prices_v1'
ARMS = ('unchanged','wrong donor return assignment','destination only','donor conditional price',
        'within-origin donor allocation','full return donor allocation','displaced-label discount',
        'same sign from two donor partitions','donor forecast beyond neural group price',
        'donor full and distant agreement')


def transfer(start,prob,p,part):
    n = len(start)
    alloc = torch.zeros((n,50,2),device=p.device)
    marginal = torch.zeros_like(alloc)
    wrong,one = torch.zeros_like(alloc),torch.zeros_like(alloc)
    for side in (0,1):
        src,dest = part[side],part[1-side]
        a,b = start[:,:,src],prob[:,:,src]
        r = p[:,src]-p[:,src].mean(1,keepdim=True)
        mass = torch.einsum('bksg,bksj->bkgj',a,b)
        def prices(v):
            return torch.einsum('bksg,bksj,bsh->bkgjh',a,b,v)/(mass[:,:,:,:,None]+2)
        ap,bp = start[:,:,dest],prob[:,:,dest]
        freq = torch.einsum('bksg,bksj->bkgj',ap,bp)/ap.sum(2)[:,:,:,None].clamp_min(.001)
        def read(table):
            joint = torch.einsum('bksg,bksj,bkgjh->bsh',ap,bp,table)/4
            avg = torch.einsum('bksg,bkgj,bkgjh->bsh',ap,freq,table)/4
            return joint-avg,avg
        alloc[:,dest],marginal[:,dest] = read(prices(r))
        wrong[:,dest] = read(prices(r.roll(7,1)))[0]
        simple = torch.einsum('bksj,bsh->bkjh',b,r)/(b.sum(2)[:,:,:,None]+2)
        one[:,dest] = torch.einsum('bksj,bkjh->bsh',bp-bp.mean(2,keepdim=True),simple)/4
    return alloc,marginal,wrong,one


@torch.inference_mode()
def panels(x,ph,refs,nets):
    sets = [np.arange(50),np.random.default_rng(62017).permutation(50)]
    parts = [[torch.as_tensor(ids[j::2].copy(),device='cuda') for j in range(2)] for ids in sets]
    curves = [[] for _ in refs]
    for i in range(0,len(x),96):
        v,s = [torch.from_numpy(q[i:i+96]).cuda() for q in (x,ph)]
        start = migration.groups(v)
        placebo = nets[0](v,s,start)[0]
        prob,neural = nets[1](v,s,start)
        own,group = migration.pieces(start,prob,neural)
        neural_delta = (own-group)[:,:,1]
        for j,ref in enumerate(refs):
            p = torch.from_numpy(ref[i:i+96]).cuda()
            r = torch.stack((p[:,:,2],p[:,:,2]-base.RATIO*p[:,:,1]),2)
            outputs = [transfer(start,prob,r,part) for part in parts]
            alloc,marginal,wrong,one = [(u+v)*.5 for u,v in zip(*outputs)]
            fake = .5*(transfer(start,placebo,r,parts[0])[0]+transfer(start,placebo,r,parts[1])[0])
            full,far = alloc.unbind(2)
            first,second = [v[0][:,:,1] for v in outputs]
            def agree(a,b):
                return torch.where(a*b>0,a.sign()*torch.minimum(abs(a),abs(b)),0.)
            z = torch.stack((wrong[:,:,1],one[:,:,1],far+marginal[:,:,1],far,full,far-fake[:,:,1],
                             agree(first,second),far-neural_delta,agree(full,far)),2)
            curves[j].append((z-z.mean(1,keepdim=True)).cpu().numpy())
    return [np.concatenate(c) for c in curves]


def train():
    OUT.mkdir(exist_ok=True)
    source,cal,folds,daily = [],[],[],[]
    for d in range(16):
        if d in (4,8,12):
            nets = migration.fit(source)
            if d>=8:
                w = base.calibrate(cal)
                group = []
            print('trained through day',d,flush=True)
        if d>=4:
            raw = history.trial.clock.book.rawday(d)
            y = history.trial.clock.labels(d).reshape(-1,50,3)
            ps = history.references(d,raw) if d>=8 else parent.training_references(d,raw)
            x,ph = rank.inputs.features(raw,np.arange(len(raw)))
            zs = panels(x,ph,ps,nets)
            if d==4:
                n=211
                short = rank.inputs.features(raw[:n],np.arange(n))
                for v,z in zip(panels(*short,ps[:,:n],nets),zs):
                    np.testing.assert_allclose(z[:n],v,rtol=0,atol=2e-6)
            pair = [base.records(p,y,z) for p,z in zip(ps,zs)]
            cal.append(pair)
            if d>=8:
                group.append(pair)
                daily.append([base.score(m,w)[0] for m in pair])
                if d in (11,15):
                    folds.append([base.score(m,w) for m in np.sum(group,0)])
        source.append(migration.samples(d))
    f,ds = np.array(folds),np.array(daily)
    ok = [j for j in range(3,len(ARMS)) if f[:,:,:,j].min()>0 and (ds[:,:,j]>0).sum(0).min()>=5 and
          (f[:,:,0,j].mean(0)>f[:,:,0,1:3].mean(0).max(1)).all()]
    pick = max(ok,key=lambda j:f[:,:,0,j].min()) if ok else 0
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,arms=ARMS,folds=f.tolist(),daily=ds.tolist(),windows=[[9,12],[13,16]]),indent=2))
    print('selected',ARMS[pick],flush=True)
    if pick:
        nets = migration.fit(source)
        st = {str(i)+'_'+k:v.cpu().numpy() for i,n in enumerate(nets) for k,v in n.state_dict().items()}
        np.savez(OUT/'model.npz',width=source[0][0].shape[-1],w=base.calibrate(cal)[pick-1],pick=pick-1,**st)
    return bool(pick)


def correction():
    with np.load(OUT/'model.npz') as f:
        nets = [migration.Migration(int(f['width'])).cuda().eval() for _ in range(2)]
        for i,n in enumerate(nets):
            n.load_state_dict({k:torch.from_numpy(f[str(i)+'_'+k]) for k in n.state_dict()})
        w,pick = float(f['w']),int(f['pick'])
    def compute(raw,p):
        x,ph = rank.inputs.features(raw,np.arange(len(raw)))
        return (w*panels(x,ph,[p],nets)[0][:,:,pick]).astype('float32')
    return compute


def layer():
    before,update = parent.layer(),correction()
    def apply(raw,initial):
        p = before(raw,initial).copy()
        p[:,:,2] += update(raw,p)
        return p
    return apply


def replay_with():
    update = correction()
    def read(d,raw):
        p,h,n = history.read(d,raw)
        p[:,:,2] += update(raw,p)
        h[:,:,2] += update(raw[:n],h)
        return p,h,n
    return history.stack.verifier,read


if __name__=='__main__' and (sys.argv[1:]==['replay'] or train()):
    base.replay.replay(sys.modules[__name__])
