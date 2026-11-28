"""Rebuild earlier-date predictions for the next stage, fitting each fold at its chronological cutoff."""
import json
import numpy as np
def single(module, output, source_dir, rolling_forecast):
    import experiment_dense_relative_path_inputs_v1 as rel
    output.mkdir(exist_ok=True)
    chosen = json.loads((module.OUT/"rolling.json").read_text())["selected"]
    j = module.ARMS.index(chosen)
    cal = []
    with np.load(module.OUT/"weights.npz") as f:
        code,bank,w=f["code"],f["bank"],f["w"]
    def load(day, key):
        path = rel.ref.base.CACHE / f"day_{day:02d}_{key}.npy"
        return np.load(path, mmap_mode="r")

    for d in range(6,16):
        p=rolling_forecast(d)
        if d in (6,8,12):
            prefix=np.array([d])
        if d in (8,12):
            coef=rel.relative.fit(cal,j,2)
        z=module.views(load(d, "x"),load(d, "time").astype(int),code,prefix,p)
        q=p.copy()
        if d>=8:
            q[...,2]+=(rel.relative.features(p,z)@coef).astype(p.dtype)
        np.save(output/f"rolling_{d:02d}.npy",q)
        cal.append(rel.relative.stats(p,load(d, "y"),z))
    np.testing.assert_allclose(w,rel.relative.fit(cal,j,2),rtol=1e-8,atol=1e-9)

def activity_tape(activity, OUT):
    OUT.mkdir(exist_ok=True)
    rel=activity.rel;source=activity.REFERENCE
    selected=json.loads((activity.OUT/"rolling.json").read_text())["selected"]
    short=rel.rolling_short_changes()
    with np.load(activity.OUT/"weights.npz") as f:final=dict(f)
    code=final["code"];history=[[] for _ in range(3)]
    def load(day,key):return np.load(rel.ref.base.CACHE/f"day_{day:02d}_{key}.npy",mmap_mode="r")
    def fit():
        w=np.zeros_like(final["w"])
        for j,k in selected.items():w[:,int(j)]=rel.relative.fit(history[int(j)],activity.ARMS.index(k),int(j))
        return w
    for d in range(4,16):
        if d in (4,8,12):bank=activity.train(d)
        original=np.load(source/f"rolling_{d:02d}.npy") if d>=6 else activity.calibration.future.forecast(d)
        p=original.copy()
        if d in short:p[...,0]+=short[d][:,None]
        x=activity.state(load(d,"x"),load(d,"time").astype(int),code)
        z=np.einsum("tif,fah->tiah",x,bank)
        y=load(d,"y")
        if d in (8,12):w=fit()
        if d>=6:
            q=original.copy()
            if d>=8:
                for j in map(int,selected):
                    q[...,j]+=(rel.relative.features(p,z[...,j])@w[:,j]).astype(p.dtype)
            np.save(OUT/f"rolling_{d:02d}.npy",q)
        for j in range(3):history[j].append(activity.records(p,y,z,j))
    np.testing.assert_allclose(fit(),final["w"],atol=1e-12,rtol=0)
    np.testing.assert_allclose(activity.train(16),final["bank"],atol=1e-12,rtol=0)

def horizons(model, output, load, indexed):
    output.mkdir(exist_ok=True)
    source = model.REFERENCE
    selected = json.loads((model.OUT/"rolling.json").read_text())["selected"]
    with np.load(model.OUT/"weights.npz") as f:
        final = dict(f)
    short = model.rel.rolling_short_changes()
    history = {int(h):[] for h in selected}
    def fit():
        w = np.zeros_like(final["w"])
        for h,rows in history.items():
            w[:,h] = model.rel.relative.fit(rows,model.ARMS.index(selected[str(h)]),h)
        return w

    for d in range(6,16):
        if d in (6,8,12):
            cutoff=np.array([d]) if indexed else model.train(d)
        p0=np.load(source/f"rolling_{d:02d}.npy")
        p=p0.copy()
        if d in short:
            p[:,:,0]+=short[d][:,None]
        raw,grid,y=[load(d,k) for k in ("x","time","y")]
        z=model.design(raw,grid,final["code"],p,cutoff)
        if d in (8,12):
            w=fit()
        q=p0.copy()
        for h in history:
            if d>=8:
                q[...,h]+=(model.rel.relative.features(p,z[...,h])@w[:,h]).astype(p.dtype)
            history[h].append(model.rel.relative.stats(p,y,z[...,h]))
        np.save(output/f"rolling_{d:02d}.npy",q)
    np.testing.assert_allclose(fit(),final["w"],rtol=0,atol=1e-11)

def angle(model, OUT, final):
    base=model.base
    history=[]
    for d in range(6,16):
        if d in (6,8,12):
            bank=base.train(d)
            prior=base.fit(history)[0] if history else 0.
        raw,grid,y=[model.onset.settled.load(d,key) for key in ("x","time","y")]
        p=model.panel.load_p(d)
        delta,r,z=model.infer(raw,grid,p,bank,prior)
        q=p.copy()
        if d>=8:
            q[...,2]+=delta[:,:,int(final["selected"])].astype(q.dtype)
        np.save(OUT/f"rolling_{d:02d}.npy",q)
        history.append(base.statistics(p,y,r,z))
    np.testing.assert_allclose(base.train(16),final["bank"],rtol=1e-10,atol=2e-10)
    assert base.fit(history)[0]==final["prior"]

def store(index, model, directory, final):
    directory.mkdir(exist_ok=True)
    if index in (1,2,4,6):
        if index == 1:
            forecast = model.proj.future.forecast
            source = None
        else:
            source = model.REFERENCE
            def forecast(day):
                return np.load(source/f'rolling_{day:02d}.npy')
        single(model,directory,source,forecast)
    elif index == 3:
        activity_tape(model,directory)
    elif index < 20:
        load = model.settled.load if index==5 else model.onset.settled.load
        horizons(model,directory,load,index<=12)
    elif index == 20:
        angle(model,directory,final)
    elif index == 21:
        from prepare_dense_fixed_anchor_reward_reference_v1 import rebuild_rolling
        rebuild_rolling(final)
    else:
        from prepare_dense_vintage_crossing_reference_v1 import rebuild_rolling
        rebuild_rolling(final,model,directory)
