"""Separate persistent stock identity, book-conditioned history and recent innovations in the error GRU."""
import numpy as np
import torch

import experiment_dense_settled_error_gru_v1 as previous

rel, REFERENCE=previous.rel,previous.REFERENCE
OUT=REFERENCE.parent/"dense_name_error_memory_prior_v1"
ARMS=("stock_memory_prior","conditional_prior","live_memory_innovation")
USES_FORECAST=True


def network(end):
    pack=torch.load(previous.OUT/f"net_{end}.pt",map_location="cpu",weights_only=True)
    model=previous.SettledDirection(len(pack["unit"])).double().eval()
    model.load_state_dict(pack["state"])
    return model,pack


def key(u):
    q=u[:,:,-3:].numpy()
    return np.concatenate((np.tanh(q),np.tanh(q)**2,np.tanh(q[:,:,1:2]*q[:,:,2:])),axis=-1)


def train(end):
    if not (previous.OUT/f"net_{end}.pt").exists():
        previous.train(end)
    return fit_prior(end)


@torch.inference_mode()
def fit_prior(end):
    model,pack=network(end)
    rows=[]
    for d in range(4,end):
        with np.load(previous.OUT/f"day_{d:02d}.npz") as f:
            states=f["x"][::2]
            histories=f["seq"][::2]
        xs=[];hs=[]
        for i in range(0,len(states),48):
            u=(torch.from_numpy(states[i:i+48]).double()/pack["unit"]).clamp(-5,5)
            seq=torch.from_numpy(histories[i:i+48]).double()
            _,h=model.gru(seq.reshape(-1,previous.STEPS,8))
            xs.append(key(u));hs.append(h[0].reshape(*u.shape[:2],-1).numpy())
        k=np.concatenate(xs);h=np.concatenate(hs)
        x0=k.mean(0);h0=h.mean(0)
        k-=x0;h-=h0
        rows.append((x0,h0,np.einsum("tif,tig->ifg",k,k)/len(k),np.einsum("tif,tij->ifj",k,h)/len(k)))
    k,hidden,g,r=map(np.stack,zip(*rows))
    baseline=hidden.mean((0,1))
    excess=hidden-hidden.mean(1,keepdims=True)
    mean=excess.mean(0)
    noise=excess.var(0)/len(rows)
    shrink=mean**2/(mean**2+noise+1e-7)
    per_name=baseline+mean*shrink
    gram=g.mean(0)
    # Pool the slope across names, then retain individual deviations only if they repeat across dates.
    shared=np.linalg.solve(gram.mean(0)+.15*np.eye(gram.shape[-1]),r.mean((0,1)))
    inverse=np.linalg.inv(.75*gram+.25*gram.mean(0)+.24*np.eye(gram.shape[-1]))
    slopes=inverse@(r-g@shared)
    m=slopes.mean(0)
    reliable=m*m/(m*m+slopes.var(0)/len(rows)+1e-7)
    weights=shared+m*reliable
    np.savez(OUT/f"prior_{end}.npz",pooled=baseline,mean=per_name,key=k.mean(0),w=weights)
    return np.array([end])


@torch.inference_mode()
def views(raw,grid,code,bank,p):
    end=int(bank[0])
    model,pack=network(end)
    with np.load(OUT/f"prior_{end}.npz") as f:
        pop,prior,center,w=[torch.from_numpy(f[k]) for k in ("pooled","mean","key","w")]
    x=previous.head.ranks.band.pairs.net.features(raw,grid,code,p)
    stream,starts=previous.observations(raw,grid,p)
    result=[]
    for t in np.array_split(np.arange(len(x)),(len(x)+47)//48):
        u=(torch.from_numpy(x[t]).double()/pack["unit"]).clamp(-5,5)
        book=model.book(u)
        c,local,radius=model.base(book).split((1,5,5),-1)
        base=c+local
        mass=radius.softmax(-1)
        zero=model.effect(torch.cat((book,torch.zeros((*u.shape[:2],12),dtype=u.dtype)),-1))
        def evaluate(hidden):
            delta=model.effect(torch.cat((book,hidden.expand(*u.shape[:2],-1)),-1))-zero
            b=base+delta[:,:,:1]
            return (mass*pack["span"]*((b+delta[:,:,1:]).sigmoid()-b.sigmoid())).sum(-1)
        hx=prior+torch.einsum("tif,ifk->tik",torch.from_numpy(key(u))-center,w)
        seq=torch.from_numpy(previous.history(stream,starts,t)).double()
        _,live=model.gru(seq.reshape(-1,previous.STEPS,8))
        typical=evaluate(prior);matched=evaluate(hx)
        result.append(torch.stack((typical-evaluate(pop),matched-typical,
                                   evaluate(live[0].reshape(*u.shape[:2],-1))-matched),-1).numpy())
    z=np.concatenate(result)
    return z-z.mean(1,keepdims=True)


if __name__=="__main__":
    torch.set_num_threads(1)
    rel.main(OUT,ARMS,train,views,with_prediction=True,rolling_reference=REFERENCE)
