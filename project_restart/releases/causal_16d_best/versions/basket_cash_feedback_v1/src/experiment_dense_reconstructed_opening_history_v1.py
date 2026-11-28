"""Infer missing opening-window sufficient statistics from artificial history cuts on training dates."""
import numpy as np
import torch

import experiment_dense_directed_book_transfer_v1 as graph

book,onset,rel,REFERENCE = graph.book,graph.onset,graph.rel,graph.REFERENCE
OUT = REFERENCE.parent / "dense_reconstructed_opening_history_v1"
ARMS = ("recovered_quote_history","recovered_price_history","both_histories")
CUTS = np.array((0,1,5,20,60,120,240))
F = book.phase.spline.FIELDS
FIELD_N = len(F)
W = FIELD_N+2


def inputs(data):
    x = np.arcsinh(data[:,:,:83].astype(np.float32))
    x[:,:,19] = 0
    return np.concatenate((np.ones((*x.shape[:2],1)),x,np.tanh(2*x)),axis=2)


def collect():
    OUT.mkdir(exist_ok=True)
    for d in range(16):
        raw,grid = [onset.settled.load(d,k) for k in ("x","time")]
        gram = 0.
        n = 0
        for a,b in rel.fields.parent.sessions(grid):
            data = raw[a:b]
            t = np.arange(240+d%10,len(data),10)
            x = inputs(data[t]).reshape(len(t)*50,-1)
            s = np.arcsinh(data[:,:,F].astype(np.float32))
            r = data[:,:,0].astype(float)-.00136
            z = np.concatenate((s,r[...,None],(r*r).clip(0,70)[...,None]),axis=2)
            total = np.concatenate((np.zeros_like(z[:1]),z.cumsum(0)))
            target = []
            for left,right in zip(CUTS[:-1],CUTS[1:]):
                val = (total[t-left]-total[t-right])/(right-left)
                val[:,:,:FIELD_N] -= s[t]
                target.append(val)
            y = np.concatenate(target,axis=2).reshape(len(x),-1)
            gram += x.T @ np.column_stack((x,y))
            n += len(x)
        np.save(OUT/f"gram_{d:02d}.npy",gram/n)


def train(end):
    g = np.mean([np.load(OUT/f"gram_{d:02d}.npy") for d in range(end)],axis=0)
    cov = g[:,:len(g)]
    return np.linalg.solve(cov+.09*np.diag(cov.diagonal().clip(.003)),g[:,len(g):])


@torch.inference_mode()
def response(x,vol,t):
    model = book.encoder()
    risk = torch.from_numpy((.45+.55*vol).astype(np.float32).clip(.55,3)).cuda()
    gate = torch.from_numpy(np.exp(-t[:,None]/400).astype(np.float32)).cuda()
    return model(torch.from_numpy(x.astype(np.float16)).cuda().float(),risk,gate)[0].cpu().numpy()


def design(raw,grid,code,p,bank):
    out = np.zeros((*p.shape[:2],len(ARMS),3))
    data = raw[:min(len(raw),240)]
    t = np.arange(len(data))
    s = np.arcsinh(data[:,:,F].astype(np.float32))
    r = data[:,:,0].astype(float)-.00136
    summaries = (inputs(data[:1])[0]@bank).reshape(50,len(CUTS)-1,W)
    summaries[:,:,:FIELD_N] += s[0,:,None,:]
    summaries[:,:,-1] = summaries[:,:,-1].clip(.10,32)
    def past(h):
        overlap = np.clip(h-t[:,None]-1-CUTS[None,:-1],0,np.diff(CUTS))
        return np.einsum("tb,sbf->tsf",overlap,summaries)
    x,vol = book.phase.spline.features(data,t)
    quote = x.copy()
    for j,h in enumerate((5,24,120)):
        sl = slice(100+j*FIELD_N,100+(j+1)*FIELD_N)
        n = np.minimum(t+1,h)[:,None,None]
        avg = (s-x[:,:,sl].astype(float))*n
        change = s-(avg+past(h)[:,:,:FIELD_N])/h
        idx = t+1<h
        quote[idx,:,sl] = change[idx].clip(-10,10)
    price = x.copy()
    v = np.sqrt((8+book.phase.inn.moving_sum(r*r,96)+past(96)[:,:,-1])/(96+8)).clip(.45,4)
    v[t+1>=96] = vol[t+1>=96]
    base = 100+3*FIELD_N
    price[:,:,base] = np.log(v)
    for j,h in enumerate((5,20,60,240)):
        ret = (book.phase.inn.moving_sum(r,h)+past(h)[:,:,-2])/np.sqrt(h)
        price[:,:,base+1+2*j] = np.arcsinh(ret).clip(-10,10)
        price[:,:,base+2+2*j] = np.arcsinh((ret-ret.mean(1,keepdims=True))/v).clip(-10,10)
    both = quote.copy()
    both[:,:,base:] = price[:,:,base:]
    ref = response(x,vol,t)
    for j,(features,risk) in enumerate(((quote,vol),(price,v),(both,v))):
        delta = response(features,risk,t)-ref
        out[t,:,j] = delta-delta.mean(1,keepdims=True)
    return out


def predict(raw,grid,p,params):
    return book.path.corrected(p,design(raw,grid,params["code"],p,params["bank"]),params)


if __name__=="__main__":
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
