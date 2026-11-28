"""Convert pairwise ordering into expected residual payoff with separate upside and downside sizes."""
import numpy as np
import torch
from torch.nn import functional as F
import experiment_dense_matched_ordinal_pairs_v1 as pairs

REFERENCE=pairs.REFERENCE
OUT=pairs.OUT.parent/"dense_pair_payoff_sizes_v1"
ARMS=("size_weighted_order","asymmetric_payoff","future_value_residual")
USES_FORECAST=True


def sizes(effect,context):
    common=.08+F.softplus(context[...,0])
    skew=.85*torch.tanh(effect[...,1])
    return common*(1+skew),common*(1-skew)


def train(days, model_class=pairs.Comparison, artifact_dir=OUT):
    torch.manual_seed(33091)
    records=[]
    for d in range(4,days):
        with np.load(pairs.net.OUT/f"day_{d:02d}.npz") as f:
            records.append((f["x"],f["y"]))
    xx,yy=map(np.concatenate,zip(*records))
    x=torch.tensor(xx,device="cuda",dtype=torch.float32)
    y=torch.tensor(yy,device="cuda")
    unit=x.square().mean((0,1)).sqrt().clamp_min(.04)
    x=(x/unit).clamp(-5,5)
    sy=y[:,:,0].square().mean().sqrt()
    y=y/sy
    model=model_class(x.shape[-1]).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.0011,weight_decay=.045)
    for epoch in range(5):
        for idx in torch.randperm(len(x),device="cuda").split(32):
            effect,context,peers,w=model(x[idx])
            target=y[idx]
            reference=target[torch.arange(len(idx),device="cuda")[:,None,None],peers]
            gap,teacher=(target[:,:,None,:]-reference).unbind(-1)
            win,lose=sizes(effect,context)
            observed=torch.where(gap>0,win,lose)
            # With fixed sign the Gamma score targets the conditional mean absolute payoff.
            size_loss=observed.log()+gap.abs().clamp(max=8)/observed
            sign_loss=F.binary_cross_entropy_with_logits(effect[...,0],(gap>0).float(),reduction="none")
            teacher_loss=F.smooth_l1_loss(effect[...,2],teacher,reduction="none",beta=.6)
            loss=(w*(sign_loss+.50*size_loss+.25*teacher_loss)).sum(-1).mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
    torch.save(dict(params=model.cpu().state_dict(),unit=unit.cpu()),artifact_dir/f"net_{days}.pt")
    return np.array([days])


@torch.inference_mode()
def views(raw,grid,code,bank,p):
    data=torch.load(OUT/f"net_{int(bank[0])}.pt",map_location="cpu",weights_only=True)
    model=pairs.Comparison(len(data["unit"])).double().eval()
    model.load_state_dict(data["params"])
    x=pairs.net.features(raw,grid,code,p)
    results=[]
    for j in range(0,len(x),32):
        u=(torch.from_numpy(x[j:j+32]).double()/data["unit"]).clamp(-5,5)
        effect,c,peers,w=model(u)
        prob=torch.sigmoid(effect[...,0])
        up,down=sizes(effect,c)
        signed=(prob-.5)*(up+down)
        payoff=signed+.5*(up-down)
        zs=[]
        empty=torch.zeros((len(u),50,50),dtype=u.dtype)
        edges=empty.scatter_add(-1,peers,w)
        adj=(edges+edges.transpose(1,2))*.5
        lap=torch.diag_embed(adj.sum(-1)+.40)-adj
        for preds in (signed,payoff,effect[...,2]):
            direct=empty.scatter_add(-1,peers,w*preds)
            zs.append((direct-direct.transpose(1,2)).sum(-1)*.5)
        results.append(torch.linalg.solve(lap,torch.stack(zs,-1)).numpy())
    z=np.concatenate(results)
    return z-z.mean(1,keepdims=True)


if __name__=="__main__":
    torch.set_num_threads(1)
    pairs.net.rel.main(OUT,ARMS,train,views,with_prediction=True,rolling_reference=REFERENCE)
