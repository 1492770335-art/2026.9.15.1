"""Reconstruct the phase-conditioned network ensemble on raw inputs using the weights selected before day 17."""
import json
import numpy as np
import torch
import experiment_phase_future_book_v1 as phase
import replay_intraday_relative_calibration_v1 as previous

def main(conditional=False, orthogonal=False, rank_readout=False, adversarial=False, factor_mode=False, postprocess=None):
    rank_readout = rank_readout or adversarial or factor_mode
    orthogonal = orthogonal or rank_readout
    conditional = conditional or orthogonal
    state=torch.load(phase.OUT/'net_16.pth',weights_only=True,map_location='cpu')
    net=phase.BookNet(state['current.0.weight'].shape[1]).cuda().eval()
    net.load_state_dict(state)
    rows=[]
    for d in range(12,16):
        p=np.load(phase.ref.OUT/f'pred_{d}.npy')
        rows.append((p,phase.inn.prior.labels(d).reshape(p.shape),np.load(phase.OUT/f'side_{d}.npy')))
    _,weights=phase.fits.coefficients(rows)
    del rows
    report=json.loads((phase.OUT/'results.json').read_text())['20']
    np.testing.assert_array_equal(weights,report['w'])
    out=phase.OUT
    expected=report['result']['score']
    if conditional:
        import experiment_phase_conditional_blend_v1 as dynamic
        out=dynamic.OUT
        expected=json.loads((out/'results.json').read_text())['20']['score']
        coef=np.load(out/'coef.npy')
    if orthogonal:
        import experiment_orthogonal_expert_innovations_v1 as adjustment
        out=adjustment.OUT
        expected=json.loads((out/'results.json').read_text())['20']['score']
        mode_coef=np.load(out/'coef.npy')
    if rank_readout:
        import experiment_cross_horizon_rank_readout_v1 as rank
        import experiment_symmetric_pressure_network_v1 as sym
        import experiment_pairwise_spread_decoder_v1 as pair
        import experiment_shared_bar_gru_v1 as bar
        st=torch.load(sym.OUT/'net_16.pth',weights_only=True,map_location='cpu')
        sn=sym.PressureNet(st['gate.0.weight'].shape[1]).cuda().eval()
        sn.load_state_dict(st)
        st=torch.load(pair.OUT/'net_16.pth',weights_only=True,map_location='cpu')
        pn=pair.PairNet(st['state.0.weight'].shape[1]).cuda().eval()
        pn.load_state_dict(st)
        bn=bar.ScaleNet().cuda().eval()
        bn.load_state_dict(torch.load(bar.OUT/'net_16.pth',weights_only=True,map_location='cpu'))
        out=rank.OUT
        rank_coef=np.load(out/'coef.npy')
        expected=json.loads((out/'results.json').read_text())['20']['score']
    if adversarial:
        import experiment_date_adversarial_book_v1 as domain
        out=domain.OUT
        domain_coef=np.load(out/'coef.npy')
        ds=torch.load(out/'net_16.pth',map_location='cpu',weights_only=True)
        dn=phase.BookNet(ds['current.0.weight'].shape[1]).cuda().eval()
        dn.load_state_dict(ds)
        expected=json.loads((out/'results.json').read_text())['20']['score']
    if factor_mode:
        import experiment_dynamic_factor_return_net_v1 as factors
        out = factors.OUT
        factor_coef = np.load(out / 'coef.npy')
        state = torch.load(out / 'net_16.pth', map_location='cpu', weights_only=True)
        factor_net = factors.FactorNet(state['current.0.weight'].shape[1]).cuda().eval()
        factor_net.load_state_dict(state)
        expected = json.loads((out / 'results.json').read_text())['20']['score']
    if postprocess is not None:
        out, refine = postprocess
        expected = json.loads((out / 'results.json').read_text())['20']['score']
    def predict(raw,opening):
        p=previous.predict(raw,opening).astype(np.float32)
        n=phase.infer(net,raw)
        prediction=phase.v1.blend(p,n,weights)
        if conditional:
            terms=dynamic.states(raw,p,n)
            prediction=prediction.astype(np.float32)+sum(z*c for z,c in zip(terms,coef.T))
        if orthogonal:
            prediction=adjustment.apply(prediction.astype(np.float32),n,p,mode_coef)
        if rank_readout:
            sources=[n,phase.infer(sn,raw,make_features=sym.features),phase.infer(pn,raw),bar.predict(bn,raw)]
            prediction=(prediction+rank.features(prediction,sources)@rank_coef).astype(np.float32)
        if adversarial:
            ms,rs=domain.readout.basis(prediction,phase.infer(dn,raw))
            prediction=(prediction+((ms@domain_coef[0])[:,None]+rs@domain_coef[1])).astype(np.float32)
        if factor_mode:
            previous.replay.observe('cross_horizon_rank_readout_v1',prediction)
            market, relative = factors.readout.basis(prediction, phase.infer(factor_net, raw))
            prediction = (prediction + (market @ factor_coef[0])[:, None] + relative @ factor_coef[1]).astype(np.float32)
        if factor_mode:previous.replay.observe('dynamic_factor_return_net_v1',prediction)
        if postprocess is not None:
            prediction = refine(raw, prediction)
        return prediction
    previous.replay.main(predict,out,expected,deep_prefix=postprocess is not None)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--conditional',action='store_true')
    parser.add_argument('--orthogonal',action='store_true')
    parser.add_argument('--rank-readout',action='store_true')
    branch = parser.add_mutually_exclusive_group()
    branch.add_argument('--factor-mode', action='store_true')
    branch.add_argument('--adversarial',action='store_true')
    args=parser.parse_args()
    main(args.conditional,args.orthogonal,args.rank_readout,args.adversarial,args.factor_mode)
