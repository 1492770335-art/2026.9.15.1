"""Rebuild the trained tree-graph improvement from raw order-book observations."""
import numpy as np
import experiment_pairwise_book_tree_v1 as tree
import replay_phase_future_book_v1 as replay


def main(refinement=None):
    root=tree.OUT
    models=[tree.lgb.Booster(model_file=str(root/f'tree_{j}.txt')) for j in range(3)]
    with np.load(root/'graph.npz') as geo:
        graph=tuple(geo[k].copy() for k in ('left','right','inverse'))
    coef=np.load(root/'coef.npy')
    output, refine = (root, None) if refinement is None else refinement
    def predict(raw,p):
        side=tree.predict(raw,models,graph)
        p = (p+tree.innovation(p,side)@coef).astype('float32')
        return p if refine is None else refine(raw,p)
    replay.main(factor_mode=True,postprocess=(output,predict))


def path_areas():
    import experiment_ordered_book_path_areas_v1 as areas
    with np.load(areas.OUT/'model.npz') as st:
        weights = (st['market'].copy(), st['relative'].copy())
    def refine(raw,p):
        a,b = areas.predict(areas.Path(raw),p,weights)
        return (p+a+b).astype('float32')
    return areas.OUT,refine


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--path-areas',action='store_true')
    args=parser.parse_args()
    main(path_areas() if args.path_areas else None)
