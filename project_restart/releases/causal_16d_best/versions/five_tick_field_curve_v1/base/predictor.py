"""Frozen sixteen-day model. A call consumes one chronological session, with all 50 stocks in code order."""

import json
import base_updates
import numpy as np
import torch
import replay_curve_quantile_market_v1 as base
import replay_intraday_relative_calibration_v1 as intraday
import experiment_phase_future_book_v1 as phase
import experiment_phase_conditional_blend_v1 as conditional
import experiment_orthogonal_expert_innovations_v1 as orthogonal
import experiment_cross_horizon_rank_readout_v1 as ranks
import experiment_symmetric_pressure_network_v1 as sym
import experiment_pairwise_spread_decoder_v1 as spread
import experiment_shared_bar_gru_v1 as bar
import experiment_dynamic_factor_return_net_v1 as factor
import experiment_pairwise_book_tree_v1 as tree
import experiment_field_tokens_skew_means_v1 as tokens
import experiment_categorical_suffix_responses_v1 as suffix
import experiment_conditional_tail_payouts_v1 as tail
import experiment_supervised_gradient_kernel_v1 as kernel
import experiment_rolling_drift_strength_v1 as drift
import experiment_propagated_return_kernel_v1 as propagated
import experiment_individual_kernel_forecast_v1 as individual
import experiment_quote_lattice_replenishment_v1 as lattice
import experiment_execution_volume_clock_v1 as volume
import experiment_price_level_execution_memory_v1 as memory
import experiment_bid_ask_side_network_v1 as side
import experiment_execution_location_markouts_v1 as markout
import experiment_persistent_execution_state_v1 as persistent
import experiment_reconstructed_depth_flow_v1 as depth
import experiment_damped_market_state_v1 as damped
import experiment_price_band_depth_curve_v1 as bands
import experiment_depth_inventory_transition_v1 as inventory
import experiment_side_network_execution_fusion_v1 as fusion
import experiment_joint_horizon_completion_v1 as joint
import experiment_legacy_horizon_completion_v1 as legacy
import experiment_partial_horizon_curve_v1 as partial
from experiment_causal_cross_horizon_ridge_v1 import ART


def load_net(module, constructor, dev):
    state = torch.load(module.OUT / "net_16.pth", map_location="cpu", weights_only=True)
    model = constructor(state)
    model.load_state_dict(state)
    return model.to(dev).eval()


def neural_update(module, constructor):
    net = load_net(module, constructor, "cpu")
    coeff = np.load(module.OUT / "coef.npy")
    return lambda raw, p: fusion.add(p, module.infer(net, raw), coeff)


class Predictor:
    def __init__(self):
        torch.set_num_threads(2)
        opening_dir = ART / "opening_expert_v1/commit_opening_expert"
        self.opening = base.opening.Evaluator("", str(opening_dir), "cuda:0")
        self.opening_mix = np.array(
            json.loads((opening_dir / "conf/Model_0.json").read_text())["opening_mix"]
        )
        self.phase = load_net(
            phase, lambda s: phase.BookNet(s["current.0.weight"].shape[1]), "cuda"
        )
        self.sym = load_net(
            sym, lambda s: sym.PressureNet(s["gate.0.weight"].shape[1]), "cuda"
        )
        self.spread = load_net(
            spread, lambda s: spread.PairNet(s["state.0.weight"].shape[1]), "cuda"
        )
        self.bar = load_net(bar, lambda s: bar.ScaleNet(), "cuda")
        self.factor = load_net(
            factor, lambda s: factor.FactorNet(s["current.0.weight"].shape[1]), "cuda"
        )
        self.phase_weight = np.array(
            json.loads((phase.OUT / "results.json").read_text())["20"]["w"]
        )
        self.conditional = np.load(conditional.OUT / "coef.npy")
        self.orthogonal = np.load(orthogonal.OUT / "coef.npy")
        self.rank_coef = np.load(ranks.OUT / "coef.npy")
        self.factor_coef = np.load(factor.OUT / "coef.npy")
        self.trees = [
            tree.lgb.Booster(model_file=str(tree.OUT / f"tree_{j}.txt"))
            for j in range(3)
        ]
        with np.load(tree.OUT / "graph.npz") as g:
            self.graph = tuple(g[k] for k in ("left", "right", "inverse"))
        self.tree_coef = np.load(tree.OUT / "coef.npy")
        self.early = [
            (m.OUT.name, fn()[1])
            for m, fn in (
                (tokens, tokens.refinement),
                (suffix, base_updates.suffix_update),
                (tail, base_updates.tail_update),
                (kernel, base_updates.kernel_update),
                (drift, base_updates.drift_update),
                (propagated, base_updates.propagated_update),
            )
        ]
        self.late = [
            (individual.OUT.name, individual.frozen()),
            (lattice.OUT.name, lattice.frozen()),
            (volume.OUT.name, volume.frozen()),
            (memory.OUT.name, volume.frozen(memory.OUT, memory.Day)),
            (
                fusion.OUT.name,
                neural_update(
                    side, lambda s: side.SideNet(s["encoder.0.weight"].shape[1])
                ),
            ),
            (markout.OUT.name, volume.frozen(markout.OUT, markout.Day)),
            (persistent.OUT.name, volume.frozen(persistent.OUT, persistent.Day)),
            (depth.OUT.name, volume.frozen(depth.OUT, depth.Day)),
            ("damped_depth_fusion_v1", neural_update(damped, damped.restore)),
            (bands.OUT.name, volume.frozen(bands.OUT, bands.Day)),
            ("inventory_band_fusion_v1", neural_update(inventory, inventory.restore)),
        ]
        self.early_completion = np.load(legacy.RAW / "completion.npy")
        self.late_completion = np.load(joint.OUT / "completion.npy")
        with np.load(partial.OUT / "model.npz") as st:
            self.partial = [st["market"], st["relative"]]

    @torch.inference_mode()
    def predict(self, features):
        raw = np.array(features, dtype=np.float16, order="C", copy=True)
        if raw.ndim != 3 or raw.shape[1:] != (50, 100) or not len(raw):
            raise ValueError("expected (time, 50, 100)")
        (_, opening), _ = base.opening.replay_features(
            self.opening, raw, self.opening_mix, 2
        )
        records = {}
        # The upstream training implementation exposes the two intermediate forecasts through a callback.
        previous_observer = base.stage_observer
        try:
            base.stage_observer = records.__setitem__
            p = intraday.predict(raw, opening.reshape(-1, 50, 3)).astype(np.float32)
        finally:
            base.stage_observer = previous_observer
        n = phase.infer(self.phase, raw)
        q = phase.v1.blend(p, n, self.phase_weight)
        q = q.astype(np.float32) + sum(
            z * c for z, c in zip(conditional.states(raw, p, n), self.conditional.T)
        )
        q = orthogonal.apply(q.astype(np.float32), n, p, self.orthogonal)
        sources = [
            n,
            phase.infer(self.sym, raw, make_features=sym.features),
            phase.infer(self.spread, raw),
            bar.predict(self.bar, raw),
        ]
        q = (q + ranks.features(q, sources) @ self.rank_coef).astype(np.float32)
        records[ranks.OUT.name] = q
        m, r = factor.readout.basis(q, phase.infer(self.factor, raw))
        q = (q + (m @ self.factor_coef[0])[:, None] + r @ self.factor_coef[1]).astype(
            np.float32
        )
        records[factor.OUT.name] = q
        n = tree.predict(raw, self.trees, self.graph)
        q = (q + tree.innovation(q, n) @ self.tree_coef).astype(np.float32)
        records[tree.OUT.name] = q
        for name, layer in self.early + self.late:
            q = layer(raw, q)
            if name in legacy.NAMES or name in joint.ROOTS:
                records[name] = q
        first = records[propagated.OUT.name]
        legacy_delta = (
            joint.apply(first, records, self.early_completion, legacy.STAGES) - first
        )
        combined = (
            joint.apply(q, records, self.late_completion) + legacy_delta
        ).astype(np.float32)
        result = partial.add(combined, partial.features(raw, combined), self.partial)[0]
        if not np.isfinite(result).all():
            raise ValueError("nonfinite predicted return")
        return result
