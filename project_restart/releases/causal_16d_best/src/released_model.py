"""Apply the frozen session, market, peer and event corrections to the base forecast."""
import numpy as np

import experiment_consensus_error_variance_v1 as consensus
import experiment_external_event_prior_v1 as best
import experiment_matured_expert_tail_v1 as tail
import experiment_matured_portfolio_posterior_v1 as posterior
import experiment_peer_implied_tail_v1 as peer
import experiment_replicated_expert_consensus_v1 as experts
import experiment_periodic_book_response_v1 as clock
from experiment_periodic_near_carry_v1 import carry
import experiment_y3_settled_rank_graph_v1 as calibration
import experiment_y3_persistent_backbone_exposure_v1 as exposure
import experiment_y3_realized_sort_rotation_v1 as sort_rotation
import experiment_joint_sort_horizon_transfer_v1 as medium_portfolios
import experiment_y3_peer_response_fingerprints_v1 as peer_response
import experiment_y3_settled_stock_gradient_v1 as settled
import experiment_y3_settled_error_attention_v1 as memory
import experiment_y3_ordered_book_area_v1 as ordered_area
import experiment_y3_unpriced_peer_pressure_v1 as peer_pressure
import experiment_y3_peer_shock_participation_v1 as participation
import experiment_y3_echo_vintage_ensemble_v1 as echoes
import experiment_y3_temporal_pair_alignment_v1 as alignment
import experiment_y3_early_settled_stock_slopes_v1 as stock_slopes
import experiment_y3_supported_morning_slopes_v1 as supported
import experiment_y3_directed_unexpected_prices_v1 as directed_prices
import experiment_y3_reversal_conditioned_memory_v1 as reversal
import experiment_y3_local_jump_variation_v1 as bipower
import experiment_y3_bipower_phase_pool_v1 as phase_pool
import experiment_y3_absolute_block_motion_v1 as motion
import experiment_y3_pressure_cohort_projection_v1 as cohort
import experiment_y3_forecast_trajectory_coupling_v1 as trajectory
import experiment_y3_incremental_magnitude_ratio_v1 as magnitude
import experiment_y3_expected_volatility_occupancy_v1 as occupancy
import experiment_y3_pooled_revision_feedback_v1 as feedback
import experiment_y2_matured_feedback_readout_v1 as medium_feedback
import experiment_y1_price_feedback_v1 as short_feedback
import experiment_y3_variance_ratio_magnitude_v1 as ratio_magnitude
import experiment_short_medium_variance_ratio_v1 as short_medium_ratio
import experiment_y3_prequential_disagreement_gate_v1 as market_gate
import experiment_y3_intraday_vintage_orientation_v1 as orientation
import experiment_y3_partial_maturity_orientation_v1 as partial_maturity
import experiment_y3_matured_revision_factor_transport_v1 as revision_transport
import experiment_y3_independent_peer_sensors_v1 as peer_sensors
import experiment_y1_source_specific_peer_response_v1 as fast_peer
import experiment_y3_peer_settled_medium_errors_v1 as peer_settlement
import experiment_y3_forecast_matchup_difficulty_v1 as matchup
import experiment_y2_whole_forecast_matchups_v1 as medium_matchup
import experiment_seasonal_pressure_surprises_v1 as seasonal
import replay_session_experts_16d as session


class SessionRefinement:
    def __init__(self, update_factory=sort_rotation.correction, *, include_medium=True, include_peer_response=True,
                 include_settled=True, include_attention=True, include_ordered_area=True, include_peer_pressure=True, include_participation=True,
                 include_echo_vintages=True, include_pair_alignment=True, include_early_stock_slopes=True, include_supported_morning=True, include_directed_prices=True, include_reversal=True, include_bipower=True, include_phase_pool=True, include_absolute_motion=True, include_pressure_cohort=True, include_peer_trajectory=True, include_magnitude_ratio=True, include_expected_occupancy=True, include_revision_feedback=True, include_medium_feedback=True, include_short_feedback=True, include_ratio_magnitude=True, include_short_medium_ratio=True, include_market_gate=True, include_vintage_orientation=True, include_partial_maturity=True, include_revision_transport=True, include_peer_sensors=True, include_fast_peer=True, include_peer_settlement=True, include_matchup=True, include_medium_matchup=True):
        self.session = session.additions()
        with np.load(peer.OUT / 'model.npz') as state:
            self.peers, self.readout = state['peers'], state['readout']
        self.experts = experts.sources()
        self.blend = np.load(session.ART / 'equal_expert_relative_tail_v1' / 'weights.npy')[0]
        self.online = np.load(posterior.OUT / 'weights.npy')
        with np.load(consensus.OUT / 'weights.npz') as state:
            self.consensus, self.covariance = state['w'], state['prior']
        with np.load(seasonal.OUT / 'model.npz') as state:
            self.profile = state['mean'], state['scale']
            self.flow = state['w']
        self.final = best.load()
        with np.load(clock.OUT / 'model.npz') as state:
            self.clock_weights = state['m'], state['r']
            self.phase_shift = int(state['shift'])
            self.mask = state['keep']
        self.calibrate = calibration.layer()
        self.exposure = exposure.correction_from_heads()
        self.short_medium_ratio = None
        self.market_gate = None
        self.orientation = None
        self.partial_maturity = None
        self.revision_transport = None
        self.peer_sensors = None
        self.fast_peer = None
        self.peer_settlement = None
        self.matchup = None
        self.medium_matchup = None
        self.ratio_magnitude = None
        self.short_feedback = None
        self.medium_feedback = None
        self.peer_response = None
        if include_peer_response:
            if not include_settled:
                self.peer_response = peer_response.correction()
            elif not include_attention:
                self.peer_response = settled.correction(settled.PRIMARY_OUT)
            elif not include_ordered_area:
                self.peer_response = memory.correction()
            elif not include_peer_pressure:
                self.peer_response = ordered_area.correction()
            elif not include_participation:
                self.peer_response = peer_pressure.correction()
            elif not include_echo_vintages:
                self.peer_response = participation.correction()
            elif not include_pair_alignment:
                self.peer_response = echoes.correction()
            elif not include_early_stock_slopes:
                self.peer_response = alignment.correction()
            elif not include_supported_morning:
                self.peer_response = stock_slopes.correction()
            elif not include_directed_prices:
                self.peer_response = supported.correction()
            elif not include_reversal:
                self.peer_response = directed_prices.correction()
            elif not include_bipower:
                self.peer_response = reversal.correction()
            elif not include_phase_pool:
                self.peer_response = bipower.correction()
            elif not include_absolute_motion:
                self.peer_response = phase_pool.correction()
            elif not include_pressure_cohort:
                self.peer_response = motion.correction()
            elif not include_peer_trajectory:
                self.peer_response = cohort.correction()
            elif not include_magnitude_ratio:
                self.peer_response = trajectory.correction()
            elif not include_expected_occupancy:
                self.peer_response = magnitude.correction()
            elif not include_revision_feedback:
                self.peer_response = occupancy.correction()
            else:
                self.peer_response = feedback.correction()
                if include_medium_feedback and include_medium:
                    self.medium_feedback = medium_feedback.load()
                    if include_short_feedback:
                        self.short_feedback = short_feedback.load()
                        if include_ratio_magnitude:
                            self.ratio_magnitude = ratio_magnitude.load_change()
                            if include_short_medium_ratio:
                                self.short_medium_ratio = short_medium_ratio.load_change()
                                if include_market_gate:
                                    self.market_gate = market_gate.load_change()
                                    if include_vintage_orientation:
                                        self.orientation = orientation.load_change()
                                        if include_partial_maturity:
                                            self.partial_maturity = partial_maturity.load_change()
                                            if include_revision_transport:
                                                self.revision_transport = revision_transport.load_change()
                                                if include_peer_sensors:
                                                    self.peer_sensors = peer_sensors.load_change()
                                                    if include_fast_peer:
                                                        self.fast_peer = fast_peer.change()
                                                        if include_peer_settlement:
                                                            self.peer_settlement = peer_settlement.load_change()
                                                            if include_matchup:
                                                                self.matchup = matchup.load_change()
                                                                if include_medium_matchup:
                                                                    self.medium_matchup = medium_matchup.load_change()
        self.y3_update = update_factory()
        self.y2_update = medium_portfolios.correction() if include_medium else None

    def predict_before_portfolios(self, raw, base, opening):
        p = self.session(raw, base, opening)
        q = p.copy()
        q[:, :, 2] += sum(peer.changes(raw, p, self.peers, self.readout))
        forecasts = [model(raw) for model in self.experts]
        fixed = tail.fixed_forecast(q, forecasts, self.blend)
        online = posterior.advance(fixed, posterior.components(raw, fixed, forecasts), self.online)
        calibrated = (online + consensus.features(forecasts, self.covariance) @ self.consensus).astype('float32')
        parent = seasonal.apply(calibrated, seasonal.features(raw, calibrated, self.profile, True), self.flow)
        fixed = self.final(raw, parent)
        terms = clock.changes(raw, np.arange(len(raw)), self.phase_shift, self.clock_weights)
        p = self.calibrate(raw, carry(fixed, terms, self.mask))
        p[:, :, 2] += self.exposure(opening, [forecasts[i] for i in exposure.EXPERT_IDS])
        return p

    def predict(self, raw, base, opening):
        p = self.predict_before_portfolios(raw, base, opening)
        return self.finish_portfolios(raw, p)

    def finish_portfolios(self, raw, p):
        medium = self.y2_update(raw, p) if self.y2_update is not None else None
        p[:, :, 2] += self.y3_update(raw, p)
        if medium is not None:
            p[:, :, 1] += medium
        if self.peer_response is not None:
            p[:, :, 2] += self.peer_response(raw, p)
        if self.ratio_magnitude is not None:
            p[:, :, 2] += self.ratio_magnitude(raw, p)
        if self.market_gate is not None:
            p[:, :, 2] += self.market_gate(raw, p)
        if self.orientation is not None:
            p[:, :, 2] += self.orientation(raw, p)
        if self.partial_maturity is not None:
            p[:, :, 2] += self.partial_maturity(raw, p)
        if self.revision_transport is not None:
            p[:, :, 2] += self.revision_transport(raw, p)
        if self.peer_sensors is not None:
            p[:, :, 2] += self.peer_sensors(raw, p)
        if self.peer_settlement is not None:
            p[:, :, 2] += self.peer_settlement(raw, p)
        if self.matchup is not None:
            p[:, :, 2] += self.matchup(raw, p)
        issued = p.copy() if self.medium_matchup is not None else None
        extra_fast = self.fast_peer(raw, p) if self.fast_peer is not None else None
        if self.medium_feedback is not None:
            p[:, :, 1] += self.medium_feedback(raw, p)
        if self.short_feedback is not None:
            p[:, :, 0] += self.short_feedback(raw, p)
        if self.short_medium_ratio is not None:
            p[:, :, :2] += self.short_medium_ratio(raw, p)
        if extra_fast is not None:
            p[:, :, 0] += extra_fast
        if self.medium_matchup is not None:
            p[:, :, 1] += self.medium_matchup(issued, p)
        return p
