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
import experiment_phase_conditioned_forecasts_v1 as calibration
import experiment_seasonal_pressure_surprises_v1 as seasonal
import replay_session_experts_16d as session


class SessionRefinement:
    def __init__(self):
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

    def predict(self, raw, base, opening):
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
        return self.calibrate(raw, carry(fixed, terms, self.mask))
