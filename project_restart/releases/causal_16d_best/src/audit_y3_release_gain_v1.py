"""Measure the cumulative score and paired block uncertainty against the published phase-conditioned release."""
import json

import numpy as np
import torch

import experiment_phase_conditioned_forecasts_v1 as published
import experiment_y3_residual_peer_curve_v1 as metrics
import experiment_y3_trade_spell_survival_v1 as candidate
import replay_periodic_parent_v1 as replay

WEIGHTS = np.array([.2, .3, .5])


def run(candidate=candidate, read=None, reference=published, read_baseline=None):
    baseline_report = json.loads((reference.OUT/'reload_verification.json').read_text())
    final_report = json.loads((candidate.OUT/'reload_verification.json').read_text())
    if read is None:
        _, read = candidate.history.replay_with((candidate,))
    first = reference.layer() if read_baseline is None else None
    blocks, daily = [], {}
    for d in range(16, 20):
        raw = replay.original.features(replay.original.DATA, d)
        if read_baseline is None:
            with np.load(replay.CACHE/f'day_{d}.npz') as f:
                assert str(f['raw_sha256'])==replay.digest(raw)
                old = first(raw, f['full'])
        else:
            old, _, _ = read_baseline(d,raw)
        new, _, _ = read(d, raw)
        y = replay.labels(d).reshape(old.shape)
        stat = replay.measure([(old.reshape(-1, 3), new.reshape(-1, 3), y.reshape(-1, 3))])
        assert abs(stat['baseline_score']-baseline_report['daily'][str(d+1)]['score'])<1e-9
        assert abs(stat['score']-final_report['daily'][str(d+1)]['score'])<1e-9
        daily[d+1] = stat['gain']
        record = []
        for j in range(3):
            # Reuse paired correlation sufficient statistics by putting each target in the measured slot.
            p, target = old.copy(), y.copy()
            p[:, :, 2], target[:, :, 2] = old[:, :, j], y[:, :, j]
            change = new[:, :, j].astype(float)-old[:, :, j]
            record.append(metrics.blockstats(p, target, [change]))
        blocks.append(np.stack(record, axis=1))
    def gain(stats):
        return float(metrics.pooled(stats)[:, 0]@WEIGHTS)
    actual = gain(sum(b.sum(0) for b in blocks))
    assert abs(actual-(final_report['pooled']['score']-baseline_report['pooled']['score']))<1e-10
    rng = np.random.default_rng(3819)
    dist = []
    for _ in range(1600):
        sampled = []
        for d in rng.integers(4, size=4):
            b = blocks[d]
            sampled.append(b[rng.integers(len(b), size=len(b))].sum(0))
        dist.append(gain(sum(sampled)))
    r = dict(reference=reference.OUT.name, candidate=candidate.OUT.name, rows=final_report['pooled']['rows'],
             score=final_report['pooled']['score'], reference_score=baseline_report['pooled']['score'],
             gain=actual, daily=daily, paired_gain_percentiles=np.quantile(dist, [.025,.5,.975]).tolist(),
             block_ticks=120, samples=1600)
    filename = 'published_reference_gain.json' if reference is published else 'incremental_gain.json'
    (candidate.OUT/filename).write_text(json.dumps(r, indent=2))
    print(r, flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    import sys
    if sys.argv[1:] == ['peer-depth']:
        import y3_peer_depth_references as depth
        run(depth.incumbent, depth.replay_with()[1])
    elif sys.argv[1:] == ['peer-news']:
        import y3_peer_terminal_news_references as news
        run(news.incumbent, news.replay_with()[1])
    elif sys.argv[1:] == ['order-price']:
        import y3_order_price_rollout_references as rollout
        run(rollout.incumbent, rollout.replay_with()[1])
    elif sys.argv[1:] == ['persistent-orders']:
        import y3_persistent_order_references as modes
        run(modes.incumbent, modes.replay_with()[1])
    elif sys.argv[1:] == ['receiver-portfolios']:
        import y3_receiver_portfolio_references as receivers
        run(receivers.incumbent, receivers.replay_with()[1])
    elif sys.argv[1:] == ['recipient-exposures']:
        import y3_recipient_exposure_references as exposure
        run(exposure.incumbent, exposure.replay_with()[1])
    elif sys.argv[1:] == ['clock-depth']:
        import y3_clock_and_depth_references as combined
        run(combined.incumbent, combined.replay_with()[1])
    elif sys.argv[1:] in (['medium-order-clocks'], ['medium-order-clocks-increment']):
        import y3_medium_order_clock_references as medium
        if sys.argv[1]=='medium-order-clocks':
            run(medium.incumbent,medium.replay_with()[1])
        else:
            before=medium.previous.previous.previous
            run(medium.incumbent,medium.replay_with()[1],before.incumbent,before.replay_with()[1])
    elif sys.argv[1:] in (['joint-order-clocks'], ['joint-order-clocks-increment']):
        import joint_order_clock_references as clocks
        if sys.argv[1]=='joint-order-clocks':
            run(clocks.incumbent,clocks.replay_with()[1])
        else:
            before=clocks.previous.previous
            run(clocks.incumbent,clocks.replay_with()[1],before.incumbent,before.replay_with()[1])
    elif sys.argv[1:] in (['early-book'], ['early-book-increment']):
        import joint_early_book_references as early
        if sys.argv[1] == 'early-book':
            run(early.incumbent,early.replay_with()[1])
        else:
            run(early.incumbent,early.replay_with()[1],early.previous.incumbent,early.previous.replay_with()[1])
    elif sys.argv[1:] == ['joint-stage-readout']:
        import joint_horizon_stage_references as stages
        run(stages.incumbent, stages.replay_with()[1])
    elif sys.argv[1:] == ['impact-mass']:
        import y3_impact_mass_references as impact
        run(impact.incumbent, impact.replay_with()[1])
    elif sys.argv[1:] == ['crossed-levels']:
        import y3_crossed_level_references as levels
        run(levels.incumbent, levels.replay_with()[1])
    else:
        run()
