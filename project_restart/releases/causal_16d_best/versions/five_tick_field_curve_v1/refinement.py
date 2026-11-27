"""Apply the fitted readouts to a continuous prefix of a single trading day."""

import numpy as np
import short, stock, market, precision, nonlinear, clock, memory, news, history_head, fine


class FieldRefinement:
    def __init__(self):
        self.weights = {}
        for module in (
            short,
            stock,
            market,
            precision,
            nonlinear,
            clock,
            memory,
            news,
            history_head,
            fine,
        ):
            with np.load(module.OUT / "weights.npz") as file:
                self.weights[module] = dict(file)

    def predict(self, raw, original):
        grid = np.arange(len(raw))
        sw, lw, mw = (self.weights[module] for module in (short, stock, market))
        dx = short.apply(raw, grid, original, sw["code"], {"market": sw["market"]})[
            "market"
        ]
        state, av = stock.state(raw, grid, lw["code"], with_market=True)
        m = market.market_state(av, grid)
        signal = m @ mw["bank"]
        dm = np.column_stack((original.mean(1, dtype=float), signal)) @ mw["w"]
        relative = original - original.mean(1, keepdims=True)
        q = original + dx
        q[:, :, 2] += np.concatenate((relative, state @ lw["bank"]), 2) @ lw["w"]
        q[:, :, 2] += dm[:, None]
        controls = [dm, signal[:, 2]]
        for module in (precision, nonlinear, clock, memory, news, history_head, fine):
            op = self.weights[module]
            means = q.mean(1, dtype=float)
            # These controls were fitted independently of the short-horizon readout.
            means[:, 0] = original[:, :, 0].mean(1, dtype=float)
            if module is nonlinear:
                z = nonlinear.expansion(m @ op["mp"]) @ op["bank"]
            elif module in (precision, clock):
                z = m @ op["bank"]
            elif module is news:
                z = news.views(news.prepare_inputs(raw, grid, m, op), grid, op)
            else:
                z = module.views(m, grid, op)
            delta = (np.column_stack((means, *controls, z)) @ op["w"]).astype(q.dtype)
            q += delta[:, None]
            if module in (clock, memory, news, history_head):
                controls.append(delta[:, 2])
        return q
