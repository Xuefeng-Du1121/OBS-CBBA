from collections import deque
import numpy as np
from .calibration import normalize


class RegionalCosts:
    def __init__(self, cfg, variant="lcb-ew"):
        self.cfg, self.variant = cfg, variant
        g = cfg.grid ** 2
        self.mean = np.full(g, cfg.prior_cost, dtype=np.float64)
        self.count = np.zeros(g, dtype=np.int64)
        self.recent = [[deque(maxlen=cfg.history) for _ in range(g)] for _ in range(cfg.uavs)]
        self.component_recent = [[deque() for _ in range(g)] for _ in range(cfg.uavs)]

    def degradation(self, uav, time):
        if self.cfg.regional_cost_mode == 'fixed-reference':
            return np.array([np.mean(v) if v else 0. for v in self.recent[uav]])
        values = []
        for queue in self.component_recent[uav]:
            # Querying is read-only; no future feedback or stale observations.
            rows = [x for t,x in queue if time-self.cfg.history < t <= time]
            values.append(np.clip(np.mean(rows, axis=0).sum()/self.cfg.regional_calibration['sigma_ref'], -1, 1) if rows else 0.)
        return np.asarray(values)

    def region(self, xy):
        cell = np.minimum((np.asarray(xy) * self.cfg.grid).astype(int), self.cfg.grid - 1)
        cell = np.maximum(cell, 0)
        return cell[..., 1] * self.cfg.grid + cell[..., 0]

    def index(self, uav, time):
        if self.variant == "fixed":
            return np.zeros_like(self.mean)
        degradation = self.degradation(uav, time)
        # New protocol follows log(t+1); retain legacy trajectories by default.
        offset = 1. if self.cfg.regional_cost_mode == 'percentile' else 2.
        bonus = self.cfg.beta * np.sqrt(np.log(time + offset) / (self.count + 1.))
        sign = 1. if self.variant.startswith("ucb") else -1.
        return self.mean + degradation + sign * bonus

    def observe(self, uav, region, cost, components=None, time=None):
        if self.cfg.regional_cost_mode == 'percentile':
            if components is None or time is None:
                raise ValueError('Timestamped raw feedback required for percentile costs')
            normalized = normalize(components, self.cfg.regional_calibration)
            queue = self.component_recent[uav][region]
            if queue and time < queue[-1][0]:
                raise ValueError('Feedback must be chronological')
            while queue and queue[0][0] <= time-self.cfg.history:
                queue.popleft()
            queue.append((time, normalized))
            weights = np.array([self.cfg.reward_length, self.cfg.reward_deviation, self.cfg.reward_failure])
            cost = float(weights @ normalized)
        cost = float(np.clip(cost, 0, 1))
        self.recent[uav][region].append(cost)
        if self.variant == "fixed":
            return cost
        self.count[region] += 1
        rate = 1. / self.count[region] if self.variant.endswith("cumulative") else self.cfg.rho
        self.mean[region] += rate * (cost - self.mean[region])
        return cost
