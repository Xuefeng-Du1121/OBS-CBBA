"""Opt-in synthetic maritime forcing, in SI units. Not an ocean reanalysis.

Only the generator sees these parameters. The observation adapter deliberately
removes them. Legacy Scenario.generate follows its original numerical path.
"""
from dataclasses import dataclass
import math
import numpy as np
from .arrivals import release_schedule, deadlines


@dataclass(frozen=True)
class MaritimeParameters:
    area_m: float = 10000.
    dt_seconds: float = 25.
    current_mps: tuple = (.3, 0.)
    shear_mps: float = .1
    current_modulation: float = .1
    current_period_seconds: float = 7200.
    windage: float = .02
    # Vectors point TOWARD the displacement direction, not meteorological FROM.
    # Entries: (first step, eastward m/s, northward m/s).
    wind_stages: tuple = ((0, 5., -5.), (25, -5., 5.), (50, 5., 0.), (75, 0., 0.))
    residual_mps: float = .05  # per-axis independent tick-average velocity SD
    anchor_radius_m: float = 100.

    def validate(self):
        scalars = (self.area_m, self.dt_seconds, self.shear_mps, self.current_modulation,
                   self.current_period_seconds, self.windage, self.residual_mps, self.anchor_radius_m)
        if not all(math.isfinite(x) for x in scalars):
            raise ValueError('Nonfinite maritime parameter')
        if (self.area_m <= 0 or self.dt_seconds <= 0 or self.current_period_seconds <= 0
                or self.residual_mps < 0 or not 0 <= self.windage <= 1
                or not 0 <= self.current_modulation <= 1
                or not 0 < self.anchor_radius_m < self.area_m / 2):
            raise ValueError('Invalid physical scale or forcing parameter')
        if len(self.current_mps) != 2 or not all(math.isfinite(x) for x in self.current_mps):
            raise ValueError('Current requires two finite velocity components')
        previous = -1
        for row in self.wind_stages:
            if (len(row) != 3 or type(row[0]) is not int or row[0] <= previous
                    or not all(math.isfinite(x) for x in row[1:])):
                raise ValueError('Wind stages require increasing integer times and finite vectors')
            previous = row[0]
        if not self.wind_stages or self.wind_stages[0][0] != 0:
            raise ValueError('Wind schedule must start at zero')
        return self

    @classmethod
    def from_config(cls, cfg):
        if not cfg.mobility.startswith('maritime-') or not isinstance(cfg.maritime, dict):
            raise ValueError('Explicit maritime parameter dictionary required')
        if cfg.nominal_speed != 0 or cfg.sigma != 0:
            raise ValueError('Set legacy nominal_speed and sigma to zero; use SI maritime residual_mps')
        return cls(**cfg.maritime).validate()

    def units(self, cfg):
        return dict(area_m=self.area_m, dt_seconds=self.dt_seconds,
                    uav_speed_mps=cfg.speed * self.area_m / self.dt_seconds,
                    residual_axis_m_per_step=self.residual_mps * self.dt_seconds,
                    residual_rms_m_per_step=math.sqrt(2) * self.residual_mps * self.dt_seconds)


def current_velocity(points, step, p):
    """Smooth shear plus a slowly modulated uniform current, in m/s."""
    phase = 2 * np.pi * step * p.dt_seconds / p.current_period_seconds
    value = np.broadcast_to(np.asarray(p.current_mps, dtype=float), np.shape(points)).copy()
    value[:, 0] += p.shear_mps * (2 * np.asarray(points)[:, 1] - 1)
    return value * (1 + p.current_modulation * np.sin(phase))


def wind_velocity(step, p):
    row = next(row for row in reversed(p.wind_stages) if step >= row[0])
    return np.asarray(row[1:], dtype=float)


def generate_maritime(cfg, seed):
    from .scenario import Scenario
    p = MaritimeParameters.from_config(cfg)
    initial_rng, motion_rng, comm_rng = [np.random.default_rng(s)
        for s in np.random.SeedSequence(seed).spawn(3)]
    arrival_rng = np.random.default_rng(np.random.SeedSequence([0x41525231, seed]))
    release = release_schedule(cfg, arrival_rng)
    n = len(release); steps = cfg.horizon + cfg.return_steps + 1
    initial = initial_rng.uniform(0, 1, (n, 2))
    radius = p.anchor_radius_m / p.area_m
    if cfg.mobility == 'maritime-anchored':
        initial = radius + (1 - 2 * radius) * initial
    anchors = initial.copy()
    positions = np.empty((steps, n, 2)); positions[0] = initial
    noise = motion_rng.normal(0, p.residual_mps * p.dt_seconds / p.area_m, (steps-1, n, 2))
    for t in range(1, steps):
        velocity = current_velocity(positions[t-1], t-1, p)
        if cfg.mobility != 'maritime-current':
            velocity += p.windage * wind_velocity(t-1, p)
        moved = positions[t-1] + velocity * p.dt_seconds / p.area_m + noise[t-1]
        if cfg.mobility == 'maritime-anchored':
            # Rejected proposals stay put. Disk is wholly inside the domain.
            # This is a bounded random walk, not a mechanical mooring model.
            inside = np.linalg.norm(moved - anchors, axis=1) <= radius
            bounded = np.where(inside[:, None], moved, positions[t-1])
        else:
            bounded = 1. - np.abs(np.mod(moved, 2.) - 1.)
        positions[t] = np.where((release < t)[:, None], bounded, initial)
    uniform = comm_rng.random((steps, n, cfg.uavs)).astype(np.float32)
    return Scenario(positions, release, deadlines(cfg, release), uniform, int(seed), cfg.dict())
