"""Exogenous randomness is generated before any policy action."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
from .config import Config
from .arrivals import release_schedule, deadlines


def split_seed(split, training_seed, instance):
    """Named independent streams; no adjacent integer ranges that can overlap."""
    code={'train':0,'validation':1,'test':2}[split]
    return int(np.random.SeedSequence([code,training_seed,instance]).generate_state(1,dtype=np.uint64)[0])


@dataclass
class Scenario:
    positions: np.ndarray  # physical time, task, xy; future never exposed to policy
    release: np.ndarray
    deadline: np.ndarray
    response_uniform: np.ndarray  # time, task, UAV; independent of call ordering
    seed: int
    config: dict

    @classmethod
    def generate(cls, cfg: Config, seed: int):
        cfg.validate()
        if cfg.mobility == 'real-current':
            from .real_drift import generate_real_drift
            return generate_real_drift(cfg, seed)
        if cfg.mobility.startswith('maritime-'):
            from .maritime import generate_maritime
            return generate_maritime(cfg, seed)
        streams = [np.random.default_rng(x) for x in np.random.SeedSequence(seed).spawn(3)]
        initial_rng, motion_rng, comm_rng = streams
        # Separate namespace leaves the original three streams bitwise unchanged.
        arrival_rng = np.random.default_rng(np.random.SeedSequence([0x41525231, seed]))
        release = release_schedule(cfg, arrival_rng)
        deadline = deadlines(cfg, release)
        steps, n = cfg.horizon + cfg.return_steps + 1, len(release)
        initial = initial_rng.uniform(0, 1, (n, 2))
        positions = np.empty((steps, n, 2), dtype=np.float64)
        positions[0] = initial
        angles = initial_rng.uniform(0, 2 * np.pi, n)
        velocity = cfg.nominal_speed * np.stack((np.cos(angles), np.sin(angles)), axis=1)
        noise = motion_rng.normal(0, cfg.sigma, (steps - 1, n, 2))
        for t in range(1, steps):
            drift = velocity.copy()
            if cfg.mobility != "iid":
                side = np.where(positions[t-1, :, 0] < .5, 1., -1.)
                if cfg.mobility == "switch" and t >= cfg.horizon // 2:
                    side = -side
                drift[:, 1] += side * cfg.sigma
            moved = positions[t-1] + drift + noise[t-1]
            # Reflection, including overshoots > 1; unreleased tasks stay at release locations.
            reflected = 1. - np.abs(np.mod(moved, 2.) - 1.)
            positions[t] = np.where((release < t)[:, None], reflected, initial)
        uniform = comm_rng.random((steps, n, cfg.uavs)).astype(np.float32)
        return cls(positions, release, deadline, uniform, int(seed), cfg.dict())

    def fingerprint(self):
        h = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode())
        for a in (self.positions, self.release, self.deadline, self.response_uniform):
            h.update(a.tobytes())
        return h.hexdigest()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, positions=self.positions, release=self.release,
                            deadline=self.deadline, response_uniform=self.response_uniform,
                            seed=self.seed, config=json.dumps(self.config))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(z['positions'], z['release'], z['deadline'],
                       z['response_uniform'], int(z['seed']), json.loads(str(z['config'])))
