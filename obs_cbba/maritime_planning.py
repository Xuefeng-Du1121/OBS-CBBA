"""Drift estimation from observed history and sampled-future lookahead.

The trusted adapter assumes exact, global telemetry for every released task at
each tick (zero delay). No future forcing, anchors, seed or residual SD leaves
that adapter. The plug-in sampler is deliberately not an oracle ocean model.
"""
from copy import copy
from dataclasses import dataclass, replace
import hashlib
import json
import math
import numpy as np
from .planning import PlanningObservation, planning_seeds, sample_future
from .commitment import (commitment_observe, commitment_planning_env, delayed_suffix,
                         event_now, release_unstarted, repair_candidates, reassign, hindsight_repair)
from .observed_lookahead import select_from_outcomes
from .route_time_repair import choose_route_time
from .policies import Controller


@dataclass(frozen=True)
class DriftSettings:
    history_steps: int = 8
    grid: int = 3
    prior_step_sd: float = .0002  # normalized displacement; public design prior
    min_step_sd: float = .000001

    def validate(self):
        if (type(self.history_steps) is not int or self.history_steps < 2
                or type(self.grid) is not int or self.grid < 1
                or not math.isfinite(self.prior_step_sd) or not math.isfinite(self.min_step_sd)
                or not 0 < self.min_step_sd <= self.prior_step_sd):
            raise ValueError('Invalid history estimator settings')
        return self


@dataclass(frozen=True)
class MaritimeObservation:
    payload: str

    def as_dict(self):
        return json.loads(self.payload)

    def fingerprint(self):
        return hashlib.sha256(self.payload.encode()).hexdigest()

    def base(self):
        return PlanningObservation(json.dumps(self.as_dict()['base'], sort_keys=True, allow_nan=False))


def observe_maritime(env, settings=DriftSettings(), *, arrival_rate_multiplier=1.):
    """Only this trusted adapter may inspect the true environment's past."""
    settings.validate(); env.cfg.validate()
    validate_arrival_belief(arrival_rate_multiplier, env.cfg.arrival_process)
    if env.cfg.mobility not in ('iid', 'maritime-current', 'maritime-wind', 'maritime-anchored', 'real-current'):
        raise ValueError('Unsupported motion mode')
    if env.cfg.observation_delay != 0 or env.cfg.nominal_speed != 0:
        raise ValueError('This history prototype requires zero delay and zero legacy drift')
    proxy = copy(env)
    # All truth-side motion settings are stripped, including residual strength.
    # Planning env needs only the shared fleet/task physics and arrival law.
    # Real-current provenance is intentionally removed from the public planner
    # input.  In particular, the local HYCOM path/hash, panel location/date and
    # future forcing must never become policy features or seed material.
    proxy.cfg = replace(env.cfg, mobility='iid', maritime=None, real_drift=None,
                        nominal_speed=0., sigma=0.)
    base = commitment_observe(proxy).as_dict()
    frames = []
    for t in range(max(0, env.time - settings.history_steps + 1), env.time + 1):
        ids = np.flatnonzero(env.scenario.release <= t)
        frames.append(dict(time=t, ids=ids.tolist(), positions=env.scenario.positions[t, ids].tolist()))
    data = dict(version='maritime-history-v1', base=base, frames=frames,
                settings=dict(vars(settings)))
    if arrival_rate_multiplier != 1:
        data['arrival_rate_multiplier'] = arrival_rate_multiplier
    return MaritimeObservation(json.dumps(data, sort_keys=True, allow_nan=False))


def observe_real_current_history(env, settings=DriftSettings(), *, arrival_rate_multiplier=1.):
    """History-only adapter for an archived HYCOM-driven episode.

    This is a named prerequisite entry point so callers can make the data
    boundary explicit.  It deliberately shares the trusted whitelist with the
    maritime adapter: only released-task telemetry through ``env.time`` and
    the public arrival law are exposed.  The result contains no real-drift
    path, hash, location/date metadata, future current field, seed, or hidden
    task count.
    """
    if getattr(env.cfg, 'mobility', None) != 'real-current':
        raise ValueError('Real-current history requires mobility=real-current')
    return observe_maritime(env, settings, arrival_rate_multiplier=arrival_rate_multiplier)


def validate_arrival_belief(multiplier, process):
    if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)) or not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('Positive finite planner arrival rate multiplier required')
    if multiplier != 1 and process != 'poisson':
        raise ValueError('Arrival rate mismatch requires Poisson arrivals')


def unpack_maritime(observation):
    if not isinstance(observation, MaritimeObservation):
        raise TypeError('Expected history-only MaritimeObservation')
    d = observation.as_dict()
    if set(d) - {'arrival_rate_multiplier'} != {'version', 'base', 'frames', 'settings'} or d['version'] != 'maritime-history-v1':
        raise ValueError('Unknown maritime observation')
    settings = DriftSettings(**d['settings']).validate()
    from .planning import unpack
    base, cfg = unpack(observation.base())
    validate_arrival_belief(d.get('arrival_rate_multiplier', 1.), cfg.arrival_process)
    if cfg.sigma != 0 or 'maritime' in base['config']:
        raise ValueError('Truth-side motion parameters in planner input')
    times = [row['time'] for row in d['frames']]
    if times != list(range(max(0, base['time'] - settings.history_steps + 1), base['time'] + 1)):
        raise ValueError('Noncausal or incomplete telemetry window')
    previous = set()
    for row in d['frames']:
        ids = row['ids']; points = np.asarray(row['positions'], dtype=float)
        if (ids != list(range(len(ids))) or not previous.issubset(ids)
                or not set(ids).issubset(base['known']) or points.shape != (len(ids), 2)
                or not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1)):
            raise ValueError('Invalid released-task history')
        previous = set(ids)
    last = d['frames'][-1]
    if last['ids'] != base['known'] or last['positions'] != base['task_positions']:
        raise ValueError('History does not end at the current observation')
    return d, settings


def cell_indices(points, grid):
    xy = np.minimum((np.asarray(points) * grid).astype(int), grid-1)
    return xy[:, 0] * grid + xy[:, 1]


def estimate_drift(observation):
    d, settings = unpack_maritime(observation)
    increments = []; cells = []
    for before, after in zip(d['frames'], d['frames'][1:]):
        n = len(before['ids'])
        if not n:
            continue
        start = np.asarray(before['positions'])
        increments.extend(np.asarray(after['positions'])[:n] - start)
        cells.extend(cell_indices(start, settings.grid))
    delta = np.asarray(increments, dtype=float).reshape(-1, 2)
    cells = np.asarray(cells, dtype=int)

    def stats(values):
        n = len(values)
        mean = values.mean(axis=0) if n else np.zeros(2)
        sd = (values.std(axis=0, ddof=1) if n >= 2
              else np.full(2, settings.prior_step_sd))
        return dict(count=n, mean=mean.tolist(),
                    sd=np.maximum(sd, settings.min_step_sd).tolist())

    global_stats = stats(delta); regions = []
    for index in range(settings.grid**2):
        values = delta[cells == index]
        # Sparse regions borrow the global estimate, not true forcing.
        regions.append(dict(**(stats(values) if len(values) >= 2 else global_stats),
                            borrowed_global=len(values) < 2))
    return dict(global_stats=global_stats, regions=regions, grid=settings.grid,
                units='normalized displacement per tick', calibrated=False)


def sample_maritime_future(observation, seed):
    d, settings = unpack_maritime(observation)
    estimate = estimate_drift(observation)
    scenario = sample_future(observation.base(), seed,
                             arrival_rate_multiplier=d.get('arrival_rate_multiplier', 1.))
    t = d['base']['time']; rng = np.random.default_rng(np.random.SeedSequence([0x4D415249, seed]))
    means = np.asarray([r['mean'] for r in estimate['regions']])
    sds = np.asarray([r['sd'] for r in estimate['regions']])
    counts = np.asarray([max(r['count'], 1) for r in estimate['regions']])
    # One regional drift error per future, shared across tasks and time; fresh
    # residuals each tick. Independence-based SE is a heuristic, not a CI.
    means = means + rng.normal(size=means.shape) * sds / np.sqrt(counts[:, None])
    initial = scenario.positions[t].copy()
    for k in range(t+1, len(scenario.positions)):
        cells = cell_indices(scenario.positions[k-1], settings.grid)
        moved = scenario.positions[k-1] + means[cells] + rng.normal(size=initial.shape) * sds[cells]
        reflected = 1. - np.abs(np.mod(moved, 2.) - 1.)
        scenario.positions[k] = np.where((scenario.release < k)[:, None], reflected, initial)
    return scenario


def evaluate_maritime(observation, candidates, check=lambda: None):
    unpack_maritime(observation)
    anchor = int.from_bytes(hashlib.sha256(('wp2a-history-k8-v1:' + observation.fingerprint()).encode()).digest()[:8], 'little')
    seeds = planning_seeds(anchor, 8) if candidates else []
    base = []; values = [[] for _ in candidates]; hashes = []
    for seed in seeds:
        check(); scenario = sample_maritime_future(observation, seed)
        fingerprint = scenario.fingerprint(); hashes.append(fingerprint)
        for index, move in enumerate([None] + candidates):
            check()
            env = commitment_planning_env(observation.base(), scenario, log_events=False)
            if move is not None:
                reassign(env, move['task'], move['destination'])
            count = delayed_suffix(env, check, prepared=True)
            (base if index == 0 else values[index-1]).append(count)
        assert scenario.fingerprint() == fingerprint
    result = select_from_outcomes(candidates, base, values, 'd_obs')
    return dict(seeds=seeds, sample_hashes=hashes, base_completed=base, **result)


def run_maritime(env, method, settings=DriftSettings(), check=lambda: None):
    """Development adapter only; no matrix, seed allocation, or launch command."""
    if method not in ('b', 'f', 'd_true', 'k8'):
        raise ValueError(method)
    observe_maritime(env, settings)  # fail closed before changing the state
    controller = Controller(env.cfg, 'cheapest'); completed = []; traces = []; events = 0
    while not env.done:
        check()
        if env.time < env.cfg.horizon:
            event = event_now(env, completed); events += int(event)
            if event:
                release_unstarted(env)
            controller.allocate(env)
            if event and method != 'b':
                if method == 'd_true':
                    result = hindsight_repair(env, check)  # explicit privileged comparator
                else:
                    obs = observe_maritime(env, settings); moves = repair_candidates(env)
                    if method == 'f':
                        moves = [dict(**m, insertion=int(env.insertion(m['destination'], m['task'])[1])) for m in moves]
                        result = choose_route_time(obs.base(), moves)
                    else:
                        result = evaluate_maritime(obs, moves[:12], check)
                    result = dict(observation=obs.as_dict(), **result)
                    if result['selected'] is not None:
                        move = result['selected']; reassign(env, move['task'], move['destination'])
                env.log('maritime_decision', method=method, selected=result['selected'])
                traces.append(dict(time=env.time, method=method, **result) if 'time' not in result else dict(method=method, **result))
            controller.plan(env)
        _, completed = env.tick()
    return dict(events=events, decisions=len(traces), traces=traces)
