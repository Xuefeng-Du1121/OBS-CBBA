"""Observation-only, IID conditional rollout prototype; not a deployed gate.

The trusted adapter alone sees the real simulator. All other entry points accept
an immutable JSON observation and independent planning seeds. Scenario prefixes
in sampled worlds are placeholders, NOT reconstructions of the observed past.
"""
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .config import Config
from .arrivals import release_schedule, deadlines
from .env import SwarmEnv
from .policies import Controller
from .scenario import Scenario
from .switching import apply_switch, switch_decision

FLEET = ('position', 'energy', 'distance', 'active', 'sensing',
         'planned_interaction', 'last_switch_step')
TASK = ('owner', 'ever_assigned', 'assigned_at', 'completed_at')
HISTORY = ('queues', 'history', 'delays', 'deviations')
COUNTERS = ('conflicts_detected', 'conflicts_resolved', 'response_decisions', 'fallback_decisions')
ACTIONS = ('base_continue', 'preempt_then_base')


def supported(cfg):
    cfg.validate()
    if (cfg.observation_delay != 0 or cfg.mobility != 'iid' or cfg.nominal_speed != 0
            or cfg.regional_cost_mode != 'fixed-reference' or cfg.regional_calibration is not None
            or cfg.task_order != 'deadline' or cfg.reward_protocol != 'event-components-v1'
            or cfg.switching_protocol != 'restart-service-v1'):
        raise ValueError('Prototype requires zero-delay IID zero-drift, fixed-reference, deadline order, '
                         'event-components-v1 and restart-service-v1')
    if not np.isfinite(cfg.sigma) or cfg.sigma < 0 or not 0 <= cfg.gamma <= 1:
        raise ValueError('Invalid motion variance or discount')


def public_schedule(cfg):
    """Arrival times/counts are a public deterministic rule in this simulator."""
    if cfg.arrival_process != 'deterministic':
        raise ValueError('Realized Poisson arrivals are not a public schedule')
    release = release_schedule(cfg)
    return release, deadlines(cfg, release)


@dataclass(frozen=True)
class PlanningObservation:
    payload: str

    def as_dict(self):
        return json.loads(self.payload)

    def fingerprint(self):
        return hashlib.sha256(self.payload.encode()).hexdigest()


def observe(env):
    """Trusted whitelist adapter. Never exports seed, raw uniforms or hidden tasks."""
    supported(env.cfg)
    if not 0 <= env.time < env.cfg.horizon:
        raise ValueError('Observe a scheduling state before the horizon')
    known = np.flatnonzero(env.scenario.release <= env.time)
    if not np.array_equal(known, np.arange(len(known))):
        raise ValueError('Released tasks must form an observed contiguous ID prefix')
    if env.cfg.arrival_process == 'deterministic':
        release, _ = public_schedule(env.cfg)
        expected = np.flatnonzero(release <= env.time)
        if not np.array_equal(known, expected) or not np.array_equal(env.scenario.release[known], release[known]):
            raise ValueError('Observed releases disagree with the public arrival schedule')
    d = dict(version=1, config=env.cfg.dict(), time=int(env.time), known=known.tolist(),
             task_positions=env.scenario.positions[env.time, known].tolist(),
             deadline=env.scenario.deadline[known].tolist(),
             available=(env.scenario.response_uniform[env.time, known] >= env.cfg.loss).tolist(),
             failed=sorted(env.reward_failed_tasks))
    if env.cfg.arrival_process == 'poisson':
        d.update(version=2, release=env.scenario.release[known].tolist())
    for key in FLEET:
        d[key] = getattr(env, key).tolist()
    for key in TASK:
        d[key] = getattr(env, key)[known].tolist()
    for key in HISTORY + COUNTERS:
        d[key] = getattr(env, key)
    d['bandit'] = dict(variant=env.bandit.variant, mean=env.bandit.mean.tolist(),
                       count=env.bandit.count.tolist(),
                       recent=[[list(q) for q in row] for row in env.bandit.recent])
    return PlanningObservation(json.dumps(d, sort_keys=True, allow_nan=False))


def unpack(observation):
    if not isinstance(observation, PlanningObservation):
        raise TypeError('Expected PlanningObservation, never a live simulator or snapshot')
    d = observation.as_dict()
    if d['version'] not in (1, 2):
        raise ValueError('Unknown observation version')
    cfg = Config(**d['config'])
    supported(cfg)
    return d, cfg


def planning_seeds(anchor, count):
    if type(anchor) is not int or anchor < 0 or type(count) is not int or count < 1:
        raise ValueError('Nonnegative integer anchor and positive integer count required')
    # Separate namespace from real scenario generation and named dataset splits.
    return [int(x.generate_state(1, dtype=np.uint64)[0])
            for x in np.random.SeedSequence([0x57465031, anchor]).spawn(count)]


def sample_future(observation, seed, *, arrival_rate_multiplier=1.):
    return _sample_future(observation, seed, arrival_rate_multiplier=arrival_rate_multiplier)


def sample_future_leaked_arrivals(observation, seed, future_release):
    """Explicit offline leakage control: arrival schedule only, not locations.

    Never call from an observation-only policy. The supplied vector contains
    ONLY future release times; sampled motion and communication remain hidden.
    """
    return _sample_future(observation, seed, leaked_release=future_release)


def _sample_future(observation, seed, leaked_release=None, *, arrival_rate_multiplier=1.):
    d, cfg = unpack(observation)
    if type(seed) is not int or seed < 0:
        raise ValueError('Nonnegative integer planning seed required')
    initial_rng, motion_rng, comm_rng = [np.random.default_rng(s)
        for s in np.random.SeedSequence([0x434F4E44, seed]).spawn(3)]
    if cfg.arrival_process == 'deterministic':
        release, deadline = public_schedule(cfg)
        if leaked_release is not None:
            raise ValueError('Arrival leakage control requires Poisson mode')
    else:
        arrival_rng = np.random.default_rng(np.random.SeedSequence([0x43415252, seed]))
        future = release_schedule(cfg, arrival_rng, after=d['time'], rate_multiplier=arrival_rate_multiplier)
        if leaked_release is not None:
            future = np.asarray(leaked_release)
            if future.size == 0:
                future = future.astype(np.int64)
            allowed_slots = release_schedule(Config(**{**cfg.dict(), 'arrival_process':'deterministic'}))
            if (future.ndim != 1 or not np.issubdtype(future.dtype, np.integer)
                    or np.any(future <= d['time']) or np.any(np.diff(future) < 0)
                    or not np.isin(future, allowed_slots).all()):
                raise ValueError('Invalid leaked future release schedule')
        release = np.concatenate((np.asarray(d['release'], dtype=np.int64), future)).astype(np.int64)
        deadline = deadlines(cfg, release)
    known = np.asarray(d['known'], dtype=int)
    t = d['time']; steps = cfg.horizon + cfg.return_steps + 1
    initial = initial_rng.uniform(0, 1, (len(release), 2))
    initial[known] = d['task_positions']
    deadline[known] = d['deadline']
    positions = np.empty((steps, len(release), 2), dtype=np.float64)
    positions[:t+1] = initial  # synthetic, unused pre-decision prefix
    for k in range(t+1, steps):
        moved = positions[k-1] + motion_rng.normal(0, cfg.sigma, initial.shape)
        reflected = 1. - np.abs(np.mod(moved, 2.) - 1.)
        positions[k] = np.where((release < k)[:, None], reflected, initial)
    uniform = comm_rng.random((steps, len(release), cfg.uavs)).astype(np.float32)
    # Representative values encode only observed availability; never latent draws.
    uniform[t, known] = np.asarray(d['available'], dtype=np.float32)
    return Scenario(positions, release, deadline, uniform, seed, cfg.dict())


def planning_env(observation, scenario, *, log_events=True):
    """Materialize a sampled world; observation carries no actual hidden future."""
    d, cfg = unpack(observation)
    known = np.asarray(d['known'], dtype=int)
    if cfg.arrival_process == 'deterministic':
        release, _ = public_schedule(cfg)
        arrivals_match = np.array_equal(scenario.release, release)
    else:
        arrivals_match = (np.array_equal(np.flatnonzero(scenario.release <= d['time']), known)
                          and np.array_equal(scenario.release[known], d['release']))
    if (scenario.config != cfg.dict() or not arrivals_match
            or not np.array_equal(scenario.positions[d['time'], known], d['task_positions'])
            or not np.array_equal(scenario.deadline[known], d['deadline'])
            or not np.array_equal(scenario.response_uniform[d['time'], known] >= cfg.loss, d['available'])):
        raise ValueError('Sample does not match the observation')
    env = SwarmEnv(cfg, scenario, d['bandit']['variant'], log_events=log_events)
    env.time = d['time']
    for key in FLEET:
        setattr(env, key, np.asarray(d[key], dtype=getattr(env, key).dtype))
    for key in TASK:
        getattr(env, key)[known] = d[key]
    for key in HISTORY + COUNTERS:
        setattr(env, key, d[key])
    env.reward_failed_tasks = set(d['failed'])
    env.bandit.mean[:] = d['bandit']['mean']
    env.bandit.count[:] = d['bandit']['count']
    env.bandit.recent = [[deque(q, maxlen=cfg.history) for q in row] for row in d['bandit']['recent']]
    return env


def finish_branch(env, uav, task, action):
    """Offline intervention, followed by ordinary cheapest at EVERY round.

    Unlike the legacy diagnostic 'continue', base_continue does not defer the
    offered task. This action pair has its own protocol identity below.
    """
    if action not in ACTIONS:
        raise ValueError('Unknown action')
    decision = switch_decision(env, uav, task)
    if decision is None:
        raise ValueError('No eligible preemption at this observation')
    if action == 'preempt_then_base':
        apply_switch(env, decision, diagnostic=True)
    controller = Controller(env.cfg, 'cheapest')
    start = len(env.reward_ledger)
    while not env.done:
        if env.time < env.cfg.horizon:
            controller.allocate(env)
            controller.plan(env)
        env.tick()
    ledger = env.reward_ledger[start:]
    return dict(action=action, discounted_return=float(sum(env.cfg.gamma**i * r['joint']
                for i, r in enumerate(ledger))), steps=len(ledger),
                completed_total=int((env.completed_at >= 0).sum()))


def evaluate_pair(observation, uav, task, seeds, *, record=None):
    """K paired conditional draws; proposal only, no live-environment mutation.

    record(draw_index, scenario, branch_env, branch_result) is for evidence I/O.
    Sample SD/SE are descriptive, not confidence bounds or safety certificates.
    """
    unpack(observation)
    seeds = list(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('At least two distinct planning seeds required')
    rows = []
    for i, seed in enumerate(seeds):
        scenario = sample_future(observation, seed)
        sample_hash = scenario.fingerprint()
        results = []
        for action in ACTIONS:
            env = planning_env(observation, scenario, log_events=record is not None)
            result = finish_branch(env, uav, task, action)
            if scenario.fingerprint() != sample_hash:
                raise RuntimeError('Rollout mutated exogenous sample')
            results.append(result)
            if record is not None:
                record(i, scenario, env, result)
        rows.append(dict(seed=seed, sample_hash=sample_hash, branches=results,
                         delta=results[1]['discounted_return']-results[0]['discounted_return']))
    deltas = np.asarray([r['delta'] for r in rows])
    mean = float(deltas.mean()); sd = float(deltas.std(ddof=1))
    return dict(protocol='conditional-pair-base-continue-v1', observation_hash=observation.fingerprint(),
                uav=int(uav), task=int(task), k=len(rows), rows=rows, mean_delta=mean,
                sample_sd=sd, descriptive_se=sd / len(rows)**.5,
                proposal=ACTIONS[int(mean > 0)], deployed=False)
