"""Observation-conditioned, paired action evaluation with a Defer-B continuation.

The policy functions accept only a whitelisted observation and observed actions.
No true-future counterfactual is available to the deployed decision rule.
"""
import hashlib
import math
import time
import numpy as np
from .planning import planning_seeds, sample_future
from .commitment import (commitment_observe, commitment_planning_env, repair_candidates,
                         delayed_suffix, reassign, release_unstarted, event_now)
from .route_time_repair import run_route_time
from .policies import Controller

METHODS = ('b', 'f', 'd_true', 'd_obs', 'd_obs_lcb')
# scipy.stats.t.ppf(.9, K-1), frozen rather than adding a runtime dependency.
T90 = {8: 1.4149239276488585, 16: 1.3406056078504547}


def observation_seeds(observation, k):
    if k not in T90:
        raise ValueError('Supports K=8 or K=16')
    identity = 'wp1g-paired-b-suffix-v1:' + observation.fingerprint()
    anchor = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'little')
    return planning_seeds(anchor, k)


def paired_statistics(base, values):
    delta = np.asarray(values, dtype=np.int64) - np.asarray(base, dtype=np.int64)
    k = len(delta)
    if k not in T90 or len(base) != len(values):
        raise ValueError('Invalid paired outcomes')
    mean = float(delta.mean())
    se = float(delta.std(ddof=1)) / math.sqrt(k)
    return dict(delta=delta.tolist(), sum_delta=int(delta.sum()), mean_delta=mean,
                standard_error=se, lcb90=mean-T90[k]*se)


def select_from_outcomes(candidates, base, values, method):
    if method not in ('d_obs', 'd_obs_lcb'):
        raise ValueError(method)
    if len(candidates) != len(values):
        raise ValueError('Missing action outcomes')
    scored = [dict(**move, completed=list(map(int, value)), **paired_statistics(base, value))
              for move, value in zip(candidates, values)]
    # Integer sums choose the mean maximizer; equal means keep canonical first.
    best = max(scored, key=lambda row: row['sum_delta'], default=None)
    chosen = None
    if best is not None and best['sum_delta'] > 0:
        if method == 'd_obs' or best['lcb90'] > 0:
            chosen = {key: best[key] for key in ('task', 'source', 'destination')}
    return dict(candidates=scored, selected=chosen)


def evaluate_observed(observation, candidates, method, k, check=lambda: None):
    seeds = observation_seeds(observation, k) if candidates else []
    base = []; values = [[] for _ in candidates]; hashes = []
    for seed in seeds:
        check()
        scenario = sample_future(observation, seed)
        sample_hash = scenario.fingerprint(); hashes.append(sample_hash)
        for index, move in enumerate([None] + candidates):
            check()
            branch = commitment_planning_env(observation, scenario, log_events=False)
            if move is not None:
                reassign(branch, move['task'], move['destination'])
            completed = delayed_suffix(branch, check, prepared=True)
            (base if index == 0 else values[index-1]).append(completed)
        assert scenario.fingerprint() == sample_hash
    decision = select_from_outcomes(candidates, base, values, method)
    return dict(seeds=seeds, sample_hashes=hashes, base_completed=base, **decision)


def observed_lookahead(env, method, k, check=lambda: None):
    started = time.perf_counter()
    observation = commitment_observe(env)
    candidates = repair_candidates(env)
    result = evaluate_observed(observation, candidates[:12], method, k, check)
    env.log('observed_lookahead', method=method, selected=result['selected'])
    move = result['selected']
    if move is not None:
        ins = env.insertion(move['destination'], move['task'])
        assert ins is not None
        env.withdraw(move['task'], 'observed-lookahead-reassignment')
        env.assign(move['task'], move['destination'], ins)
    return dict(time=env.time, method=method, k=k, observation=observation.as_dict(),
                eligible=len(candidates), tested=min(12, len(candidates)), **result,
                seconds=time.perf_counter()-started)


def run_lookahead(env, method, k=16, check=lambda: None, recorded=None):
    """recorded supplies frozen decisions for cheap structural replay in auditors.

    Normal experiment execution never supplies recorded. It is deliberately an
    explicit argument, not an environment flag or global mutable cache.
    """
    if method in ('b', 'f', 'd_true'):
        return run_route_time(env, 'd' if method == 'd_true' else method, check)
    if method not in ('d_obs', 'd_obs_lcb') or k not in T90:
        raise ValueError((method, k))
    controller = Controller(env.cfg, 'cheapest')
    completed = []; traces = []; events = 0
    while not env.done:
        check()
        if env.time < env.cfg.horizon:
            event = event_now(env, completed); events += int(event)
            if event:
                release_unstarted(env)
            controller.allocate(env)
            if event:
                if recorded is None:
                    row = observed_lookahead(env, method, k, check)
                else:
                    row = next(recorded)
                    assert row['observation'] == commitment_observe(env).as_dict()
                    env.log('observed_lookahead', method=method, selected=row['selected'])
                    move = row['selected']
                    if move is not None:
                        ins = env.insertion(move['destination'], move['task'])
                        assert ins is not None
                        env.withdraw(move['task'], 'observed-lookahead-reassignment')
                        env.assign(move['task'], move['destination'], ins)
                traces.append(row)
            controller.plan(env)
        _, completed = env.tick()
    if recorded is not None:
        assert next(recorded, None) is None
    return dict(events=events, decisions=len(traces), traces=traces)
