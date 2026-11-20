"""OBS-LA: observation-only event ordering with a Defer-B continuation.

The selector accepts only immutable telemetry. Live allocation is separate;
privileged agreement diagnostics belong exclusively to the offline audit.
"""
import hashlib
import time
from .commitment import (commitment_planning_env, delayed_suffix, event_now,
                         release_unstarted)
from .decision_headroom import ORDERS, allocate
from .maritime_planning import observe_maritime, unpack_maritime, sample_maritime_future
from .planning import planning_seeds
from .policies import Controller

NAMESPACE = 'wp2b2c-observable-order-v1:'


def select_order(observation, k=8, check=lambda: None):
    if type(k) is not int or k not in (4, 8):
        raise ValueError('Only cost-frozen K=8 or K=4 supported')
    data, _ = unpack_maritime(observation)
    anchor = int.from_bytes(hashlib.sha256(
        (NAMESPACE + observation.fingerprint()).encode()).digest()[:8], 'little')
    seeds = planning_seeds(anchor, k)
    values = [[] for _ in ORDERS]; hashes = []
    for seed in seeds:
        check(); scenario = sample_maritime_future(observation, seed)
        fingerprint = scenario.fingerprint(); hashes.append(fingerprint)
        for index, order in enumerate(ORDERS):
            check()
            branch = commitment_planning_env(observation.base(), scenario, log_events=False)
            allocate(branch, order, check=check)
            values[index].append(delayed_suffix(branch, check, prepared=True))
            if scenario.fingerprint() != fingerprint:
                raise RuntimeError('Branch mutated its shared sampled future')
    sums = [sum(row) for row in values]
    winner = max(range(len(ORDERS)), key=lambda i: sums[i])
    return dict(time=data['base']['time'], observation=data,
                observation_sha256=observation.fingerprint(), k=k,
                seeds=seeds, sample_hashes=hashes, orders=list(ORDERS),
                completed=values, sums=sums,
                means=[dict(numerator=v, denominator=k) for v in sums],
                selected=ORDERS[winner])


def run(env, k=8, check=lambda: None, *, arrival_rate_multiplier=1.):
    observe_maritime(env, arrival_rate_multiplier=arrival_rate_multiplier)  # Fail closed before mutation.
    planner = Controller(env.cfg, 'cheapest'); previous = []; traces = []; events = 0
    seconds = dict(selection=0., allocation=0., planning=0.)
    while not env.done:
        check()
        if env.time < env.cfg.horizon:
            event = event_now(env, previous); events += int(event)
            if event:
                release_unstarted(env)
                start = time.perf_counter()
                row = select_order(observe_maritime(env, arrival_rate_multiplier=arrival_rate_multiplier), k, check)
                row['seconds'] = time.perf_counter()-start
                seconds['selection'] += row['seconds']; traces.append(row)
                order = row['selected']
            else:
                order = 'cheapest'
            start = time.perf_counter(); allocate(env, order, check=check)
            seconds['allocation'] += time.perf_counter()-start
            start = time.perf_counter(); planner.plan(env)
            seconds['planning'] += time.perf_counter()-start
        _, previous = env.tick()
    return dict(events=events, decisions=len(traces), traces=traces), dict(seconds=seconds,
        sampled_suffixes=4*k*len(traces), decision_seconds=seconds['selection'])
