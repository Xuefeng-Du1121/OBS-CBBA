"""Fixed insertion orders and privileged reference allocators (development references)."""
import math
import time
import numpy as np
from .commitment import (CommitmentEnv, delayed_suffix, event_now,
                         hindsight_repair, release_unstarted)
from .policies import Controller
from .route_geometry import RouteGeometry

ORDERS = ('cheapest', 'deadline', 'regret', 'slack')


def category(feasible, responding, empty):
    if any(f and r for f, r in zip(feasible, responding)):
        return 'inserted'
    if any(feasible):
        return 'response_blocked'
    return 'commitment_constrained' if any(empty) else 'empty_route_infeasible'


def allocate(env, order='cheapest', observations=None, check=lambda: None):
    """Static ranking at allocation entry, followed by ordinary greedy insertion.

    Empty-route probes are diagnostic relaxations (even active work is removed).
    They never modify the live queue or availability. Response is current only.
    """
    if order not in ORDERS:
        raise ValueError(order)
    pending = list(env.pending())
    geometry = RouteGeometry(env) if pending else None
    if order == 'regret':
        def regret(j):
            costs = sorted(ins[0] for u in range(env.cfg.uavs)
                           if (ins := env.insertion(u, j, geometry=geometry)) is not None)
            return costs[1]-costs[0] if len(costs)>1 else (1e6 if costs else -1e6)
        pending.sort(key=regret, reverse=True)
    elif order == 'slack':
        positions = env.observed_positions()
        def slack(j):
            travel = min(math.ceil(max(float(np.linalg.norm(positions[j]-p))-1e-9, 0.)
                                   /env.cfg.speed) for p in env.position)
            return (int(env.scenario.deadline[j])-env.time-travel-env.cfg.sensing_steps,
                    int(env.scenario.deadline[j]), int(j))
        pending.sort(key=slack)
    # cheapest and deadline deliberately coincide: env.pending is already EDF.
    for j in pending:
        check(); started = time.perf_counter()
        insertions = [env.insertion(u, j, geometry=geometry) for u in range(env.cfg.uavs)]
        responding = (env.scenario.response_uniform[env.time,j] >= env.cfg.loss).tolist()
        feasible = [ins is not None for ins in insertions]
        valid = [u for u in range(env.cfg.uavs) if feasible[u] and responding[u]]
        if observations is not None:
            empty = [bool(env.route_estimate(u, [j], reset_service=True)['feasible'])
                     for u in range(env.cfg.uavs)]
            observations.append(dict(time=int(env.time), task=int(j), feasible=feasible,
                responding=responding, empty=empty, category=category(feasible,responding,empty)))
        elapsed = time.perf_counter()-started; env.compute_seconds += elapsed
        if valid:
            u = min(valid, key=lambda u: insertions[u][0])
            env.assign(j, u, insertions[u], elapsed)


def order_oracle(env, check=lambda: None):
    snap = env.snapshot(); values = []
    started = time.perf_counter()
    for order in ORDERS:
        check(); branch = CommitmentEnv.from_snapshot(snap)
        branch.log_events = False; branch.events = []; branch.reward_ledger = []
        allocate(branch, order, check=check)
        values.append(delayed_suffix(branch, check, prepared=True))
    chosen = max(range(len(ORDERS)), key=lambda i: values[i])
    allocate(env, ORDERS[chosen], check=check)
    return dict(time=int(env.time), orders=list(ORDERS), completed=values,
                selected=ORDERS[chosen], improvement=values[chosen]-values[0],
                seconds=time.perf_counter()-started)


def run(env, method, check=lambda: None):
    if method not in ('b', 'd_full', 'o_true'):
        raise ValueError(method)
    planner = Controller(env.cfg, 'cheapest'); previous = []; traces = []; observations = []
    events = 0
    while not env.done:
        check()
        if env.time < env.cfg.horizon:
            event = event_now(env, previous); events += int(event)
            if event:
                release_unstarted(env)
            if method == 'o_true' and event:
                traces.append(order_oracle(env, check))
            else:
                allocate(env, observations=observations if method == 'b' else None, check=check)
                if method == 'd_full' and event:
                    # Python [:None] evaluates the complete candidate list, not a cap.
                    traces.append(hindsight_repair(env, check, max_candidates=None))
            planner.plan(env)
        _, previous = env.tick()
    return dict(events=events, decisions=len(traces), traces=traces, observations=observations)
