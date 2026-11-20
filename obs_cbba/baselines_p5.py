"""CBBA, CBBA-PR and MSA baselines; each planner receives observations only.

CBBA uses synchronous all-to-all consensus, acquisition-order suffix removal,
route insertion, and nonincreasing bid clipping. Persistent bundles are locked
between events; PR resets only their unstarted acquisition-order tails.
MSA is a bounded scenario-routing/next-joint-action consensus approximation.
No planner below receives the live environment or its true scenario seed.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib
import math
import time
from types import SimpleNamespace
import numpy as np
from .config import Config
from .maritime_planning import observe_maritime, unpack_maritime, sample_maritime_future
from .planning import planning_seeds
from .route_geometry import RouteGeometry


@dataclass(frozen=True)
class Parameters:
    bundle_limit: int = 6
    reset_count: int = 2
    reset_fraction: float | None = None
    scenarios: int = 8
    future_limit: int = 12
    planning_seed: int = 0

    def validate(self):
        for name in ('bundle_limit', 'scenarios'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name)
        for name in ('reset_count', 'future_limit', 'planning_seed'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(name)
        if self.reset_fraction is not None and not 0 <= self.reset_fraction <= 1:
            raise ValueError('reset_fraction')
        return self


def observed_geometry(observation):
    d, _ = unpack_maritime(observation)
    b = d['base']; points = np.asarray(b['task_positions']).reshape(-1, 2)
    proxy = SimpleNamespace(cfg=Config(**b['config']), time=b['time'],
        position=np.asarray(b['position']), energy=np.asarray(b['energy']),
        distance=np.asarray(b['distance']), sensing=np.asarray(b['sensing']),
        active=np.asarray(b['active']), scenario=SimpleNamespace(deadline=b['deadline']),
        observed_positions=lambda: points)
    return b, RouteGeometry(proxy)


def insertion(geometry, u, task, route, active, counts):
    counts['insertion_calls'] += 1
    old = geometry.estimate(u, route)['length']; best = None
    first = int(active[u] >= 0 and bool(route))
    for k in range(first, len(route)+1):
        counts['route_evaluations'] += 1
        trial = route[:k]+[task]+route[k:]; estimate = geometry.estimate(u, trial)
        if estimate['feasible']:
            offer = (estimate['length']-old, k, trial)
            if best is None or offer[:2] < best[:2]:
                best = offer
    return best


def reset_tail(bundle, started, *, count=2, fraction=None):
    """Tail of remaining unstarted acquisition order, NOT execution route order.

    Completed/expired tasks are pruned by caller first. Started tasks are always
    protected. Fraction takes precedence and uses ceil(fraction * eligible).
    """
    Parameters(reset_count=count, reset_fraction=fraction).validate()
    eligible = [j for j in bundle if started[j] < 0]
    n = min(len(eligible), count if fraction is None else math.ceil(fraction*len(eligible)))
    return eligible[-n:] if n else []


def consensus(claims):
    """Perfect broadcast: highest bid wins; exact ties prefer lower UAV ID."""
    winners = {}
    for u, bundle in enumerate(claims):
        for j, bid in bundle:
            if j not in winners or (bid, -u) > (winners[j][1], -winners[j][0]):
                winners[j] = (u, bid)
    return winners


def cbba_allocate(observation, bundles, parameters=Parameters(), check=lambda: None):
    """Return new route plans without mutating telemetry or persistent bundles.

    Existing ownership is frozen. Only currently public tasks enter this auction.
    Synchronous local bundle growth is followed by global winner dissemination;
    the first lost bid invalidates that agent's entire dependent suffix.
    """
    p = parameters.validate(); b, geometry = observed_geometry(observation)
    n = b['config']['uavs']; fixed = [list(q) for q in b['queues']]
    pending = [j for j in b['known'] if b['owner'][j] < 0 and b['completed_at'][j] < 0
               and b['deadline'][j] > b['time']]
    claims = [[] for _ in range(n)]; routes = [q.copy() for q in fixed]
    winners = {}; counts = dict(insertion_calls=0, route_evaluations=0)
    rounds = 0; conflicts = 0
    # Fail closed if unsupported non-DMG geometry fails to converge; no silent fallback.
    for rounds in range(1, 2*n*max(1, len(pending))*p.bundle_limit+2):
        check()
        for u in range(n):
            while len(fixed[u])+len(claims[u]) < p.bundle_limit:
                check(); offers = []
                for j in pending:
                    if j in routes[u] or not b['available'][j][u]:
                        continue
                    offer = insertion(geometry, u, j, routes[u], b['active'], counts)
                    if offer is None:
                        continue
                    # Marginal reward for F(route)=|route| - 0.1*length(route).
                    bid = 1.-.1*offer[0]
                    if claims[u]:
                        bid = min(bid, claims[u][-1][1])
                    if bid <= 0:
                        continue
                    if j in winners and (bid, -u) <= (winners[j][1], -winners[j][0]):
                        continue
                    offers.append((bid, j, offer))
                if not offers:
                    break
                bid, j, offer = min(offers, key=lambda x: (-x[0], x[1], x[2][1]))
                claims[u].append((j, bid)); routes[u] = offer[2]
        shared = consensus(claims); lost = False
        for u in range(n):
            cut = next((k for k, (j, _) in enumerate(claims[u]) if shared[j][0] != u), len(claims[u]))
            if cut < len(claims[u]):
                lost = True; conflicts += 1
                removed = {j for j, _ in claims[u][cut:]}
                claims[u] = claims[u][:cut]
                routes[u] = [j for j in routes[u] if j not in removed]
        winners = consensus(claims)
        if not lost:
            break
    else:
        raise RuntimeError('CBBA consensus did not converge within the explicit round bound')
    out_bundles = [list(bundles[u])+[j for j, _ in claims[u]] for u in range(n)]
    return dict(routes=routes, bundles=out_bundles, rounds=rounds,
                conflicts=conflicts, claims=claims, counts=counts)


class ScenarioGeometry(RouteGeometry):
    """Fast MSA static insertion: known points now, future points at release.

    Release-time waiting and deadlines are enforced. Later motion, collisions,
    communication loss and hover energy are left to the unchanged live simulator.
    """
    def __init__(self, observation, scenario):
        b, observed = observed_geometry(observation)
        super().__init__(observed.env)
        self.scenario = scenario
        ticks = np.maximum(scenario.release, b['time'])
        self.points = scenario.positions[ticks, np.arange(len(ticks))]

    def estimate(self, u, route):
        e = self.env; c = e.cfg; s = self.scenario
        finish = e.time; length = consumption = late = 0.; previous = None
        for j in route:
            distance, travel = self.leg(u, previous, j)
            service = max(c.sensing_steps-int(e.sensing[u]), 1) if j == e.active[u] else c.sensing_steps
            finish = max(finish+travel, int(s.release[j]))+service
            late = max(late, finish-int(s.deadline[j]))
            length += distance; consumption += c.flight_energy*distance+c.sensing_energy*service/c.sensing_steps
            previous = j
        home, travel = self.home(u, previous)
        length += home; consumption += home*c.flight_energy
        return dict(length=length, finish=finish, feasible=bool(late <= 0 and
            e.energy[u]-consumption >= c.reserve and e.distance[u]+length <= c.max_distance and
            finish+travel <= c.horizon+c.return_steps))


def scenario_plan(observation, scenario, parameters, counts, check=lambda: None):
    b, live_geometry = observed_geometry(observation); n = b['config']['uavs']
    geometry = ScenarioGeometry(observation, scenario)
    routes = [list(q) for q in b['queues']]
    known = len(b['known'])
    tasks = [j for j in b['known'] if b['owner'][j] < 0 and b['completed_at'][j] < 0
             and b['deadline'][j] > b['time']]
    future = sorted(range(known, len(scenario.release)), key=lambda j: (scenario.release[j], j))
    tasks += future[:parameters.future_limit]
    tasks.sort(key=lambda j: (int(scenario.deadline[j]), int(scenario.release[j]), j))
    for j in tasks:
        check(); offers = []
        for u in range(n):
            if len(routes[u]) >= parameters.bundle_limit or (j < known and not b['available'][j][u]):
                continue
            offer = insertion(geometry, u, j, routes[u], b['active'], counts)
            if offer is not None:
                offers.append((offer[0], u, offer))
        if offers:
            _, u, offer = min(offers, key=lambda x: (x[0], x[1], x[2][1]))
            routes[u] = offer[2]
    actions = []
    for u, route in enumerate(routes):
        j = route[0] if route and b['active'][u] < 0 else -1
        # Only a currently feasible, available, observed next task is executable.
        if (j >= known or j < 0 or not b['available'][j][u]
                or not live_geometry.estimate(u, [j])['feasible']):
            j = -1
        actions.append(j)
    return tuple(actions)


def consensus_action(votes):
    """Modal whole-fleet next action; ties lexicographic by task ID, wait last.

    Whole-plan voting preserves unique ownership. Component-wise voting could
    assign one task to several agents and is deliberately not used here.
    """
    if not votes:
        raise ValueError('At least one scenario vote required')
    counts = Counter(tuple(v) for v in votes)
    return min(counts, key=lambda a: (-counts[a], tuple(math.inf if j < 0 else j for j in a)))


def msa_select(observation, parameters=Parameters(), check=lambda: None):
    p = parameters.validate(); unpack_maritime(observation)
    anchor = int.from_bytes(hashlib.sha256(
        f'p5-msa-v1:{p.planning_seed}:{observation.fingerprint()}'.encode()).digest()[:8], 'little')
    seeds = planning_seeds(anchor, p.scenarios)
    counts = dict(insertion_calls=0, route_evaluations=0); votes = []; hashes = []
    for seed in seeds:
        check(); scenario = sample_maritime_future(observation, seed)
        hashes.append(scenario.fingerprint())
        votes.append(scenario_plan(observation, scenario, p, counts, check))
    return dict(selected=consensus_action(votes), votes=votes, seeds=seeds,
                sample_hashes=hashes, counts=counts)


def install_routes(env, routes):
    """Execute observed plans through unchanged assign/route_estimate contracts."""
    for u, route in enumerate(routes):
        current = set(env.queues[u])
        # Incremental subsequences of the final route preserve its relative order.
        for j in route:
            if j in current:
                continue
            current.add(j); partial = [k for k in route if k in current]
            estimate = env.route_estimate(u, partial)
            if not estimate['feasible']:
                raise RuntimeError('Planned route failed live observed feasibility')
            old = env.route_estimate(u, env.queues[u])['length']
            env.assign(j, u, (estimate['length']-old, partial.index(j), estimate))
        env.queues[u] = list(route)


def run(env, method, parameters=Parameters(), check=lambda: None):
    if method not in ('cbba', 'cbba-pr', 'msa'):
        raise ValueError(method)
    p = parameters.validate(); observe_maritime(env)  # fail before mutation
    bundles = [[] for _ in range(env.cfg.uavs)]
    # Support a resumed environment; acquisition order comes from first assignment.
    for u, q in enumerate(env.queues):
        bundles[u] = sorted(q, key=lambda j: (env.assigned_at[j], j))
    previous = []; seen = set(); traces = []; seconds = 0.
    counts = dict(insertion_calls=0, route_evaluations=0)
    while not env.done:
        check()
        if env.time < env.cfg.horizon:
            start = time.perf_counter(); obs = observe_maritime(env); b = obs.as_dict()['base']
            event = bool(set(b['known'])-seen) or bool(previous); seen.update(b['known'])
            if event:
                released = []
                for u in range(env.cfg.uavs):
                    bundles[u] = [j for j in bundles[u] if j in env.queues[u]]
                    if method == 'cbba-pr':
                        tail = reset_tail(bundles[u], env.started_at, count=p.reset_count, fraction=p.reset_fraction)
                    elif method == 'msa':
                        tail = [j for j in env.queues[u] if env.started_at[j] < 0]
                    else:
                        tail = []
                    for j in tail:
                        env.withdraw(j, method+'-reset'); released.append(j)
                    bundles[u] = [j for j in bundles[u] if j not in tail]
                obs = observe_maritime(env)
                if method == 'msa':
                    result = msa_select(obs, p, check)
                    routes = [[j] if j >= 0 else list(env.queues[u]) for u, j in enumerate(result['selected'])]
                else:
                    result = cbba_allocate(obs, bundles, p, check)
                    routes = result['routes']; bundles = result['bundles']
                install_routes(env, routes)
                for key in counts:
                    counts[key] += result['counts'][key]
                traces.append(dict(time=env.time, released=released,
                    observation_sha256=obs.fingerprint(), **result))
            # Route order is part of the auction/scenario plan; never reorder it here.
            for u, q in enumerate(env.queues):
                if env.active[u] < 0 and q and q[0] in env.route_candidates(u):
                    env.select(u, q[0])
            elapsed = time.perf_counter()-start; seconds += elapsed; env.compute_seconds += elapsed
        _, previous = env.tick()
    return dict(events=len(traces), decisions=len(traces), traces=traces), dict(
        decision_seconds=seconds, counts=counts, scenario_solves=p.scenarios*len(traces) if method == 'msa' else 0)
