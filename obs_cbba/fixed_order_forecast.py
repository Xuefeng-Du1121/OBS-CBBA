"""Defer-B, Regret-B and Fixed-F allocators, with an optional drift forecast."""
from contextlib import contextmanager
import hashlib
import json
import math
import time
import numpy as np
from .commitment import event_now, release_unstarted, repair_candidates, reassign
from .maritime_planning import observe_maritime, estimate_drift, unpack_maritime, cell_indices
from .policies import Controller
from .route_geometry import RouteGeometry
from .route_time_repair import choose_route_time

FIXED = ('b', 'regret-b', 'slack-b')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class Cost:
    def __init__(self):
        self.counts = dict(ranking_insertion_calls=0, allocation_insertion_calls=0,
                          repair_insertion_calls=0, candidate_route_evaluations=0,
                          forecast_integer_candidates=0, cache_hits=0)
        self.seconds = dict(allocation=0., planning=0., repair=0.)

    def insertion(self, env, u, j, phase, geometry=None):
        self.counts[phase+'_insertion_calls'] += 1
        first = int(env.active[u] >= 0 and bool(env.queues[u]))
        self.counts['candidate_route_evaluations'] += len(env.queues[u])+2-first
        return env.insertion(u, j, geometry=geometry)

    @contextmanager
    def repair(self, env):
        original = env.insertion
        def measured(u, j, geometry=None):
            self.counts['repair_insertion_calls'] += 1
            first = int(env.active[u] >= 0 and bool(env.queues[u]))
            self.counts['candidate_route_evaluations'] += len(env.queues[u])+2-first
            return original(u, j, geometry=geometry)
        env.insertion = measured
        try:
            yield
        finally:
            del env.insertion


class MeasuredGeometry(RouteGeometry):
    def __init__(self, env, cost):
        super().__init__(env)
        self.cost = cost

    def estimate(self, u, route):
        before = len(self.legs)+len(self.homes)
        result = super().estimate(u, route)
        self.cost.counts['cache_hits'] += len(route)+1-(len(self.legs)+len(self.homes)-before)
        return result


class ForecastGeometry:
    """Pure observation geometry: no environment, raw scenario or true config."""
    def __init__(self, observation, cost, check=lambda: None):
        data, _ = unpack_maritime(observation)
        self.base = data['base']; self.cfg = self.base['config']
        self.estimate_data = estimate_drift(observation)
        self.points = np.asarray(self.base['task_positions'], dtype=float)
        means = np.asarray([r['mean'] for r in self.estimate_data['regions']])
        self.velocity = means[cell_indices(self.points, self.estimate_data['grid'])]
        self.cost, self.check, self.cache = cost, check, {}
        self.bound = math.ceil(math.sqrt(2)/self.cfg['speed'])
        assert np.isfinite(self.velocity).all()
        assert np.all((np.asarray(self.base['position']) >= 0) &
                      (np.asarray(self.base['position']) <= 1))

    def point(self, j, tau):
        # Reflection roundoff must not perturb the exact zero-drift control.
        if not np.any(self.velocity[j]):
            return self.points[j]
        moved = self.points[j]+self.velocity[j]*tau
        return 1.-np.abs(np.mod(moved, 2.)-1.)

    def intercept(self, x, j, elapsed):
        key = (tuple(x), int(j), int(elapsed))
        if key in self.cache:
            self.cost.counts['cache_hits'] += 1
            return self.cache[key]
        for k in range(self.bound+1):
            self.check(); self.cost.counts['forecast_integer_candidates'] += 1
            point = self.point(j, elapsed+k)
            distance = float(np.linalg.norm(point-x))
            if distance <= self.cfg['speed']*k+1e-9:
                self.cache[key] = (point, distance, k)
                return self.cache[key]
        raise ValueError('No bounded forecast intercept')

    def estimate(self, u, route):
        self.cost.counts['candidate_route_evaluations'] += 1
        d, c = self.base, self.cfg
        x = np.asarray(d['position'][u]); finish = d['time']
        length = consumption = late = 0.; arrivals = []
        for j in route:
            x, distance, travel = self.intercept(x, j, finish-d['time'])
            finish += travel
            arrivals.append(dict(task=int(j), tau=int(finish-d['time']), point=x.tolist()))
            service = max(c['sensing_steps']-d['sensing'][u], 1) if j == d['active'][u] else c['sensing_steps']
            finish += service; length += distance
            consumption += c['flight_energy']*distance+c['sensing_energy']*service/c['sensing_steps']
            late = max(late, finish-d['deadline'][j])
        home = float(np.linalg.norm(x-.5)); length += home; consumption += home*c['flight_energy']
        feasible = (late <= 0 and d['energy'][u]-consumption >= c['reserve'] and
                    d['distance'][u]+length <= c['max_distance'] and
                    finish+math.ceil(home/c['speed']) <= c['horizon']+c['return_steps'])
        return dict(feasible=bool(feasible), length=length, energy=consumption,
                    finish=finish, late=max(late, 0.), arrivals=arrivals)

    def insertion(self, u, j, queue):
        self.cost.counts['allocation_insertion_calls'] += 1
        old_length = self.estimate(u, queue)['length']
        first = int(self.base['active'][u] >= 0 and bool(queue)); offers = []
        for k in range(first, len(queue)+1):
            estimate = self.estimate(u, queue[:k]+[int(j)]+queue[k:])
            if estimate['feasible']:
                offers.append((estimate['length']-old_length, k, estimate))
        return min(offers, key=lambda x:(x[0], x[1])) if offers else None


def allocate(env, order, forecast, cost, check=lambda: None):
    assert order in FIXED
    started = time.perf_counter(); pending = list(env.pending())
    geometry = MeasuredGeometry(env, cost)
    observation = observe_maritime(env) if forecast else None
    future = ForecastGeometry(observation, cost, check) if forecast else None
    if order == 'regret-b':
        def regret(j):
            offers = [cost.insertion(env, u, j, 'ranking', geometry) for u in range(env.cfg.uavs)]
            values = sorted(x[0] for x in offers if x is not None)
            return values[1]-values[0] if len(values)>1 else (1e6 if values else -1e6)
        pending.sort(key=regret, reverse=True)
    elif order == 'slack-b':
        points = env.observed_positions()
        def slack(j):
            travel = min(math.ceil(max(float(np.linalg.norm(points[j]-x))-1e-9, 0.)/env.cfg.speed)
                         for x in env.position)
            return (int(env.scenario.deadline[j])-env.time-travel-env.cfg.sensing_steps,
                    int(env.scenario.deadline[j]), j)
        pending.sort(key=slack)
    decisions = []
    for j in pending:
        check()
        offers = [(future.insertion(u, j, list(env.queues[u])) if forecast else
                   cost.insertion(env, u, j, 'allocation', geometry)) for u in range(env.cfg.uavs)]
        valid = [u for u in range(env.cfg.uavs) if offers[u] is not None and
                 env.scenario.response_uniform[env.time, j, u] >= env.cfg.loss]
        chosen = min(valid, key=lambda u:(offers[u][0], u)) if valid else None
        decisions.append(dict(task=int(j), uav=chosen,
                              offer=None if chosen is None else list(offers[chosen])))
        if chosen is not None:
            env.assign(j, chosen, offers[chosen])
    elapsed = time.perf_counter()-started
    cost.seconds['allocation'] += elapsed; env.compute_seconds += elapsed
    return dict(time=int(env.time), order=order, forecast=forecast, pending=pending, choices=decisions,
                observation_sha256=observation.fingerprint() if forecast else None,
                drift_sha256=digest(future.estimate_data) if forecast else None)


def run(env, method, best_fixed='b', check=lambda: None):
    if method not in (*FIXED, 'f', 'b_fc', 'best_fixed_fc'):
        raise ValueError(method)
    if best_fixed not in FIXED or (method == 'best_fixed_fc' and best_fixed == 'b'):
        raise ValueError('Conditional forecast must be nonduplicate')
    order = best_fixed if method == 'best_fixed_fc' else method if method in FIXED else 'b'
    forecast = method in ('b_fc', 'best_fixed_fc')
    planner = Controller(env.cfg, 'cheapest'); cost = Cost(); previous = []; events = 0
    rows = []; repairs = []
    while not env.done:
        check()
        if env.time < env.cfg.horizon:
            event = event_now(env, previous); events += int(event)
            if event:
                release_unstarted(env)
            rows.append(allocate(env, order, forecast, cost, check))
            if method == 'f' and event:
                start = time.perf_counter()
                with cost.repair(env):
                    obs = observe_maritime(env); moves = repair_candidates(env)
                    moves = [dict(**m, insertion=int(env.insertion(m['destination'], m['task'])[1])) for m in moves]
                    decision = choose_route_time(obs.base(), moves)
                    if decision['selected'] is not None:
                        move = decision['selected']; reassign(env, move['task'], move['destination'])
                env.log('maritime_decision', method='f', selected=decision['selected'])
                repairs.append(dict(time=int(env.time), observation_sha256=obs.fingerprint(), **decision))
                elapsed = time.perf_counter()-start; cost.seconds['repair'] += elapsed
            start = time.perf_counter(); planner.plan(env)
            cost.seconds['planning'] += time.perf_counter()-start
        _, previous = env.tick()
    return dict(events=events, allocations=rows, repairs=repairs), dict(seconds=cost.seconds, counts=cost.counts)
