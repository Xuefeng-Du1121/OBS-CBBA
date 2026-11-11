"""Observed static route-time score, lexicographic and parameter free."""
import time
import numpy as np
from .commitment import commitment_observe, repair_candidates, event_now, release_unstarted
from .observed_repair import run_observed
from .policies import Controller


def route_time_score(observation, queues):
    d = observation.as_dict(); cfg = d['config']; late = flow = 0
    points = np.asarray(d['task_positions'])
    for u, queue in enumerate(queues):
        position = np.asarray(d['position'][u]); finish = d['time']
        for j in queue:
            distance = float(np.linalg.norm(points[j]-position))
            travel = int(np.ceil(max(distance-1e-9, 0.)/cfg['speed']))
            service = (max(cfg['sensing_steps']-d['sensing'][u], 1)
                       if j == d['active'][u] else cfg['sensing_steps'])
            finish += travel+service
            late += max(finish-d['deadline'][j], 0)
            flow += finish-d['time']
            position = points[j]
    return [int(late), int(flow)]


def choose_route_time(observation, candidates):
    d = observation.as_dict(); queues = d['queues']
    base = route_time_score(observation, queues); best = base
    selected = None; scored = []
    for candidate in sorted(candidates, key=lambda m: (d['deadline'][m['task']],m['task'],m['destination'])):
        j, src, dst = candidate['task'], candidate['source'], candidate['destination']
        routes = list(map(list, queues)); routes[src].remove(j)
        routes[dst].insert(candidate['insertion'], j)
        score = route_time_score(observation, routes)
        scored.append(dict(**candidate, score=score))
        if score < best:
            best = score
            selected = {k:candidate[k] for k in ('task','source','destination')}
    return dict(base_score=base, best_score=best, selected=selected,
                candidates=scored, eligible=len(scored))


def route_time_repair(env):
    started = time.perf_counter(); observation = commitment_observe(env)
    candidates = []
    for move in repair_candidates(env):
        ins = env.insertion(move['destination'],move['task']); assert ins is not None
        candidates.append(dict(**move,insertion=int(ins[1])))
    decision = choose_route_time(observation,candidates)
    env.log('route_time_repair', method='f', selected=decision['selected'])
    move = decision['selected']
    if move is not None:
        j,u = move['task'],move['destination'];ins=env.insertion(u,j)
        assert ins is not None and env.scenario.response_uniform[env.time,j,u]>=env.cfg.loss
        env.withdraw(j,'route-time-reassignment')
        env.assign(j,u,ins)
    return dict(time=env.time,method='f',observation=observation.as_dict(),**decision,
                seconds=time.perf_counter()-started)


def run_route_time(env,method,check=lambda:None):
    if method in ('b','e2','d'):return run_observed(env,method,check)
    if method!='f':raise ValueError(method)
    controller=Controller(env.cfg,'cheapest');previous_completed=[];traces=[];events=0
    while not env.done:
        check()
        if env.time<env.cfg.horizon:
            event=event_now(env,previous_completed);events+=int(event)
            if event:release_unstarted(env)
            controller.allocate(env)
            if event:traces.append(route_time_repair(env))
            controller.plan(env)
        _,previous_completed=env.tick()
    return dict(events=events,decisions=len(traces),traces=traces)
