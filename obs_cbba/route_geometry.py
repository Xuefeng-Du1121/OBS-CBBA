"""Allocation-local memoization; scalar arithmetic matches route_estimate.

Only Controller.allocate owns this cache. Assignments change queues, not UAV or
observed task positions. No cache is stored in an env, snapshot or controller.
"""
import numpy as np


class RouteGeometry:
    def __init__(self, env):
        self.env = env
        self.time = env.time
        self.points = env.observed_positions()
        self.starts = env.position.copy()
        self.legs = {}
        self.homes = {}

    def leg(self, u, previous, task):
        source = -int(u)-1 if previous is None else int(previous)
        key = (source, int(task))
        value = self.legs.get(key)
        if value is None:
            pos = self.starts[u] if previous is None else self.points[previous]
            distance = float(np.linalg.norm(self.points[task]-pos))
            travel = int(np.ceil(max(distance-1e-9, 0.)/self.env.cfg.speed))
            value = (distance, travel)
            self.legs[key] = value
        return value

    def home(self, u, previous):
        key = -int(u)-1 if previous is None else int(previous)
        value = self.homes.get(key)
        if value is None:
            pos = self.starts[u] if previous is None else self.points[previous]
            distance = float(np.linalg.norm(pos-.5))
            value = (distance, int(np.ceil(distance/self.env.cfg.speed)))
            self.homes[key] = value
        return value

    def estimate(self, u, route):
        env = self.env
        if env.time != self.time:
            raise RuntimeError('Route geometry cannot cross physical steps')
        c, s = env.cfg, env.scenario
        time, length, consumption, late = env.time, 0., 0., 0.
        previous = None
        for j in route:
            dist, travel = self.leg(u, previous, j)
            service = (max(c.sensing_steps-int(env.sensing[u]), 1)
                       if j == env.active[u] else c.sensing_steps)
            time += travel + service
            length += dist
            consumption += c.flight_energy*dist + c.sensing_energy*service/c.sensing_steps
            late = max(late, time-int(s.deadline[j]))
            previous = j
        home, travel_home = self.home(u, previous)
        length += home
        consumption += home*c.flight_energy
        feasible = (late <= 0 and env.energy[u]-consumption >= c.reserve and
                    env.distance[u]+length <= c.max_distance and
                    time+travel_home <= c.horizon+c.return_steps)
        return dict(feasible=bool(feasible), length=length, energy=consumption,
                    finish=time, late=max(late, 0.))
