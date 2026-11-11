"""Synchronous physical simulation; assignment itself does not advance time."""
import numpy as np
from .bandit import RegionalCosts
from .rewards import reward_step
from .snapshot import EnvSnapshot


class SwarmEnv:
    def __init__(self, cfg, scenario, bandit_variant="lcb-ew", log_events=True):
        self.cfg, self.scenario = cfg, scenario
        self.time = 0
        n, u = len(scenario.release), cfg.uavs
        self.position = np.full((u, 2), .5)
        self.energy = np.full(u, cfg.energy)
        self.distance = np.zeros(u)
        self.queues = [[] for _ in range(u)]
        self.owner = np.full(n, -1, dtype=int)
        self.ever_assigned = np.zeros(n, dtype=bool)
        self.assigned_at = np.full(n, -1, dtype=int)
        self.completed_at = np.full(n, -1, dtype=int)
        self.assignment_seconds = np.zeros(n)
        self.active = np.full(u, -1, dtype=int)
        self.sensing = np.zeros(u, dtype=int)
        self.planned_interaction = np.full((u, 2), .5)
        self.bandit = RegionalCosts(cfg, bandit_variant)
        self.events = []
        self.log_events = log_events
        self.deviations, self.delays = [], []
        self.history = []
        self.compute_seconds = 0.
        self.conflicts_detected = self.conflicts_resolved = 0
        self.response_decisions = self.fallback_decisions = 0
        self.reward_ledger = []
        self.reward_failed_tasks = set()
        self.last_switch_step = np.full(u, -1, dtype=int)

    def snapshot(self):
        """Capture simulator-owned state, including hidden futures and timings."""
        return EnvSnapshot.capture(self)

    def restore(self, snapshot):
        """Restore in place, dropping later attributes; do not retain array views.

        Previously held references to config/scenario/arrays are not rebound.
        Callers should read these through this environment after restoring.
        """
        if not isinstance(snapshot, EnvSnapshot):
            raise TypeError('Expected EnvSnapshot')
        state = snapshot.materialize()  # Failure leaves the live env unchanged.
        self.__dict__.clear()
        self.__dict__.update(state)
        return self

    @classmethod
    def from_snapshot(cls, snapshot):
        """Create an independent simulator branch without rerunning __init__."""
        return cls.__new__(cls).restore(snapshot)

    def log(self, kind, **kw):
        if self.log_events:
            self.events.append(dict(time=self.time, kind=kind, **kw))

    def observed_positions(self):
        return self.scenario.positions[max(0, self.time - self.cfg.observation_delay)].copy()

    def pending(self):
        s = self.scenario
        if self.time >= self.cfg.horizon:
            return []
        mask = (s.release <= self.time) & (s.deadline > self.time) & (self.owner < 0) & (self.completed_at < 0)
        return sorted(np.flatnonzero(mask).tolist(), key=lambda j: (s.deadline[j], j))

    def route_estimate(self, u, route, reset_service=False):
        """Conservative integer-tick service schedule using currently observed positions."""
        c, s = self.cfg, self.scenario
        pos = self.position[u].copy()
        points = self.observed_positions()
        time, length, consumption, late = self.time, 0., 0., 0.
        for j in route:
            dist = float(np.linalg.norm(points[j] - pos))
            travel = int(np.ceil(max(dist - 1e-9, 0.) / c.speed))
            service = (max(c.sensing_steps - int(self.sensing[u]), 1)
                       if j == self.active[u] and not reset_service else c.sensing_steps)
            time += travel + service
            length += dist
            consumption += c.flight_energy * dist + c.sensing_energy * service / c.sensing_steps
            late = max(late, time - int(s.deadline[j]))
            pos = points[j]
        home = float(np.linalg.norm(pos - .5))
        length += home
        consumption += home * c.flight_energy
        feasible = (late <= 0 and self.energy[u] - consumption >= c.reserve and
                    self.distance[u] + length <= c.max_distance and
                    time + int(np.ceil(home / c.speed)) <= c.horizon + c.return_steps)
        return dict(feasible=bool(feasible), length=length, energy=consumption,
                    finish=time, late=max(late, 0.))

    def insertion(self, u, j, geometry=None):
        # Geometry is allocation-local; it never caches feasibility or queues.
        if geometry is not None and geometry.env is not self:
            raise ValueError('Route geometry belongs to another environment')
        estimate_route = self.route_estimate if geometry is None else geometry.estimate
        old = list(self.queues[u])
        old_cost = estimate_route(u, old)['length']
        first = 1 if self.active[u] >= 0 and old else 0
        candidates = []
        for k in range(first, len(old) + 1):
            route = old[:k] + [int(j)] + old[k:]
            estimate = estimate_route(u, route)
            if estimate['feasible']:
                candidates.append((estimate['length'] - old_cost, k, estimate))
        return min(candidates, key=lambda x: (x[0], x[1])) if candidates else None

    def assign(self, j, u, insertion, compute_seconds=0.):
        assert j in self.pending() and self.owner[j] < 0
        assert insertion is not None and insertion[2]['feasible']
        self.queues[u].insert(insertion[1], int(j))
        self.owner[j] = u
        self.ever_assigned[j] = True
        self.assigned_at[j] = self.time
        self.assignment_seconds[j] = compute_seconds
        self.log('assigned', task=int(j), uav=int(u), route=list(self.queues[u]),
                 compute_seconds=float(compute_seconds), release=int(self.scenario.release[j]),
                 deadline=int(self.scenario.deadline[j]))

    def route_candidates(self, u):
        if self.active[u] >= 0:
            return []
        route = self.queues[u]
        candidates = []
        for j in route:
            order = [j] + [k for k in route if k != j]
            if self.route_estimate(u, order)['feasible']:
                candidates.append(j)
        # An updated observation can invalidate all full routes. Preserve queue and
        # attempt its first individually feasible task; never silently delete it.
        if not candidates:
            candidates = [j for j in route if self.route_estimate(u, [j])['feasible']]
        return candidates

    def select(self, u, j):
        assert self.active[u] < 0 and j in self.queues[u]
        self.queues[u].remove(j)
        self.queues[u].insert(0, j)
        self.active[u] = j
        self.sensing[u] = 0
        self.planned_interaction[u] = self.observed_positions()[j]
        self.log('route_select', uav=int(u), task=int(j),
                 interaction_point=self.planned_interaction[u].tolist())

    @staticmethod
    def closest_distance(a0, a1, b0, b1):
        r, v = a0 - b0, (a1 - a0) - (b1 - b0)
        den = float(v @ v)
        t = float(np.clip(-(r @ v) / den, 0, 1)) if den > 1e-15 else 0.
        return float(np.linalg.norm(r + t * v))

    def tick(self):
        c, s = self.cfg, self.scenario
        rewards = np.zeros(c.uavs)
        energy_before = self.energy.copy()
        raw = {k:np.zeros(c.uavs) for k in
               ('completed','energy','expired','conflicts','length','deviation','infeasible')}
        completions = []
        old = self.position.copy()
        proposed = old.copy()
        observed = self.observed_positions()
        sensing_now = np.zeros(c.uavs, dtype=bool)
        for u in range(c.uavs):
            j = self.active[u]
            if j >= 0:
                target = observed[j]
                dist = float(np.linalg.norm(target - old[u]))
                if dist <= c.sensing_radius:
                    sensing_now[u] = True
                else:
                    proposed[u] += (target - old[u]) * min(1., c.speed / dist)
                    self.planned_interaction[u] = proposed[u].copy()
            else:
                # Idle vehicles head home; future new tasks can interrupt return.
                dist = float(np.linalg.norm(old[u] - .5))
                if dist > 1e-8:
                    proposed[u] += (.5 - old[u]) * min(1., c.speed / dist)
            step_distance = float(np.linalg.norm(proposed[u] - old[u]))
            home = float(np.linalg.norm(proposed[u] - .5))
            service_cost = c.sensing_energy / c.sensing_steps if sensing_now[u] else 0.
            if (self.energy[u] - c.flight_energy * (step_distance + home) - service_cost < c.reserve or
                self.distance[u] + step_distance + home > c.max_distance):
                # Abort active service, retain the task assignment as uncompleted.
                self.active[u] = -1
                sensing_now[u] = False
                dist = float(np.linalg.norm(old[u] - .5))
                proposed[u] = old[u] + (.5 - old[u]) * min(1., c.speed / max(dist, 1e-12))
                rewards[u] -= c.reward_failure
                if j >= 0 and int(j) not in self.reward_failed_tasks:
                    self.reward_failed_tasks.add(int(j)); raw['infeasible'][u] += 1
                self.log('reserve_return', uav=u, task=int(j))
        # Stagger simultaneous departures from the shared launch zone. An exemption
        # for the entire departing segment would otherwise create overlapping UAVs
        # immediately outside the depot, followed by permanent mutual waiting.
        launchers=[u for u in range(c.uavs) if np.linalg.norm(old[u]-.5)<c.depot_radius
                   and np.linalg.norm(proposed[u]-.5)>=c.depot_radius]
        if len(launchers)>1:
            priority=sorted(launchers,key=lambda u:(u-self.time)%c.uavs)
            for u in priority[1:]:
                proposed[u]=old[u];sensing_now[u]=False
        def violates(u,v,pu,pv):
            r=old[u]-old[v];vel=(pu-old[u])-(pv-old[v])
            den=float(vel@vel)
            tau=float(np.clip(-(r@vel)/den,0,1)) if den>1e-15 else 0.
            au=old[u]+tau*(pu-old[u]);av=old[v]+tau*(pv-old[v])
            if max(np.linalg.norm(au-.5),np.linalg.norm(av-.5))<c.depot_radius:
                # The closest point of a diverging pair can be exempt
                # (both inside) while the step ends with one UAV outside and
                # the pair still closer than separation. That end state makes
                # the next step's closest point unexempt and unrepairable.
                return (max(np.linalg.norm(pu-.5),np.linalg.norm(pv-.5))>=c.depot_radius and
                        np.linalg.norm(pu-pv)<c.separation)
            return np.linalg.norm(au-av)<c.separation
        def affordable(u,point):
            step=float(np.linalg.norm(point-old[u]));home=float(np.linalg.norm(point-.5))
            return (step<=c.speed+1e-8 and self.energy[u]-c.flight_energy*(step+home)>=c.reserve and
                    self.distance[u]+step+home<=c.max_distance)
        # Deterministic yielding with a small lateral step for head-on encounters;
        # rescan all pairs because avoiding one conflict may create another.
        detected = set()
        for _ in range(c.uavs + 1):
            changed = False
            for u in range(c.uavs):
                for v in range(u + 1, c.uavs):
                    if violates(u,v,proposed[u],proposed[v]):
                        detected.add((u, v))
                        options=[(proposed[u].copy(),old[v].copy()),(old[u].copy(),proposed[v].copy())]
                        direction=old[v]-old[u]
                        perpendicular=np.array([-direction[1],direction[0]])
                        perpendicular/=max(np.linalg.norm(perpendicular),1e-12)
                        for sign in (1.,-1.):
                            shift=sign*min(c.speed/2,4*c.separation)*perpendicular
                            options.append((proposed[u].copy(),np.clip(old[v]+shift,0,1)))
                            options.append((np.clip(old[u]-shift,0,1),proposed[v].copy()))
                        options.append((old[u].copy(),old[v].copy()))
                        for pu,pv in options:
                            if affordable(u,pu) and affordable(v,pv) and not violates(u,v,pu,pv):
                                if np.any(pu!=proposed[u]) or np.any(pv!=proposed[v]):
                                    proposed[u],proposed[v]=pu,pv
                                    sensing_now[[u,v]]=False;changed=True
                                break
            if not changed:
                break
        self.conflicts_detected += len(detected)
        for u, v in detected:
            resolved = not violates(u,v,proposed[u],proposed[v])
            self.conflicts_resolved += int(resolved)
            rewards[[u, v]] -= c.reward_conflict
            raw['conflicts'][[u, v]] += 1
            self.log('conflict', pair=[u, v], resolved=bool(resolved))
        self.position = proposed
        lengths = np.linalg.norm(proposed - old, axis=1)
        self.distance += lengths
        self.energy -= c.flight_energy * lengths
        # Waiting away from depot has a small, explicitly budgeted hover cost.
        hovering = (lengths < 1e-10) & ~sensing_now & (np.linalg.norm(proposed-.5, axis=1) > 1e-8)
        affordable = self.energy - c.flight_energy * np.linalg.norm(proposed-.5, axis=1) - c.reserve
        self.energy -= np.where(hovering, np.minimum(c.hover_energy, np.maximum(affordable, 0)), 0)
        rewards -= c.reward_length * lengths
        next_time = self.time + 1
        for u in range(c.uavs):
            j = self.active[u]
            if j < 0:
                continue
            cost = min(lengths[u] / c.speed, 1.) / 3.
            deviation = 0.
            if sensing_now[u]:
                self.energy[u] -= c.sensing_energy / c.sensing_steps
                actual = s.positions[next_time, j]
                deviation = float(np.linalg.norm(actual - self.planned_interaction[u]))
                raw['deviation'][u] += deviation
                self.deviations.append(deviation)
                success = np.linalg.norm(actual - self.position[u]) <= c.sensing_radius
                self.sensing[u] = self.sensing[u] + 1 if success else 0
                rewards[u] -= c.reward_deviation * min(deviation / c.deviation_scale, 1.)
                cost += min(deviation / c.deviation_scale, 1.) / 3.
                self.log('interaction', task=int(j), uav=u, actual=actual.tolist(),
                         planned=self.planned_interaction[u].tolist(), deviation=float(deviation),
                         within_radius=bool(success))
                if self.sensing[u] >= c.sensing_steps and next_time <= s.deadline[j]:
                    self.completed_at[j] = next_time
                    self.delays.append(0.)
                    self.queues[u].remove(j)
                    self.active[u] = -1
                    rewards[u] += 1.
                    raw['completed'][u] += 1
                    completions.append(int(j))
                    self.log('completed', task=int(j), uav=u, completion_time=next_time)
            region = int(self.bandit.region(observed[j]))
            failure = next_time >= s.deadline[j] and self.completed_at[j] < 0
            cost += float(failure) / 3.
            components = [float(lengths[u]), float(deviation), float(failure)]
            cost = self.bandit.observe(u, region, cost, components=components, time=next_time)
            self.log('region_feedback', uav=u, region=region, cost=float(cost),
                     components=components, observed_time=next_time,
                     estimate=float(self.bandit.mean[region]), count=int(self.bandit.count[region]))
        self.time = next_time
        for u in range(c.uavs):
            expired = [j for j in self.queues[u] if s.deadline[j] <= self.time and self.completed_at[j] < 0]
            for j in expired:
                self.queues[u].remove(j)
                if self.active[u] == j:
                    self.active[u] = -1
                    self.sensing[u] = 0
                self.delays.append(float(c.delay_scale))  # capped missed-deadline penalty
                rewards[u] -= c.reward_failure
                raw['expired'][u] += 1
                if int(j) not in self.reward_failed_tasks:
                    self.reward_failed_tasks.add(int(j)); raw['infeasible'][u] += 1
                self.log('expired', task=int(j), uav=u)
        self.expire_public_commitments(raw, rewards)
        self.history.append([float(np.mean(rewards < 0)), len(completions) / max(c.uavs, 1),
                             float(lengths.mean() / c.speed), float(self.bandit.mean.mean())])
        self.history = self.history[-c.history:]
        # Keep observation history identical across reward protocols: changing a
        # training objective must not silently change frozen-policy observations.
        raw['energy'] = energy_before-self.energy
        raw['length'] = lengths
        entry = reward_step(c, self.time, raw)
        self.reward_ledger.append(entry)
        if c.reward_protocol == 'event-components-v1':
            rewards = np.asarray(entry['path'])
        self.log('physical_step', position=self.position.tolist(), energy=self.energy.tolist(),
                 distance=self.distance.tolist(), reward=rewards.tolist())
        assert np.isfinite(self.energy).all() and (self.energy >= -1e-7).all()
        assert len([j for q in self.queues for j in q]) == len(set(j for q in self.queues for j in q))
        return rewards, completions

    @property
    def done(self):
        return self.time >= self.cfg.horizon + self.cfg.return_steps

    def expire_public_commitments(self, raw, rewards):
        """Extension hook; legacy environments never withdraw commitments."""
        pass
