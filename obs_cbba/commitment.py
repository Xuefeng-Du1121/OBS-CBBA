"""Explicit unstarted commitments, observed admission and repair (Persist-A and helpers).

The legacy simulator remains unchanged except for a no-op expiration hook.
All reassignments preserve first-assignment accounting and forbid active preemption.
"""
import hashlib
import json
import time
import numpy as np
from .env import SwarmEnv
from .planning import observe, PlanningObservation, planning_env, planning_seeds, sample_future
from .policies import Controller
from .route_geometry import RouteGeometry


class CommitmentEnv(SwarmEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started_at = np.full(len(self.owner), -1, dtype=int)
        self.last_owner = np.full(len(self.owner), -1, dtype=int)

    def assign(self, j, u, insertion, compute_seconds=0.):
        first = self.assigned_at[j]
        first_seconds = self.assignment_seconds[j]
        super().assign(j, u, insertion, compute_seconds)
        self.last_owner[j] = u
        if first >= 0:
            self.assigned_at[j] = first
            self.assignment_seconds[j] = first_seconds

    def select(self, u, j):
        super().select(u, j)
        if self.started_at[j] < 0:
            self.started_at[j] = self.time

    def withdraw(self, j, reason):
        u = int(self.owner[j])
        if (u < 0 or j not in self.queues[u] or self.started_at[j] >= 0
                or j in self.active or self.completed_at[j] >= 0
                or self.scenario.deadline[j] <= self.time):
            raise ValueError('Only unstarted, live queued commitments can be withdrawn')
        self.queues[u].remove(j)
        self.owner[j] = -1
        self.log('commitment_release', task=int(j), uav=u, reason=reason,
                 route=list(self.queues[u]))

    def expire_public_commitments(self, raw, rewards):
        # Charge a withdrawn promise exactly once when it expires in the public pool.
        expired = np.flatnonzero(self.ever_assigned & (self.owner < 0)
            & (self.completed_at < 0) & (self.scenario.deadline == self.time))
        for j in expired:
            u = int(self.last_owner[j]); assert u >= 0
            self.delays.append(float(self.cfg.delay_scale))
            rewards[u] -= self.cfg.reward_failure
            raw['expired'][u] += 1
            if int(j) not in self.reward_failed_tasks:
                self.reward_failed_tasks.add(int(j)); raw['infeasible'][u] += 1
            self.log('commitment_expired', task=int(j), uav=u)


def commitment_observe(env):
    d = observe(env).as_dict()
    known = np.asarray(d['known'], dtype=int)
    d['commitment'] = {k: getattr(env, k)[known].tolist() for k in ('started_at', 'last_owner')}
    return PlanningObservation(json.dumps(d, sort_keys=True, allow_nan=False))


def commitment_planning_env(observation, scenario, log_events=False):
    env = planning_env(observation, scenario, log_events=log_events)
    env.__class__ = CommitmentEnv
    d = observation.as_dict(); known = np.asarray(d['known'], dtype=int)
    for key in ('started_at', 'last_owner'):
        values = np.full(len(env.owner), -1, dtype=int)
        values[known] = d['commitment'][key]
        setattr(env, key, values)
    return env


def event_now(env, previous_completed):
    return env.time < env.cfg.horizon and (bool(previous_completed)
        or bool(np.any(env.scenario.release == env.time)))


def release_unstarted(env):
    count = 0
    for queue in list(map(list, env.queues)):
        for j in queue:
            if env.started_at[j] < 0:
                env.withdraw(j, 'delayed-commitment'); count += 1
    return count


def delayed_suffix(env, check=lambda: None, prepared=False):
    """Complete b continuation; prepared skips this instant's already-applied allocation."""
    controller = Controller(env.cfg, 'cheapest')
    completed = []
    while not env.done:
        check()
        if env.time < env.cfg.horizon:
            if not prepared:
                if event_now(env, completed): release_unstarted(env)
                controller.allocate(env)
            controller.plan(env)
        _, completed = env.tick()
        prepared = False
    return int((env.completed_at >= 0).sum())


def reassign(env, task, destination):
    env.withdraw(task, 'hindsight-reassignment')
    insertion = env.insertion(destination, task)
    if insertion is None or env.scenario.response_uniform[env.time, task, destination] < env.cfg.loss:
        raise ValueError('Infeasible reassignment')
    env.assign(task, destination, insertion)


def repair_candidates(env):
    """Observed-only candidate generation; one unstarted task, one other vehicle."""
    candidates = []
    geometry = RouteGeometry(env)
    for j in sorted((j for q in env.queues for j in q if env.started_at[j] < 0),
                    key=lambda j: (env.scenario.deadline[j], j)):
        for u in range(env.cfg.uavs):
            if u == env.owner[j]: continue
            ins = env.insertion(u, j, geometry=geometry)
            if ins is not None and env.scenario.response_uniform[env.time, j, u] >= env.cfg.loss:
                candidates.append(dict(task=int(j), destination=u, source=int(env.owner[j])))
    return candidates


def hindsight_repair(env, check=lambda: None, max_candidates=12):
    started = time.perf_counter()
    candidates = repair_candidates(env)
    tested = candidates[:max_candidates]
    trace = dict(time=env.time, eligible=len(candidates), tested=len(tested),
                 selected=None, improvement=0, candidates=[])
    if tested:
        snap = env.snapshot()
        def evaluate(move):
            check(); branch = CommitmentEnv.from_snapshot(snap)
            branch.log_events = False; branch.events = []; branch.reward_ledger = []
            if move is not None: reassign(branch, move['task'], move['destination'])
            return delayed_suffix(branch, check, prepared=True)
        base = best = evaluate(None); chosen = None
        for move in tested:
            value = evaluate(move)
            trace['candidates'].append(dict(**move, completed=value))
            if value > best: best = value; chosen = move
        trace.update(base_completed=base, best_completed=best, selected=chosen,
                     improvement=best-base)
        if chosen is not None: reassign(env, chosen['task'], chosen['destination'])
    trace['seconds'] = time.perf_counter()-started
    return trace


def admission_probability(observation, task, uav, check=lambda: None):
    """K=16 exact fleet physics under frozen-queue/no-new-admission continuation.

    This estimates a specified continuation probability, not a calibrated guarantee
    under subsequent actual admissions. Seed identity depends only on observation.
    """
    identity = f'wp1d-admission-v1:{observation.fingerprint()}:{task}:{uav}'
    anchor = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'little')
    seeds = planning_seeds(anchor, 16); outcomes = []
    for seed in seeds:
        check()
        branch = commitment_planning_env(observation, sample_future(observation, seed))
        ins = branch.insertion(uav, task)
        assert ins is not None
        branch.assign(task, uav, ins)
        controller = Controller(branch.cfg, 'cheapest')
        while branch.completed_at[task] < 0 and branch.time < branch.scenario.deadline[task] and not branch.done:
            check()
            if branch.time < branch.cfg.horizon: controller.plan(branch)
            branch.tick()
        outcomes.append(bool(0 <= branch.completed_at[task] <= branch.scenario.deadline[task]))
    return dict(seeds=seeds, outcomes=outcomes, probability=sum(outcomes)/16)


def probability_allocate(env, tau, traces, attempted, check=lambda: None):
    geometry = RouteGeometry(env)
    for j in env.pending():
        if j in attempted: continue
        check()
        insertions = [env.insertion(u, j, geometry=geometry) for u in range(env.cfg.uavs)]
        valid = [u for u, ins in enumerate(insertions) if ins is not None
                 and env.scenario.response_uniform[env.time, j, u] >= env.cfg.loss]
        if not valid: continue
        u = min(valid, key=lambda u: insertions[u][0])
        started = time.perf_counter(); observation = commitment_observe(env)
        result = admission_probability(observation, j, u, check)
        elapsed = time.perf_counter()-started; env.compute_seconds += elapsed
        accepted = result['probability'] >= tau
        attempted.add(j)
        traces.append(dict(time=env.time, task=int(j), uav=u, tau=tau,
            accepted=accepted, seconds=elapsed, observation=observation.as_dict(), **result))
        env.log('probability_admission', task=int(j), uav=u, tau=tau,
                probability=result['probability'], accepted=accepted)
        if accepted: env.assign(j, u, insertions[u], elapsed)


def run_commitment(env, method, check=lambda: None):
    if method not in ('a','b','c05','c03','c07','d'): raise ValueError(method)
    controller = Controller(env.cfg, 'cheapest')
    traces = []; previous_completed = []; attempted = set(); events = 0
    while not env.done:
        check()
        if env.time < env.cfg.horizon:
            event = event_now(env, previous_completed)
            events += int(event)
            # At most one probability probe per task between successive events.
            # A rejection is reconsidered after a new arrival or completion.
            if event: attempted.clear()
            if method in ('b','d') and event: release_unstarted(env)
            if method.startswith('c'):
                probability_allocate(env, int(method[1:])/10, traces, attempted, check)
            else:
                controller.allocate(env)
            if method == 'd' and event: traces.append(hindsight_repair(env, check))
            controller.plan(env)
        _, previous_completed = env.tick()
    return dict(events=events, decisions=len(traces), traces=traces)
