"""DA-CBBA with duplicate-branch grouping and a persistent process pool.

Decisions are identical to observable_cbba.py. One job owns one sampled future; original
label order, integer outcomes, seed order and first-label ties are preserved.
Use ``with EvaluationPool(N) as pool`` across select_action calls; run() owns
one such pool for its entire episode when none is supplied.
"""
from __future__ import annotations

import hashlib
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
from contextlib import nullcontext

from .baselines_p5 import Parameters
from .commitment import commitment_planning_env, delayed_suffix
from .maritime_planning import sample_maritime_future
from .planning import planning_seeds
from . import observable_cbba as production

WORKER_LIMITS = frozenset((1, 2, 4, 8))
ABLATION_BRANCHES = {
    'full': ('cbba-pr', 'cheapest', 'deadline', 'regret', 'slack'),
    'compact': ('cbba-pr', 'cheapest', 'regret'),
    'base': ('cbba-pr',),
}
ABLATION_SUFFIXES = frozenset(('cbba-pr', 'b'))


def _validate_workers(workers):
    if type(workers) is not int or workers not in WORKER_LIMITS:
        raise ValueError('workers must be one of 1, 2, 4, 8')


def _state_key(env, bundles):
    """Suffix-state bytes + ordered acquisition history, without hash collisions.

    Conservative: includes bookkeeping as well as all suffix-dependent state.
    Only compute_seconds and assignment_seconds are excluded: wall telemetry
    is written by allocation and read by metrics, never by suffix decisions.
    Group only within the same sampled future. Equal bytes imply identical
    restored state; unequal representations merely miss an optimization.
    """
    state = env.snapshot().materialize()
    del state['compute_seconds']
    del state['assignment_seconds']
    return (pickle.dumps(state, protocol=5),
            tuple(tuple(int(j) for j in q) for q in bundles))


def _signature(env, bundles):
    return dict(queues=[list(q) for q in env.queues],
                active=env.active.tolist(), bundles=[list(q) for q in bundles])


def _b_suffix(env, *, deferred=True):
    """Evaluate the b continuation, with an explicit no-withdrawal mode."""
    if deferred:
        return delayed_suffix(env, prepared=True)
    # Deferred-commitment-off keeps every queued promise and only performs the
    # ordinary cheapest controller at subsequent events.
    from .policies import Controller
    controller = Controller(env.cfg, 'cheapest')
    previous = []
    first = True
    while not env.done:
        if env.time < env.cfg.horizon:
            if not first and (previous or any(r == env.time for r in env.scenario.release)):
                controller.allocate(env)
            controller.plan(env)
        _, previous = env.tick(); first = False
    return int((env.completed_at >= 0).sum())


def _evaluate_future(job):
    if len(job) == 6:  # compatibility with the six-field job format
        observation, bundles, seed, actions, parameters, deduplicate = job
        suffix, deferred = 'cbba-pr', True
    else:
        observation, bundles, seed, actions, parameters, deduplicate, suffix, deferred = job
    scenario = sample_maritime_future(observation, seed)
    fingerprint = scenario.fingerprint()
    groups = {}; signatures = []; values = [None] * len(actions)
    for i, action in enumerate(actions):
        env = commitment_planning_env(observation.base(), scenario, log_events=False)
        next_bundles = production.apply_action(env, bundles, action, parameters,
                                               deferred_commitment=deferred)
        signatures.append(_signature(env, next_bundles))
        key = _state_key(env, next_bundles) if deduplicate else (i,)
        if key not in groups:
            groups[key] = (env, next_bundles, [])
        groups[key][2].append(i)
    for env, next_bundles, members in groups.values():
        value = (_b_suffix(env, deferred=deferred) if suffix == 'b' else
                 production.cbba_suffix(env, next_bundles, prepared=True,
                                        parameters=parameters,
                                        deferred_commitment=deferred))
        if type(value) is not int:
            raise TypeError('Suffix completion count must be a Python integer')
        for i in members:
            values[i] = value
        if scenario.fingerprint() != fingerprint:
            raise RuntimeError('Shared future mutated')
    return fingerprint, signatures, values, len(groups)


class EvaluationPool:
    """Explicit persistent spawn pool, including N=1; never retries/falls back.

    Callbacks run in the parent while awaiting results. Any error poisons this
    pool, so a caller cannot accidentally reuse partially failed evaluation.
    Pool startup is lazy and is charged to the first selection.
    """
    def __init__(self, workers=1):
        _validate_workers(workers)
        self.workers = workers
        self._executor = None
        self._failed = False
        self._closed = False

    def __enter__(self):
        return self

    def evaluate(self, jobs, check):
        if self._failed or self._closed:
            raise RuntimeError('Process pool failed or was closed')
        futures = []
        try:
            check()
            if self._executor is None:
                self._executor = ProcessPoolExecutor(max_workers=self.workers,
                    mp_context=multiprocessing.get_context('spawn'))
            futures = [self._executor.submit(_evaluate_future, job) for job in jobs]
            results = []
            # Completion order can vary; aggregation must follow submission order.
            for future in futures:
                while True:
                    check()
                    try:
                        results.append(future.result(timeout=.1))
                        break
                    except FutureTimeout:
                        if future.done():  # A worker itself raised TimeoutError.
                            raise
            return results
        except BaseException as exc:
            self._failed = True
            for future in futures:
                future.cancel()
            if not isinstance(exc, Exception):
                raise
            raise RuntimeError('Process-pool evaluation failed') from exc

    def close(self):
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __exit__(self, *args):
        self.close()


def select_action(observation, bundles, k=8, check=lambda: None, *,
                  variant='full', parameters=Parameters(), workers=1,
                  deduplicate=True, pool=None, suffix='cbba-pr',
                  deferred_commitment=True):
    _validate_workers(workers)
    if type(k) is not int or k not in (1, 4, 8):
        raise ValueError('Ablation K must be one of 1, 4, 8')
    if variant not in ABLATION_BRANCHES:
        raise ValueError(variant)
    if suffix not in ABLATION_SUFFIXES:
        raise ValueError('Unknown continuation suffix')
    if pool is not None and pool.workers != workers:
        raise ValueError('Pool worker count does not match workers')
    data = production.validate_bundles(observation, bundles)
    parameters.validate()
    anchor = int.from_bytes(hashlib.sha256(
        (production.NAMESPACE + observation.fingerprint()).encode()).digest()[:8], 'little')
    seeds = planning_seeds(anchor, k)
    actions = ABLATION_BRANCHES[variant]
    jobs = [(observation, bundles, seed, actions, parameters, deduplicate,
             suffix, bool(deferred_commitment)) for seed in seeds]
    with EvaluationPool(workers) if pool is None else nullcontext(pool) as evaluator:
        results = evaluator.evaluate(jobs, check)
    signatures = results[0][1]
    if any(result[1] != signatures for result in results):
        raise RuntimeError('Current action depends on a sampled hidden future')
    values = [[result[2][i] for result in results] for i in range(len(actions))]
    sums = [sum(v) for v in values]
    winner = max(range(len(actions)), key=lambda i: sums[i])
    row = dict(time=data['base']['time'], observation=data,
        observation_sha256=observation.fingerprint(), bundles=[list(q) for q in bundles],
        k=k, variant=variant, actions=list(actions), seeds=seeds,
        sample_hashes=[r[0] for r in results], completed=values, sums=sums,
        signatures=signatures, selected=actions[winner], suffix=suffix,
        parameters=vars(parameters), workers=workers, deduplicated=bool(deduplicate),
        unique_suffixes=sum(r[3] for r in results))
    if suffix != 'cbba-pr' or not deferred_commitment:
        row['deferred_commitment'] = bool(deferred_commitment)
    return row


def run(env, k=8, check=lambda: None, *, variant='full', parameters=Parameters(),
        on_decision=None, workers=1, deduplicate=True, pool=None, suffix='cbba-pr',
        deferred_commitment=True):
    """Production event protocol, with one persistent pool across all decisions."""
    _validate_workers(workers)
    if pool is not None and pool.workers != workers:
        raise ValueError('Pool worker count does not match workers')
    with EvaluationPool(workers) if pool is None else nullcontext(pool) as evaluator:
        return _run(env, k, check, variant=variant, parameters=parameters,
                    on_decision=on_decision, workers=workers,
                    deduplicate=deduplicate, pool=evaluator, suffix=suffix,
                    deferred_commitment=deferred_commitment)


def _run(env, k, check, *, variant, parameters, on_decision, workers, deduplicate, pool,
         suffix='cbba-pr', deferred_commitment=True):
    if variant not in ABLATION_BRANCHES or type(k) is not int or k not in (1, 4, 8):
        raise ValueError('Invalid policy')
    if suffix not in ABLATION_SUFFIXES:
        raise ValueError('Unknown continuation suffix')
    production.observe_maritime(env); parameters.validate()
    bundles = production.initial_bundles(env); seen = set(); previous = []; traces = []
    seconds = dict(selection=0., allocation=0., dispatch=0.)
    import time
    while not env.done:
        check()
        if env.time < env.cfg.horizon:
            obs = production.observe_maritime(env)
            known = set(obs.as_dict()['base']['known'])
            event = bool(known - seen) or bool(previous); seen.update(known)
            if event:
                bundles = [[j for j in q if j in env.queues[u]] for u, q in enumerate(bundles)]
                started = time.perf_counter()
                row = select_action(obs, bundles, k, check, variant=variant,
                    parameters=parameters, workers=workers, deduplicate=deduplicate, pool=pool,
                    suffix=suffix, deferred_commitment=deferred_commitment)
                row['seconds'] = time.perf_counter() - started
                seconds['selection'] += row['seconds']; env.compute_seconds += row['seconds']
                started = time.perf_counter()
                bundles = production.apply_action(env, bundles, row['selected'], parameters, check,
                                                  deferred_commitment=deferred_commitment)
                seconds['allocation'] += time.perf_counter() - started
                chosen = row['actions'].index(row['selected'])
                assert row['signatures'][chosen] == _signature(env, bundles)
                traces.append(row)
                if on_decision is not None:
                    on_decision(row)
            started = time.perf_counter(); production.dispatch(env)
            seconds['dispatch'] += time.perf_counter() - started
        _, previous = env.tick()
    return dict(events=len(traces), decisions=len(traces), traces=traces), dict(
        seconds=seconds, decision_seconds=seconds['selection'],
        sampled_suffixes=sum(r['unique_suffixes'] for r in traces), suffix=suffix,
        deferred_commitment=bool(deferred_commitment))


select_action_optimized = select_action
run_optimized = run
