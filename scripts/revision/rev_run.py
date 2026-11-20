"""Revision experiments (reviewer round, 2026-11-19). Post hoc, no change to the planner.

    python rev_run.py gen   --root <repo> --scenes <scenes.json> --digests <csv> --store <dir> --module main
    python rev_run.py gen   --root <repo> --scale --store <dir>
    python rev_run.py run   --root <repo> --store <dir> --out <dir> --arms a,b,... [--workers 8]

Arms (all use the snapshot future of the paper, K=1, the CBBA-PR continuation,
a 25-s budget per call and the CBBA-PR fallback, unless stated):
    pr        CBBA-PR at every event (reference; must equal the archived rows)
    snap      paper set {PR2, deadline, regret, slack}; must equal the archived snapshot rows
    amount    {PR2, PR4, PRall}: only the amount reopened varies, CBBA auction re-insertion
    order     {PRall, deadline, regret, slack}: amount fixed (all unstarted), only the insertion varies
    union     {PR2, PR4, PRall, deadline, regret, slack}
    static    paper set, but each branch scored by its immediate plan (tasks owned, then length),
              no continuation
    ms32      PR2 and 32 full revisions (deadline, regret, slack and 29 random orders),
              scored like static: equal-or-larger compute, no rollout
    pen05     paper set, rollout value = completed - 0.5 * (assigned, not completed)
    pen10     same with weight 1
    apr       assignment-preserving replanning adapter (no scoring)
    msa64     MSA with 64 scenarios instead of 8
    snapapr   paper set plus the APR revision as a fifth candidate (same continuation)
    ms128     PR2 and 128 full revisions scored like static (compute matched to snap)
Every planned call also records the number of route evaluations (machine-independent cost).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import platform
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXTRA = HERE.parent / 'posthoc'
BUDGET = 25.

PAPER = ('pr2', 'deadline', 'regret', 'slack')
ARMS = {
    'pr': dict(kind='fixed'),
    'snap': dict(kind='rollout', branches=PAPER),
    'amount': dict(kind='rollout', branches=('pr2', 'pr4', 'prall')),
    'order': dict(kind='rollout', branches=('prall', 'deadline', 'regret', 'slack')),
    'union': dict(kind='rollout', branches=('pr2', 'pr4', 'prall', 'deadline', 'regret', 'slack')),
    'static': dict(kind='static', branches=PAPER),
    'ms32': dict(kind='static', branches=PAPER, random_orders=29),
    'pen05': dict(kind='rollout', branches=PAPER, penalty=.5),
    'pen10': dict(kind='rollout', branches=PAPER, penalty=1.),
    'apr': dict(kind='apr'),
    'msa8': dict(kind='msa', scenarios=8),
    'msa64': dict(kind='msa', scenarios=64),
    'snapapr': dict(kind='rollout', branches=PAPER + ('apr',)),
    'ms128': dict(kind='static', branches=PAPER, random_orders=125),
}
# Scale cells (new seeds, never used before): U in {10, 12}, high arrivals.
SCALE = [('u10_n200_high', 10, 200, 41000), ('u12_n240_high', 12, 240, 41030),
         ('u12_n360_high', 12, 360, 41060)]
PER_CELL = 30


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scene_digest(scene):
    import numpy as np
    h = hashlib.sha256()
    for name in ('positions', 'release', 'deadline', 'response_uniform'):
        a = np.asarray(getattr(scene, name))
        h.update(f'{name}:{a.dtype.str}:{a.shape};'.encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- scenes
def generate(args):
    sys.path.insert(0, args.root)
    import numpy as np
    from obs_cbba.config import Config
    from obs_cbba.scenario import Scenario
    table = json.loads(Path(args.scenes).read_text())
    store = Path(args.store); store.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.scale:
        base = None
        for cell in table['cells']:
            if cell['module'] == 'main' and cell['cell'] == 'u6_n180_high':
                g = cell['groups'][0]; base = dict(table['base_config']); base.update(g['overrides'])
                for k in g.get('drop', []): base.pop(k)
        arrivals = 60  # high intensity: 10 per slot, six slots
        for name, u, n, first in SCALE:
            cfg = dict(base, uavs=u, tasks=n - arrivals)
            jobs += [(name, first + i, cfg, None) for i in range(PER_CELL)]
    else:
        digests = {int(r['seed']): r['sha256'] for r in csv.DictReader(open(args.digests))}
        for cell in table['cells']:
            if cell['module'] != args.module:
                continue
            for g in cell['groups']:
                cfg = dict(table['base_config']); cfg.update(g['overrides'])
                for k in g.get('drop', []): cfg.pop(k)
                for s in range(g['seeds'][0], g['seeds'][1] + 1):
                    jobs.append((cell['cell'], s, cfg, digests[s]))
    index = []
    for cell, seed, cfg, expected in jobs:
        scene = Scenario.generate(Config(**cfg), seed)
        d = scene_digest(scene)
        if expected is not None and d != expected:
            raise SystemExit(f'digest mismatch for seed {seed}')
        path = store / f'{seed}.npz'
        if not path.exists():
            np.savez_compressed(path, positions=scene.positions, release=scene.release,
                                deadline=scene.deadline, response_uniform=scene.response_uniform,
                                seed=seed, config=json.dumps(scene.config))
        index.append(dict(seed=seed, cell=cell, digest=d, sha256=sha(path)))
    name = 'index_scale.json' if args.scale else f'index_{args.module}.json'
    (store / name).write_text(json.dumps(index, indent=1))
    print('scenes', len(index), 'digest-checked' if not args.scale else 'new', name)


# ---------------------------------------------------------------- planner pieces
class Counter:
    n = 0


def instrument():
    """Count route evaluations (each feasibility/length estimate of one route)."""
    from obs_cbba.env import SwarmEnv
    from obs_cbba.route_geometry import RouteGeometry
    from obs_cbba import baselines_p5
    for cls in (SwarmEnv, RouteGeometry, baselines_p5.ScenarioGeometry):
        f = cls.__dict__.get('route_estimate' if cls is SwarmEnv else 'estimate')
        name = 'route_estimate' if cls is SwarmEnv else 'estimate'

        def wrap(f):
            def g(*a, **k):
                Counter.n += 1
                return f(*a, **k)
            return g
        setattr(cls, name, wrap(f))


def params_of(branch):
    from obs_cbba.baselines_p5 import Parameters
    if branch == 'pr4':
        return 'cbba-pr', Parameters(reset_count=4)
    if branch == 'prall':
        return 'cbba-pr', Parameters(reset_fraction=1.0)
    if branch == 'pr2':
        return 'cbba-pr', Parameters()
    return branch, Parameters()


def allocate_order(env, order, check=lambda: None):
    """Greedy cheapest insertion in a given task order (same rule as the paper's branches)."""
    from obs_cbba.route_geometry import RouteGeometry
    pending = set(env.pending())
    order = [j for j in order if j in pending]
    geometry = RouteGeometry(env) if order else None
    for j in order:
        insertions = [env.insertion(u, j, geometry=geometry) for u in range(env.cfg.uavs)]
        responding = (env.scenario.response_uniform[env.time, j] >= env.cfg.loss).tolist()
        valid = [u for u in range(env.cfg.uavs) if insertions[u] is not None and responding[u]]
        if valid:
            u = min(valid, key=lambda u: insertions[u][0])
            env.assign(j, u, insertions[u], 0.)


def apply_branch(env, bundles, branch):
    """Apply one revision; returns the new acquisition bundles (as production.apply_action)."""
    from obs_cbba import observable_cbba as production
    from obs_cbba.commitment import release_unstarted
    from obs_cbba.policies import Controller
    if isinstance(branch, tuple):  # ('order', [task ids]) -> full revision in that order
        release_unstarted(env)
        bundles = [[j for j in q if j in env.queues[u]] for u, q in enumerate(bundles)]
        original = env.assign

        def record(j, u, insertion, compute_seconds=0.):
            original(j, u, insertion, compute_seconds); bundles[u].append(int(j))
        env.assign = record
        try:
            allocate_order(env, branch[1])
        finally:
            del env.assign
        Controller(env.cfg, 'cheapest').plan(env)
        return bundles
    if branch == 'apr':
        return apr_event(env, bundles)
    action, p = params_of(branch)
    if action == 'deadline':
        action = 'cheapest'  # identical order (env.pending is EDF); label used by the archive
    return production.apply_action(env, bundles, action, p, deferred_commitment=True)


def snapshot_future(obs):
    from obs_cbba import observable_cbba as production
    from obs_cbba.planning import planning_seeds
    sys.path.insert(0, str(EXTRA))
    import r2_r3_extra as extra
    anchor = int.from_bytes(hashlib.sha256(
        (production.NAMESPACE + obs.fingerprint()).encode()).digest()[:8], 'little')
    seed = planning_seeds(anchor, 8)[0]
    return extra.make_sampler('snapshot')(obs, seed), anchor


def penalized(env, completed, weight):
    if not weight:
        return completed
    s = env.scenario
    released = s.release < env.cfg.horizon
    failed = env.ever_assigned & (env.completed_at < 0) & released
    return completed - weight * int(failed.sum())


def score_rollout(obs, bundles, branches, weight=0.):
    from obs_cbba.commitment import commitment_planning_env
    from obs_cbba import observable_cbba as production
    from obs_cbba import observable_cbba_latency as lat
    scenario, _ = snapshot_future(obs)
    values, groups = [], {}
    for b in branches:
        env = commitment_planning_env(obs.base(), scenario, log_events=False)
        nb = apply_branch(env, bundles, b)
        key = lat._state_key(env, nb)
        if key not in groups:
            done = production.cbba_suffix(env, nb, prepared=True)
            groups[key] = penalized(env, done, weight)
        values.append(groups[key])
    return values


def plan_value(env):
    owned = sum(len(q) for q in env.queues)
    length = sum(env.route_estimate(u, env.queues[u])['length'] for u in range(env.cfg.uavs))
    return (owned, -length)


def score_static(obs, bundles, branches):
    from obs_cbba.commitment import commitment_planning_env
    scenario, _ = snapshot_future(obs)
    values = []
    for b in branches:
        env = commitment_planning_env(obs.base(), scenario, log_events=False)
        apply_branch(env, bundles, b)
        values.append(plan_value(env))
    return values


def random_orders(obs, env, count):
    from obs_cbba.commitment import release_unstarted, commitment_planning_env
    pending_ids = None
    scenario, anchor = snapshot_future(obs)
    probe = commitment_planning_env(obs.base(), scenario, log_events=False)
    release_unstarted(probe)
    pending_ids = list(probe.pending())
    rng = random.Random(anchor)
    out = []
    for _ in range(count):
        order = pending_ids[:]
        rng.shuffle(order)
        out.append(('order', order))
    return out


# ---------------------------------------------------------------- APR adapter
def apr_event(env, bundles):
    """Assignment-preserving replanning (after Hwang et al., 2026), adapted to this simulator.

    1. Unaffected commitments (owned, unstarted, on a route that is still feasible on the
       current observation) are protected with their relative order. A route that became
       infeasible releases unstarted tasks from its tail until it is feasible again
       (directly affected tasks).
    2. Repair within the protected scope: CBBA auction of the released and new tasks,
       existing ownership frozen.
    3. Escalation: for each task still unassigned (deadline order), release protected
       unstarted tasks of one UAV one at a time, from the route tail, until the task fits;
       re-insert the released tasks by cheapest insertion. The step is kept only if the
       number of owned tasks increases; otherwise it is undone.
    """
    from obs_cbba import observable_cbba as production
    from obs_cbba.baselines_p5 import Parameters, cbba_allocate, install_routes
    from obs_cbba.maritime_planning import observe_maritime
    # 1. directly affected: infeasible routes shed unstarted tail tasks
    for u in range(env.cfg.uavs):
        while env.queues[u] and not env.route_estimate(u, env.queues[u])['feasible']:
            tail = [j for j in env.queues[u] if env.started_at[j] < 0 and j != env.active[u]
                    and env.scenario.deadline[j] > env.time]
            if not tail:
                break
            env.withdraw(tail[-1], 'apr-affected')
    bundles = [[j for j in q if j in env.queues[u]] for u, q in enumerate(bundles)]
    # 2. repair inside the protected scope (CBBA, no reset)
    result = cbba_allocate(observe_maritime(env), bundles, Parameters())
    install_routes(env, result['routes']); bundles = result['bundles']
    # 3. escalation, one protected task at a time
    for j in list(env.pending()):
        if j not in env.pending():
            continue
        for u in sorted(range(env.cfg.uavs),
                        key=lambda u: float(((env.observed_positions()[j] - env.position[u]) ** 2).sum())):
            if env.scenario.response_uniform[env.time, j][u] < env.cfg.loss:
                continue
            unstarted = [k for k in env.queues[u] if env.started_at[k] < 0 and k != env.active[u]
                         and env.scenario.deadline[k] > env.time]
            if not unstarted:
                continue
            snap = env.snapshot(); before = sum(len(q) for q in env.queues); kept = False
            for r in range(1, len(unstarted) + 1):
                env.withdraw(unstarted[-r], 'apr-escalate')
                ins = env.insertion(u, j)
                if ins is None:
                    continue
                env.assign(j, u, ins, 0.)
                allocate_order(env, [k for k in env.pending() if k in unstarted[-r:]])
                if sum(len(q) for q in env.queues) > before:
                    kept = True
                break
            if kept:
                break
            env.restore(snap)
        bundles = [[k for k in q] for q in env.queues]
    production.dispatch(env)
    return [sorted(q, key=lambda k: (env.assigned_at[k], k)) for q in env.queues]


# ---------------------------------------------------------------- episode
def run_scene(job):
    root, store, cell, seed, expected, arm = job
    for p in (root, str(EXTRA)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    torch.set_num_threads(1)
    from obs_cbba.scenario import Scenario
    from obs_cbba.config import Config
    from obs_cbba.commitment import CommitmentEnv
    from obs_cbba.metrics import metrics
    from obs_cbba import observable_cbba as production
    from obs_cbba import baselines_p5
    instrument()
    spec = ARMS[arm]
    path = Path(store) / f'{seed}.npz'
    if sha(path) != expected:
        raise ValueError(f'scene {seed} changed')
    scene = Scenario.load(path)
    env = CommitmentEnv(Config(**scene.config), scene, log_events=False)
    extra_row = {}
    if spec['kind'] == 'msa':
        Counter.n = 0
        t0 = time.perf_counter()
        _, info = baselines_p5.run(env, 'msa', baselines_p5.Parameters(scenarios=spec['scenarios']))
        events = info.get('scenario_solves', 0) // spec['scenarios']
        extra_row = dict(calls=events, decision_seconds=time.perf_counter() - t0,
                         route_evals=Counter.n, selected='[]', call_seconds='[]', call_evals='[]',
                         fallbacks=0, revisions=0, max_decision=0., times='[]')
    else:
        production.observe_maritime(env)
        bundles = production.initial_bundles(env)
        seen, previous = set(), []
        calls = fallbacks = 0
        seconds, selected, evals, times = [], [], [], []
        while not env.done:
            if env.time < env.cfg.horizon:
                obs = production.observe_maritime(env)
                known = set(obs.as_dict()['base']['known'])
                event = bool(known - seen) or bool(previous)
                seen.update(known)
                if event:
                    bundles = [[j for j in q if j in env.queues[u]] for u, q in enumerate(bundles)]
                    calls += 1
                    if spec['kind'] == 'fixed':
                        bundles = apply_branch(env, bundles, 'pr2')
                    elif spec['kind'] == 'apr':
                        Counter.n = 0; t0 = time.perf_counter()
                        bundles = apr_event(env, bundles)
                        seconds.append(time.perf_counter() - t0); evals.append(Counter.n)
                    else:
                        Counter.n = 0; t0 = time.perf_counter()
                        branches = list(spec['branches'])
                        if spec.get('random_orders'):
                            branches += random_orders(obs, env, spec['random_orders'])
                        if spec['kind'] == 'rollout':
                            vals = score_rollout(obs, bundles, branches, spec.get('penalty', 0.))
                        else:
                            vals = score_static(obs, bundles, branches)
                        w = max(range(len(branches)), key=lambda i: vals[i])
                        late = time.perf_counter() - t0 > BUDGET
                        if late:
                            w = 0; fallbacks += 1
                        seconds.append(time.perf_counter() - t0); evals.append(Counter.n)
                        b = branches[w]
                        selected.append(b if isinstance(b, str) else 'random')
                        times.append(int(env.time))
                        bundles = apply_branch(env, bundles, b)
                production.dispatch(env)
            _, previous = env.tick()
        extra_row = dict(calls=calls, fallbacks=fallbacks,
                         revisions=sum(a != 'pr2' for a in selected),
                         decision_seconds=float(sum(seconds)),
                         max_decision=float(max(seconds, default=0.)),
                         route_evals=int(sum(evals)),
                         call_seconds=json.dumps([round(x, 3) for x in seconds]),
                         call_evals=json.dumps(evals), selected=json.dumps(selected),
                         times=json.dumps(times))
    m = dict(metrics(env))
    released = scene.release < env.cfg.horizon
    on_time = (env.completed_at >= 0) & (env.completed_at <= scene.deadline) & released
    never = released & ~env.ever_assigned
    m.update(n_never_assigned=int(never.sum()),
             n_assigned_not_completed=int((released & ~never & ~on_time).sum()),
             n_released=int(released.sum()), n_on_time=int(on_time.sum()),
             cell=cell, seed=int(seed), arm=arm, host=platform.node(), cpu=cpu_name(), **extra_row)
    return m


def cpu_name():
    try:
        if sys.platform == 'darwin':
            return subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def run(args):
    store, out = Path(args.store), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((store / args.index).read_text())
    if args.seeds:
        lo, hi = map(int, args.seeds.split('-'))
        index = [r for r in index if lo <= r['seed'] <= hi]
    if args.cells != 'all':
        keep = set(args.cells.split(','))
        index = [r for r in index if r['cell'] in keep]
    for arm in args.arms.split(','):
        if arm not in ARMS:
            raise SystemExit(f'unknown arm {arm}')
        jobs = [(args.root, str(store), r['cell'], r['seed'], r['sha256'], arm) for r in index]
        name = out / f'{arm}__{args.tag}_rows.csv'
        if name.exists():
            print('skip existing', name, flush=True)
            continue
        rows = []
        tmp = name.with_suffix('.partial')
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as ex:
            for m in ex.map(run_scene, jobs):
                rows.append(m)
                print(json.dumps({k: m.get(k) for k in ('arm', 'cell', 'seed', 'TCR', 'calls',
                                                        'fallbacks', 'max_decision')}), flush=True)
                with tmp.open('w', newline='') as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    w.writeheader(); w.writerows(rows)
        tmp.rename(name)
        print('done', arm, len(rows), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gen')
    g.add_argument('--root', required=True); g.add_argument('--scenes', required=True)
    g.add_argument('--digests'); g.add_argument('--store', required=True)
    g.add_argument('--module', default='main'); g.add_argument('--scale', action='store_true')
    r = sub.add_parser('run')
    r.add_argument('--root', required=True); r.add_argument('--store', required=True)
    r.add_argument('--out', required=True); r.add_argument('--arms', required=True)
    r.add_argument('--index', default='index_main.json'); r.add_argument('--cells', default='all')
    r.add_argument('--seeds', default=''); r.add_argument('--tag', default='main')
    r.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    generate(args) if args.cmd == 'gen' else run(args)


if __name__ == '__main__':
    main()
