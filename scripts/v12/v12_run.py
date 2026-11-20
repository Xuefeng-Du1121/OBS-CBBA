"""Protocol v1.2 runner: real-time configurations on the reserved seeds 40600-40959.

    python v12_run.py generate --root <formal repo> --plan <v1.1 plan.json> --store <dir>
    python v12_run.py run --root <formal repo> --store <dir> --out <dir> --arms pr,snap,... \
        [--cells all] [--workers 8]

Arms (see p5_protocol_v1_2.md):
    pr        CBBA-PR, applied at once
    snap      K=1 snapshot future, 25-s budget, CBBA-PR fallback
    pr_d1     CBBA-PR, one-tick execution delay
    snap_d1a  snapshot, one-tick delay, delay-aware scoring, 25-s budget, fallback
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXTRA = HERE.parent / 'posthoc'
FIRST_SEED, PER_CELL = 40600, 30
ARMS = {  # arm: (k, sampler mode, budget seconds or None, delay, aware, planned)
    'pr': (0, None, None, 0, 0, False),
    'snap': (1, 'snapshot', 25., 0, 0, True),
    'pr_d1': (0, None, None, 1, 0, False),
    'snap_d1a': (1, 'snapshot', 25., 1, 1, True),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(args):
    sys.path.insert(0, args.root)
    import numpy as np
    from obs_cbba.config import Config
    from obs_cbba.scenario import Scenario
    plan = json.loads(Path(args.plan).read_text())
    store = Path(args.store)
    store.mkdir(parents=True, exist_ok=True)
    index = []
    for i, cell in enumerate(plan['main']['cells']):
        for j in range(PER_CELL):
            seed = FIRST_SEED + PER_CELL * i + j
            path = store / f'{seed}.npz'
            scenario = Scenario.generate(Config(**cell['config']), seed)
            with path.open('xb') as fh:  # exclusive creation: one generation per seed
                np.savez_compressed(fh, positions=scenario.positions, release=scenario.release,
                                    deadline=scenario.deadline,
                                    response_uniform=scenario.response_uniform, seed=seed,
                                    config=json.dumps(scenario.config))
                fh.flush()
                os.fsync(fh.fileno())
            path.chmod(0o444)
            index.append(dict(seed=seed, cell=cell['id'], sha256=sha(path)))
    (store / 'index.json').write_text(json.dumps(index, indent=1))
    print('generated', len(index), 'scenes; index sha256', sha(store / 'index.json'))


def run_scene(job):
    root, store, cell, seed, expected, arm = job
    for p in (root, str(EXTRA)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    torch.set_num_threads(1)
    import r2_r3_extra as extra
    from obs_cbba.scenario import Scenario
    from obs_cbba.config import Config
    from obs_cbba.commitment import CommitmentEnv
    from obs_cbba.metrics import metrics
    from obs_cbba.baselines_p5 import Parameters
    from obs_cbba.planning import planning_seeds
    from obs_cbba import observable_cbba as production
    from obs_cbba import observable_cbba_latency as lat

    k, mode, budget, delay, aware, planned = ARMS[arm]
    path = Path(store) / f'{seed}.npz'
    if sha(path) != expected:
        raise ValueError(f'scene {seed} changed')
    if planned:
        lat.sample_maritime_future = extra.make_sampler(mode)
    scene = Scenario.load(path)
    env = CommitmentEnv(Config(**scene.config), scene, log_events=False)
    parameters = Parameters()
    parameters.validate()
    actions = lat.ABLATION_BRANCHES['full']
    production.observe_maritime(env)
    bundles = production.initial_bundles(env)
    seen, previous, pending = set(), [], None
    calls = fallbacks = 0
    seconds, selected = [], []
    while not env.done:
        if env.time < env.cfg.horizon:
            if pending is not None:
                bundles = [[j for j in q if j in env.queues[u]] for u, q in enumerate(bundles)]
                bundles = production.apply_action(env, bundles, pending, parameters,
                                                  deferred_commitment=True)
                pending = None
            obs = production.observe_maritime(env)
            known = set(obs.as_dict()['base']['known'])
            event = bool(known - seen) or bool(previous)
            seen.update(known)
            if event:
                bundles = [[j for j in q if j in env.queues[u]] for u, q in enumerate(bundles)]
                calls += 1
                started = time.perf_counter()
                winner = 'cbba-pr'
                if planned:
                    anchor = int.from_bytes(hashlib.sha256(
                        (production.NAMESPACE + obs.fingerprint()).encode()).digest()[:8], 'little')
                    done = []
                    for s in planning_seeds(anchor, 8)[:k]:
                        if aware:
                            _, _, vals, _ = extra.evaluate_future_delayed(
                                (obs, bundles, s, actions, parameters))
                        else:
                            _, _, vals, _ = lat._evaluate_future(
                                (obs, bundles, s, actions, parameters, True, 'cbba-pr', True))
                        done.append(vals)
                    late = budget is not None and time.perf_counter() - started > budget
                    if late:
                        fallbacks += 1
                    else:
                        sums = [sum(v[i] for v in done) for i in range(len(actions))]
                        winner = actions[max(range(len(actions)), key=lambda i: sums[i])]
                    seconds.append(time.perf_counter() - started)
                selected.append(winner)
                if delay:
                    pending = winner
                else:
                    bundles = production.apply_action(env, bundles, winner, parameters,
                                                      deferred_commitment=True)
            production.dispatch(env)
        _, previous = env.tick()
    m = dict(metrics(env))
    released = scene.release < env.cfg.horizon
    on_time = (env.completed_at >= 0) & (env.completed_at <= scene.deadline) & released
    never = released & ~env.ever_assigned
    m.update(n_never_assigned=int(never.sum()),
             n_assigned_not_completed=int((released & ~never & ~on_time).sum()),
             cell=cell, seed=int(seed), arm=arm, calls=calls, fallbacks=fallbacks,
             revisions=sum(a != 'cbba-pr' for a in selected),
             decision_seconds=float(sum(seconds)), max_decision=float(max(seconds, default=0.)),
             call_seconds=json.dumps([round(x, 3) for x in seconds]),
             selected=json.dumps(selected), host=platform.node(), cpu=cpu_name())
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
    index = json.loads((store / 'index.json').read_text())
    cells = None if args.cells == 'all' else set(args.cells.split(','))
    for arm in args.arms.split(','):
        if arm not in ARMS:
            raise SystemExit(f'unknown arm {arm}')
        jobs = [(args.root, str(store), r['cell'], r['seed'], r['sha256'], arm)
                for r in index if cells is None or r['cell'] in cells]
        name = out / f'v12_{arm}__{args.tag}_rows.csv'
        if name.exists():
            raise SystemExit(f'{name} exists; v1.2 forbids re-runs')
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as ex:
            for m in ex.map(run_scene, jobs):
                rows.append(m)
                print(json.dumps({k: m.get(k) for k in ('arm', 'cell', 'seed', 'TCR', 'calls',
                                                        'fallbacks', 'max_decision')}), flush=True)
                tmp = name.with_suffix('.partial')
                with tmp.open('w', newline='') as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
        tmp.rename(name)
        print('done', arm, len(rows), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('generate')
    g.add_argument('--root', required=True)
    g.add_argument('--plan', required=True)
    g.add_argument('--store', required=True)
    r = sub.add_parser('run')
    r.add_argument('--root', required=True)
    r.add_argument('--store', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--arms', required=True)
    r.add_argument('--cells', default='all')
    r.add_argument('--tag', default='all')
    r.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    generate(args) if args.cmd == 'generate' else run(args)


if __name__ == '__main__':
    main()
