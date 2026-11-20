"""S3: fixed revision rules, the controls missing from the ablation.

Every event applies the same revision, with no evaluation:
    pr2        CBBA-PR as in the paper (reopen the last 2 unstarted tasks per bundle);
               must reproduce the archived CBBA-PR episodes exactly
    pr4        CBBA-PR reopening the last 4 unstarted tasks per bundle
    prall      CBBA-PR reopening every unstarted task (reset fraction 1)
    deadline   withdraw all unstarted tasks, re-insert in deadline order
    regret     withdraw all unstarted tasks, re-insert in regret order
    slack      withdraw all unstarted tasks, re-insert in slack order
The last three are the alternative branches of the proposed method, applied
at every event instead of being chosen.

Usage (server):
    python -I fixed_rules.py --root <repo> --archive <archive> --episodes <episodes.csv> \
        --out <dir> --rule <rule> [--module main --cells all --workers 8]
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

RULES = {'pr2': ('cbba-pr', dict(reset_count=2)),
         'pr4': ('cbba-pr', dict(reset_count=4)),
         'prall': ('cbba-pr', dict(reset_fraction=1.0)),
         'deadline': ('deadline', {}),
         'regret': ('regret', {}),
         'slack': ('slack', {})}


def run_scene(job):
    root, archive, batch, episode, cell, seed, rule = job
    if root not in sys.path:
        sys.path.insert(0, root)
    import torch
    torch.set_num_threads(1)
    from obs_cbba.scenario import Scenario
    from obs_cbba.config import Config
    from obs_cbba.commitment import CommitmentEnv
    from obs_cbba.metrics import metrics
    from obs_cbba.baselines_p5 import Parameters
    from obs_cbba import observable_cbba as production

    action, overrides = RULES[rule]
    directory = Path(archive) / batch / 'episodes' / episode
    scene = Scenario.load(directory / 'scenario.npz')
    env = CommitmentEnv(Config(**scene.config), scene, log_events=False)
    parameters = Parameters(**overrides)
    parameters.validate()
    production.observe_maritime(env)
    bundles = production.initial_bundles(env)
    seen, previous = set(), []
    calls = 0
    while not env.done:
        if env.time < env.cfg.horizon:
            obs = production.observe_maritime(env)
            known = set(obs.as_dict()['base']['known'])
            event = bool(known - seen) or bool(previous)
            seen.update(known)
            if event:
                calls += 1
                bundles = [[j for j in q if j in env.queues[u]] for u, q in enumerate(bundles)]
                bundles = production.apply_action(env, bundles, action, parameters,
                                                  deferred_commitment=True)
            production.dispatch(env)
        _, previous = env.tick()
    m = dict(metrics(env))
    released = scene.release < env.cfg.horizon
    on_time = (env.completed_at >= 0) & (env.completed_at <= scene.deadline) & released
    never = released & ~env.ever_assigned
    m.update(n_never_assigned=int(never.sum()),
             n_assigned_not_completed=int((released & ~never & ~on_time).sum()),
             cell=cell, seed=int(seed), batch=batch, episode_dir=episode, rule=rule, calls=calls)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--episodes', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--rule', choices=sorted(RULES), required=True)
    ap.add_argument('--module', default='main')
    ap.add_argument('--cells', default='all', help='comma-separated cells, or all')
    ap.add_argument('--scenes', type=int, default=30, help='scenes per cell')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    cells = None if args.cells == 'all' else set(args.cells.split(','))
    jobs, per_cell = [], {}
    with open(args.episodes) as fh:
        for rec in csv.DictReader(fh):
            if (rec['arm'] != 'o_obs-cbba' or rec['status'] != 'complete'
                    or rec['module'] != args.module or (cells and rec['cell'] not in cells)):
                continue
            if per_cell.get(rec['cell'], 0) >= args.scenes:
                continue
            per_cell[rec['cell']] = per_cell.get(rec['cell'], 0) + 1
            jobs.append((args.root, args.archive, rec['batch'], rec['episode_dir'],
                         rec['cell'], rec['seed'], args.rule))
    if not jobs:
        raise SystemExit('no matching episodes')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f'fixed_{args.rule}__{args.module}_rows.csv'
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as ex:
        for m in ex.map(run_scene, jobs):
            rows.append(m)
            print(json.dumps({k: m.get(k) for k in ('cell', 'seed', 'rule', 'TCR', 'calls')}), flush=True)
            with (out / name).open('w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print('done', len(rows), flush=True)


if __name__ == '__main__':
    main()
