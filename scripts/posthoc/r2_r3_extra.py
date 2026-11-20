"""R2 and R3: forecast-content ablation and extended real-time budget replays.

All modes reuse archived ``scenario.npz`` files and the production branch
evaluation (``observable_cbba_latency._evaluate_future``). Only the content of
the sampled futures, or the rule applied when planning overruns, changes.

R2 -- what the sampled futures contain (branch set, revision, K=8 and the
CBBA-PR continuation are unchanged):
    full      drift sampled, arrivals sampled (must reproduce the archive)
    nodrift   released buoys stay at their observed positions; arrivals sampled
    noarr     drift sampled; no future arrivals
    snapshot  buoys stay put and no future arrivals: one deterministic future

R3 -- 25-s budget, futures evaluated inline one at a time:
    a3f       overrun -> the reference branch (CBBA-PR) is applied instead
    anytime   overrun -> branches are scored on the futures that finished
              inside the budget; CBBA-PR only if none finished
``--delay 1`` applies every decision one tick after the observation it was
computed from, on the state found at that tick.
``--aware`` (with ``--delay 1``) makes the planner account for its own delay:
every branch is scored as it will be executed, i.e. the current plan runs for
one more tick, the branch is applied at the next tick, and the CBBA-PR
continuation is delayed by one tick as well.

Usage (server):
    python -I r2_r3_extra.py --root <repo> --archive <archive> \
        --episodes <episodes.csv> --out <dir> --mode <mode> \
        [--module main --cell u6_n120_high --scenes 30 --workers 8]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

R2_MODES = ('full', 'nodrift', 'noarr', 'snapshot')
R3_MODES = ('a3f', 'anytime')


def make_sampler(mode):
    """Return the function that builds one sampled future for this mode."""
    from obs_cbba.maritime_planning import sample_maritime_future, unpack_maritime
    from obs_cbba.planning import sample_future
    from obs_cbba.scenario import Scenario

    def frozen(observation, seed):
        # The planner-side config carries no motion (sigma = 0), so the base
        # sampler keeps every buoy where it was observed and samples arrivals.
        d, _ = unpack_maritime(observation)
        return sample_future(observation.base(), seed,
                             arrival_rate_multiplier=d.get('arrival_rate_multiplier', 1.))

    def known_only(observation, scenario):
        n = len(observation.as_dict()['base']['known'])
        return Scenario(scenario.positions[:, :n].copy(), scenario.release[:n].copy(),
                        scenario.deadline[:n].copy(), scenario.response_uniform[:, :n].copy(),
                        scenario.seed, scenario.config)

    if mode == 'nodrift':
        return frozen
    if mode == 'noarr':
        return lambda o, s: known_only(o, sample_maritime_future(o, s))
    if mode == 'snapshot':
        return lambda o, s: known_only(o, frozen(o, s))
    return sample_maritime_future


def evaluate_future_delayed(job):
    """Score every branch under the one-tick execution delay of the harness.

    Mirrors the main loop of ``run_scene`` with ``--delay 1``: the decision is
    pending during the event tick, is applied at the start of the next tick,
    and every later event is answered by CBBA-PR one tick late.
    """
    from obs_cbba.commitment import commitment_planning_env
    from obs_cbba import observable_cbba as production
    from obs_cbba import observable_cbba_latency as lat
    observation, bundles, seed, actions, parameters = job
    scenario = lat.sample_maritime_future(observation, seed)
    groups = {}
    values = [None] * len(actions)

    def known_now(env):
        return set(production.observe_maritime(env).as_dict()['base']['known'])

    for i, action in enumerate(actions):
        env = commitment_planning_env(observation.base(), scenario, log_events=False)
        b = [list(q) for q in bundles]
        seen = known_now(env)
        production.dispatch(env)          # event tick: the old plan is executed
        _, previous = env.tick()
        if not env.done and env.time < env.cfg.horizon:
            b = [[j for j in q if j in env.queues[u]] for u, q in enumerate(b)]
            b = production.apply_action(env, b, action, parameters, deferred_commitment=True)
        key = lat._state_key(env, b)
        if key not in groups:
            groups[key] = (env, b, seen, previous, [])
        groups[key][4].append(i)
    for env, b, seen, previous, members in groups.values():
        pending = None
        first = True
        while not env.done:
            if env.time < env.cfg.horizon:
                if pending is not None and not first:
                    b = [[j for j in q if j in env.queues[u]] for u, q in enumerate(b)]
                    b = production.apply_action(env, b, pending, parameters,
                                                deferred_commitment=True)
                    pending = None
                known = known_now(env)
                if bool(known - seen) or bool(previous):
                    pending = 'cbba-pr'
                seen.update(known)
                production.dispatch(env)
            _, previous = env.tick()
            first = False
        value = int((env.completed_at >= 0).sum())
        for i in members:
            values[i] = value
    return None, None, values, len(groups)


def run_scene(job):
    (root, archive, batch, episode, cell, seed, mode, budget, delay, max_time, aware) = job
    if root not in sys.path:
        sys.path.insert(0, root)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from obs_cbba.scenario import Scenario
    from obs_cbba.config import Config
    from obs_cbba.commitment import CommitmentEnv
    from obs_cbba.metrics import metrics
    from obs_cbba.baselines_p5 import Parameters
    from obs_cbba.planning import planning_seeds
    from obs_cbba import observable_cbba as production
    from obs_cbba import observable_cbba_latency as lat

    lat.sample_maritime_future = make_sampler(mode)
    k = 1 if mode == 'snapshot' else 8
    budgeted = mode in R3_MODES

    directory = Path(archive) / batch / 'episodes' / episode
    scene = Scenario.load(directory / 'scenario.npz')
    env = CommitmentEnv(Config(**scene.config), scene, log_events=False)
    parameters = Parameters()
    parameters.validate()
    actions = lat.ABLATION_BRANCHES['full']
    production.observe_maritime(env)
    bundles = production.initial_bundles(env)
    seen, previous = set(), []
    calls = overruns = fallbacks = 0
    seconds, selected, used = [], [], []
    pending = None
    while not env.done:
        if max_time is not None and env.time >= max_time:
            break
        if env.time < env.cfg.horizon:
            if pending is not None:  # decision computed during the previous tick
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
                started = time.perf_counter()
                calls += 1
                anchor = int.from_bytes(hashlib.sha256(
                    (production.NAMESPACE + obs.fingerprint()).encode()).digest()[:8], 'little')
                done = []
                late = False
                for s in planning_seeds(anchor, 8)[:k]:
                    if aware:
                        _, _, vals, _ = evaluate_future_delayed(
                            (obs, bundles, s, actions, parameters))
                    else:
                        _, _, vals, _ = lat._evaluate_future(
                            (obs, bundles, s, actions, parameters, True, 'cbba-pr', True))
                    if budgeted and time.perf_counter() - started > budget:
                        late = True  # this future finished after the tick
                        break
                    done.append(vals)
                if late:
                    overruns += 1
                if late and (mode == 'a3f' or not done):
                    winner = 0  # CBBA-PR, the reference branch
                    fallbacks += 1
                else:
                    sums = [sum(v[i] for v in done) for i in range(len(actions))]
                    winner = max(range(len(actions)), key=lambda i: sums[i])
                used.append(len(done))
                seconds.append(time.perf_counter() - started)
                selected.append(actions[winner])
                if delay:
                    pending = actions[winner]
                else:
                    bundles = production.apply_action(env, bundles, actions[winner],
                                                      parameters, deferred_commitment=True)
            production.dispatch(env)
        _, previous = env.tick()

    row = dict(cell=cell, seed=int(seed), batch=batch, episode_dir=episode, mode=mode,
               budget=budget if budgeted else '', delay=delay, aware=aware, k=k, calls=calls,
               overruns=overruns, fallbacks=fallbacks,
               decision_seconds=float(sum(seconds)),
               max_decision=float(max(seconds) if seconds else 0.),
               revisions=sum(a != 'cbba-pr' for a in selected),
               selected=json.dumps(selected), futures_used=json.dumps(used),
               call_seconds=json.dumps([round(x, 3) for x in seconds]))
    if max_time is None:
        m = dict(metrics(env))
        released = scene.release < env.cfg.horizon
        on_time = (env.completed_at >= 0) & (env.completed_at <= scene.deadline) & released
        never = released & ~env.ever_assigned
        m.update(n_never_assigned=int(never.sum()),
                 n_assigned_not_completed=int((released & ~never & ~on_time).sum()))
        m.update(row)
        row = m
    if mode == 'full':  # harness check against the archived trace
        ref = [t['selected'] for t in json.loads((directory / 'trace.json').read_text())['traces']]
        row['archive_prefix_match'] = ref[:len(selected)] == selected and (
            max_time is not None or len(ref) == len(selected))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--episodes', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', choices=R2_MODES + R3_MODES, required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--budget', type=float, default=25.)
    ap.add_argument('--delay', type=int, choices=(0, 1), default=0)
    ap.add_argument('--aware', type=int, choices=(0, 1), default=0,
                    help='with --delay 1: score branches under the delayed execution')
    ap.add_argument('--module', default='main')
    ap.add_argument('--cell', default='u6_n120_high')
    ap.add_argument('--scenes', type=int, default=30)
    ap.add_argument('--max-time', type=int, default=None,
                    help='stop after this tick (smoke test only; no metrics)')
    args = ap.parse_args()
    if args.aware and not args.delay:
        raise SystemExit('--aware needs --delay 1')

    jobs = []
    with open(args.episodes) as fh:
        for rec in csv.DictReader(fh):
            if (rec['arm'] != 'o_obs-cbba' or rec['status'] != 'complete'
                    or rec['module'] != args.module or rec['cell'] != args.cell):
                continue
            jobs.append((args.root, args.archive, rec['batch'], rec['episode_dir'],
                         rec['cell'], rec['seed'], args.mode, args.budget, args.delay,
                         args.max_time, args.aware))
            if len(jobs) >= args.scenes:
                break
    if not jobs:
        raise SystemExit('no matching episodes')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f'{args.mode}{"_d1" if args.delay else ""}{"a" if args.aware else ""}__{args.module}__{args.cell}_rows.csv'
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=mp.get_context('spawn')) as ex:
        for m in ex.map(run_scene, jobs):
            rows.append(m)
            print(json.dumps({key: m.get(key) for key in (
                'seed', 'mode', 'TCR', 'calls', 'overruns', 'revisions',
                'decision_seconds', 'archive_prefix_match')}), flush=True)
            with (out / name).open('w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print('done', len(rows), flush=True)


if __name__ == '__main__':
    main()
