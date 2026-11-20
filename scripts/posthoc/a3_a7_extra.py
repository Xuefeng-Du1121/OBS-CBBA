"""A3 and A7: the two missing deployment experiments.

A3 -- a decision whose result arrives after the 25-s allocation tick cannot be
used. Rule tested: discard that decision and keep the previous allocation
(no branch is applied), then let the simulator advance. Futures are evaluated
inline, one at a time, exactly as the archived workers=1 runs did; after each
future the elapsed wall time is checked, and once it exceeds the budget the
decision is abandoned. The simulator clock does not advance during planning,
so the rule is applied to the measured decision time.

With ``--budget 1e9`` the A3 loop must reproduce the archived o_obs-cbba
episode exactly; this is the harness check run before the real batch.

A3f -- identical to A3 except that an overrun decision falls back to the
reference branch (CBBA-PR insertion of the new tasks), which costs well under
a second; this is the natural deployment rule.

A7 -- clairvoyant upper bound. The same five branches are scored on the
realized future (true positions, true arrivals, true response draws) instead
of on eight sampled futures. Not deployable; it bounds how much of the
remaining gap is attributable to forecast uncertainty.

Both arms reuse archived ``scenario.npz`` files of the main matrix. No seed is
drawn.

Usage (server):
    python -I a3_a7_extra.py --root <repo> --archive <archive> \
        --episodes <episodes.csv> --out <dir> --mode a3|a7 --workers 8
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


def oracle_scenario(observation, true_scenario):
    """True future with the planner-side (motion-stripped) config attached."""
    from obs_cbba.planning import unpack
    from obs_cbba.scenario import Scenario
    _, cfg = unpack(observation.base())
    return Scenario(true_scenario.positions.copy(), true_scenario.release.copy(),
                    true_scenario.deadline.copy(), true_scenario.response_uniform.copy(),
                    int(true_scenario.seed), cfg.dict())


def score_true_future(observation, bundles, actions, parameters, oracle):
    from obs_cbba.commitment import commitment_planning_env
    from obs_cbba import observable_cbba as production
    values = []
    for action in actions:
        env = commitment_planning_env(observation.base(), oracle, log_events=False)
        nxt = production.apply_action(env, bundles, action, parameters,
                                      deferred_commitment=True)
        value = production.cbba_suffix(env, nxt, prepared=True, parameters=parameters,
                                       deferred_commitment=True)
        if type(value) is not int:
            raise TypeError('Suffix completion count must be a Python int')
        values.append(value)
    return values


def run_scene(job):
    root, archive, batch, episode, cell, seed, mode, budget = job
    if root not in sys.path:
        sys.path.insert(0, root)
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

    scene = Scenario.load(Path(archive) / batch / 'episodes' / episode / 'scenario.npz')
    env = CommitmentEnv(Config(**scene.config), scene, log_events=False)
    parameters = Parameters()
    parameters.validate()
    actions = lat.ABLATION_BRANCHES['full']
    production.observe_maritime(env)
    bundles = production.initial_bundles(env)
    seen, previous = set(), []
    calls = overruns = futures_done = 0
    seconds = []
    selected = []
    while not env.done:
        if env.time < env.cfg.horizon:
            obs = production.observe_maritime(env)
            known = set(obs.as_dict()['base']['known'])
            event = bool(known - seen) or bool(previous)
            seen.update(known)
            if event:
                bundles = [[j for j in q if j in env.queues[u]] for u, q in enumerate(bundles)]
                started = time.perf_counter()
                calls += 1
                winner = None
                if mode == 'a7':
                    sums = score_true_future(obs, bundles, actions, parameters,
                                             oracle_scenario(obs, scene))
                    winner = max(range(len(actions)), key=lambda i: sums[i])
                else:  # a3 keeps the allocation on overrun; a3f falls back to cbba-pr
                    anchor = int.from_bytes(hashlib.sha256(
                        (production.NAMESPACE + obs.fingerprint()).encode()).digest()[:8],
                        'little')
                    done = []
                    for s in planning_seeds(anchor, 8):
                        _, _, vals, _ = lat._evaluate_future(
                            (obs, bundles, s, actions, parameters, True, 'cbba-pr', True))
                        futures_done += 1
                        done.append(vals)
                        if time.perf_counter() - started > budget:
                            break
                    late = time.perf_counter() - started > budget
                    if late:
                        overruns += 1  # result arrives after the tick
                        if mode == 'a3f':
                            winner = 0  # fall back to the cheap reference branch (cbba-pr)
                    else:
                        sums = [sum(v[i] for v in done) for i in range(len(actions))]
                        winner = max(range(len(actions)), key=lambda i: sums[i])
                seconds.append(time.perf_counter() - started)
                if winner is not None:
                    bundles = production.apply_action(env, bundles, actions[winner],
                                                      parameters, deferred_commitment=True)
                    selected.append(actions[winner])
                else:
                    selected.append('kept')
            production.dispatch(env)
        _, previous = env.tick()

    m = dict(metrics(env))
    m.update(cell=cell, seed=int(seed), batch=batch, episode_dir=episode, mode=mode,
             budget=budget, calls=calls, overruns=overruns, futures=futures_done,
             decision_seconds=float(sum(seconds)),
             max_decision=float(max(seconds) if seconds else 0.),
             selected=json.dumps(selected))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--episodes', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', choices=('a3', 'a3f', 'a7'), required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--budget', type=float, default=25.)
    ap.add_argument('--cell', default='u6_n120_high')
    ap.add_argument('--scenes', type=int, default=30)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    jobs = []
    with open(args.episodes) as fh:
        for rec in csv.DictReader(fh):
            if (rec['arm'] != 'o_obs-cbba' or rec['status'] != 'complete'
                    or rec['cell'] != args.cell):
                continue
            jobs.append((args.root, args.archive, rec['batch'], rec['episode_dir'],
                         rec['cell'], rec['seed'], args.mode, args.budget))
            if len(jobs) >= args.scenes:
                break
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f'{args.mode}{args.tag}_rows.csv'
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=mp.get_context('spawn')) as ex:
        for m in ex.map(run_scene, jobs):
            rows.append(m)
            print(json.dumps({k: m[k] for k in ('seed', 'mode', 'calls', 'overruns',
                                                'decision_seconds')}), flush=True)
            with (out / name).open('w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print('done', len(rows), flush=True)


if __name__ == '__main__':
    main()
