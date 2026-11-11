"""Run one method on one scene of the paper and print its metrics.

    python run_episode.py --list                         # cells and seed ranges
    python run_episode.py --seed 40030 --method cbba-pr
    python run_episode.py --seed 40030 --method da-cbba --workers 4

A scene is generated from its seed and the configuration in data/scenes.json,
and its arrays are checked against data/scene_digests.csv. When the same
(scene, method) pair is in results/episodes.csv, the archived row is printed
next to the new result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

# name used in the paper -> name used in the code and in results/episodes.csv
METHODS = {
    'persist-a': 'a', 'defer-b': 'b', 'regret-b': 'regret-b', 'fixed-f': 'f',
    'cbba': 'cbba', 'cbba-pr': 'cbba-pr', 'msa': 'msa', 'obs-la': 'o_obs',
    'da-cbba': 'o_obs-cbba', 'defer-b-forecast': 'b_fc',
}
# ablation variants of DA-CBBA: K, branch set, continuation, deferred commitment
VARIANTS = {
    'o_obs-cbba': dict(k=8, variant='full', suffix='cbba-pr', deferred=True),
    'full_k8': dict(k=8, variant='full', suffix='cbba-pr', deferred=True),
    'full_k4': dict(k=4, variant='full', suffix='cbba-pr', deferred=True),
    'full_k1': dict(k=1, variant='full', suffix='cbba-pr', deferred=True),
    'compact_k8': dict(k=8, variant='compact', suffix='cbba-pr', deferred=True),
    'suffix_b': dict(k=8, variant='full', suffix='b', deferred=True),
    'commitment_off': dict(k=8, variant='full', suffix='cbba-pr', deferred=False),
}
TRIGGER_THRESHOLDS = ('0.05', '0.12456826310046251', '0.25')
TRIGGER_SHA256 = '5a0d0bf090e0d3d0d94b92d1e41afb355959ea5f369f0fd777da2c42d1e5253f'


def scene_table():
    return json.loads((ROOT / 'data' / 'scenes.json').read_text())


def scene_config(seed):
    """Return (module, cell, configuration) of a scene seed."""
    table = scene_table()
    for cell in table['cells']:
        for group in cell['groups']:
            if group['seeds'][0] <= seed <= group['seeds'][1]:
                cfg = dict(table['base_config'])
                cfg.update(group['overrides'])
                for key in group.get('drop', []):
                    cfg.pop(key)
                return cell['module'], cell['cell'], cfg
    raise SystemExit(f'seed {seed} is not a scene of the paper (see --list)')


def make_scene(seed):
    from obs_cbba.config import Config
    from obs_cbba.scenario import Scenario
    module, cell, cfg = scene_config(seed)
    if cfg['mobility'] != 'real-current':
        return module, cell, Scenario.generate(Config(**cfg), seed)
    # HYCOM cells: buoys are advected through an archived surface-current panel.
    from obs_cbba.arrivals import deadlines, release_schedule
    from obs_cbba.real_drift import RealDriftParameters, advect_positions, load_hycom
    drift = dict(cfg['real_drift'])
    drift['file'] = str(ROOT / 'data' / 'hycom' / drift['file'])
    c = Config(**dict(cfg, real_drift=drift)).validate()
    p = RealDriftParameters.from_config(c)
    field = load_hycom(p.file, p.center_lon, p.sha256)
    initial_rng, _, comm_rng = [np.random.default_rng(s)
                                for s in np.random.SeedSequence(seed).spawn(3)]
    releases = release_schedule(c, np.random.default_rng(np.random.SeedSequence([0x41525231, seed])))
    steps = c.horizon + c.return_steps + 1
    initial = initial_rng.uniform(0, 1, (len(releases), 2))
    positions = advect_positions(field, initial, releases, steps, p)
    uniform = comm_rng.random((steps, len(releases), c.uavs)).astype(np.float32)
    return module, cell, Scenario(positions, releases, deadlines(c, releases), uniform, seed, c.dict())


def scene_digest(scene):
    """SHA-256 over the four scene arrays (dtype, shape and bytes)."""
    h = hashlib.sha256()
    for name in ('positions', 'release', 'deadline', 'response_uniform'):
        a = np.asarray(getattr(scene, name))
        h.update(f'{name}:{a.dtype.str}:{a.shape};'.encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def archived_digest(seed):
    with open(ROOT / 'data' / 'scene_digests.csv', newline='') as fh:
        for row in csv.DictReader(fh):
            if int(row['seed']) == seed:
                return row['sha256']


def run_method(scene, arm, workers=1):
    """Run one method on a scene; return (environment, planner cost record)."""
    import torch
    torch.set_num_threads(1)
    from obs_cbba.config import Config
    from obs_cbba.commitment import CommitmentEnv, run_commitment
    env = CommitmentEnv(Config(**scene.config), scene, log_events=False)
    cost = {}
    if arm == 'a':
        run_commitment(env, 'a')
    elif arm in ('b', 'regret-b', 'f', 'b_fc'):
        from obs_cbba.fixed_order_forecast import run
        _, cost = run(env, arm)
    elif arm in ('cbba', 'cbba-pr', 'msa', 'base_only'):
        from obs_cbba.baselines_p5 import run
        _, cost = run(env, 'cbba-pr' if arm == 'base_only' else arm)
    elif arm == 'o_obs':
        from obs_cbba.observable_order import run
        _, cost = run(env)
    elif arm.startswith('trigger_'):
        from obs_cbba import observable_cbba_latency
        from obs_cbba.trigger_cbba import load_selector, run
        selector, _ = load_selector(ROOT / 'checkpoints' / 'trigger.pt',
                                    expected_sha256=TRIGGER_SHA256,
                                    teacher_module=observable_cbba_latency, workers=workers)
        selector.threshold = float(arm.split('_', 1)[1])
        _, cost = run(env, selector)
    elif arm in VARIANTS:
        from obs_cbba.observable_cbba_latency import run
        v = VARIANTS[arm]
        _, cost = run(env, k=v['k'], variant=v['variant'], suffix=v['suffix'],
                      deferred_commitment=v['deferred'], workers=workers, deduplicate=True)
    else:
        raise SystemExit(f'unknown method {arm}')
    return env, cost


def archived_row(module, cell, seed, arm):
    path = ROOT / 'results' / 'episodes.csv'
    if not path.exists():
        return None
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            if (row['module'], row['cell'], row['arm']) == (module, cell, arm) \
                    and int(row['seed']) == seed and not row['init_seed']:
                return row


def main():
    names = sorted(METHODS) + [v for v in VARIANTS if v != 'o_obs-cbba'] + ['base_only'] + \
        [f'trigger_{t}' for t in TRIGGER_THRESHOLDS]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', action='store_true', help='list cells and seed ranges')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--method', default='cbba-pr', help=', '.join(names))
    ap.add_argument('--workers', type=int, default=1, choices=(1, 2, 4, 8),
                    help='processes for DA-CBBA branch evaluation (results do not depend on it)')
    args = ap.parse_args()
    if args.list:
        for cell in scene_table()['cells']:
            seeds = [g['seeds'] for g in cell['groups']]
            print(f"{cell['module']:13s} {cell['cell']:22s} seeds {seeds[0][0]}-{seeds[-1][1]}")
        return
    if args.seed is None:
        ap.error('--seed is required')
    arm = METHODS.get(args.method, args.method)
    module, cell, scene = make_scene(args.seed)
    digest = scene_digest(scene)
    same = digest == archived_digest(args.seed)
    print(f'scene {args.seed}: module {module}, cell {cell}, {len(scene.release)} buoys, '
          f'{scene.config["uavs"]} UAVs')
    print('scene arrays', 'match the archived scene' if same else
          'DIFFER from the archived scene (floating-point differences between platforms); '
          'the result below is not comparable row by row')

    from obs_cbba.metrics import metrics
    started = time.perf_counter()
    env, cost = run_method(scene, arm, args.workers)
    wall = time.perf_counter() - started
    m = metrics(env)
    print(f'{args.method}: TCR {100 * m["TCR"]:.2f}%  PST {100 * m["PST"]:.2f}%  '
          f'completed {m["completed"]} of {m["total"]}  '
          f'planning {cost.get("decision_seconds", env.compute_seconds):.1f} s  wall {wall:.1f} s')
    row = archived_row(module, cell, args.seed, arm)
    if row is None:
        print('no archived row for this scene and method')
    else:
        ok = (int(float(row['completed_on_time'])) == m['completed']
              and int(float(row['total'])) == m['total'])
        print(f'archived: TCR {100 * float(row["TCR"]):.2f}%  completed '
              f'{int(float(row["completed_on_time"]))} of {int(float(row["total"]))}  '
              f'-> {"identical" if ok else "different"}')


if __name__ == '__main__':
    main()
