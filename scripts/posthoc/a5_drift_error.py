"""A5: quantitative accuracy of the drift estimator used by DA-CBBA.

Replays the frozen estimator on every main-matrix scene from the archived
scenario arrays, compares its one-step cellwise mean against the realized
one-step displacement of the same released tasks, and propagates the error to
multi-step position predictions. No new seed is generated: the script reads
only ``scenario.npz`` from episodes that already ran.

Usage:
    python -I a5_drift_error.py --archive <dir> --episodes <episodes.csv> --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HISTORY_STEPS = 8
GRID = 3
HORIZONS = (1, 2, 4, 8, 16, 32, 50)
MIN_CELL_N = 2


def cell_indices(points, grid=GRID):
    xy = np.minimum((np.asarray(points) * grid).astype(int), grid - 1)
    return xy[:, 0] * grid + xy[:, 1]


def estimate(positions, release, t, *, grid=GRID, history=HISTORY_STEPS):
    """Identical arithmetic to dhgc.maritime_planning.estimate_drift.

    Returns per-cell mean one-step displacement, plus the global fallback and
    the cell counts, at decision time ``t``. ``positions`` is indexed by
    physical tick and task, exactly as the trusted adapter reads it.
    """
    increments = []
    cells = []
    for tau in range(max(0, t - history + 1), t):
        ids = np.flatnonzero(release <= tau)
        if ids.size == 0:
            continue
        start = positions[tau, ids]
        nxt = positions[tau + 1, ids]
        increments.append(nxt - start)
        cells.append(cell_indices(start, grid))
    delta = np.concatenate(increments) if increments else np.zeros((0, 2))
    cells = np.concatenate(cells) if cells else np.zeros(0, dtype=int)
    counts = np.bincount(cells, minlength=grid * grid)
    global_mean = delta.mean(axis=0) if len(delta) else np.zeros(2)
    means = np.tile(global_mean, (grid * grid, 1))
    borrowed = np.ones(grid * grid, dtype=bool)
    for g in range(grid * grid):
        v = delta[cells == g]
        if len(v) >= MIN_CELL_N:
            means[g] = v.mean(axis=0)
            borrowed[g] = False
    return dict(means=means, counts=counts, borrowed=borrowed, global_mean=global_mean)


def realized(positions, release, t, horizon, grid=GRID):
    """Realized displacement over ``horizon`` ticks and its per-cell mean."""
    ids = np.flatnonzero((release <= t) & (t + horizon <= positions.shape[0] - 1))
    if ids.size == 0:
        return None
    start = positions[t, ids]
    disp = positions[t + horizon, ids] - start
    cells = cell_indices(start, grid)
    return dict(ids=ids, start=start, disp=disp, cells=cells)


def analyse_scene(path, order=None):
    with np.load(path) as z:
        positions = np.asarray(z['positions'], dtype=float)
        release = np.asarray(z['release'], dtype=int)
    t_max = positions.shape[0] - 1
    rows = []
    for t in range(8, min(100, t_max) + 1):
        est = estimate(positions, release, t)
        for horizon in HORIZONS:
            got = realized(positions, release, t, horizon)
            if got is None:
                continue
            # Predicted displacement under the estimated constant cell mean.
            pred = est['means'][got['cells']] * horizon
            err = np.linalg.norm(got['disp'] - pred, axis=1)
            zero = np.linalg.norm(got['disp'], axis=1)
            # One-step estimator error against the realized global mean, which
            # isolates estimation error from the drift-model mismatch.
            actual_global = got['disp'].mean(axis=0) / horizon
            rows.append(dict(t=t, horizon=horizon, n=int(got['ids'].size),
                             mae=float(err.mean()), rmse=float(np.sqrt((err ** 2).mean())),
                             zero_rmse=float(np.sqrt((zero ** 2).mean())),
                             bias=float(np.linalg.norm(est['global_mean'] - actual_global)),
                             borrowed=int(est['borrowed'].sum())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', required=True)
    ap.add_argument('--episodes', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = Path(args.archive)
    rows = []
    handled = set()
    with open(args.episodes) as fh:
        reader = csv.DictReader(fh)
        for rec in reader:
            if rec['status'] != 'complete':
                continue
            key = (rec['batch'], rec['episode_dir'])
            if key in handled:
                continue
            handled.add(key)
            folder = archive / rec['batch'] / 'episodes' / rec['episode_dir'] / 'scenario.npz'
            if not folder.exists():
                continue
            for row in analyse_scene(folder):
                rows.append(dict(cell=rec['cell'], seed=int(rec['seed']), arm=rec['arm'], **row))
            if args.limit and len(handled) >= args.limit:
                break

    with (out / 'a5_drift_error.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    agg = defaultdict(list)
    for r in rows:
        agg[r['horizon']].append(r)
    summary = []
    for horizon in HORIZONS:
        v = agg[horizon]
        if not v:
            continue
        summary.append(dict(horizon=horizon, scenes=len(v),
                            rmse=float(np.mean([x['rmse'] for x in v])),
                            zero_rmse=float(np.mean([x['zero_rmse'] for x in v])),
                            mae=float(np.mean([x['mae'] for x in v])),
                            bias=float(np.mean([x['bias'] for x in v]))))
    (out / 'a5_summary.json').write_text(json.dumps(dict(
        scenes=len(handled), rows=len(rows), by_horizon=summary), indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
