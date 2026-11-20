"""Numbers for the forecast-content ablation and the real-time tables of the paper.

Usage: python r2r3_tables.py <episodes.csv> <r2r3_out> <r2r3_out_ref> <bootstrap_dir>
Per-cell intervals: percentile bootstrap over scenes (10,000 resamples, seed 20261111).
Pooled intervals: cell-stratified paired bootstrap of the frozen rule (50,000 resamples),
with seed 20261111 because these analyses are post hoc.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

episodes, out, ref, bdir = sys.argv[1:5]
sys.path.insert(0, bdir)
from bootstrap import paired_stratified_bootstrap  # noqa: E402

E = {}
for r in csv.DictReader(open(episodes)):
    if r['status'] == 'complete':
        E[(r['module'], r['cell'], int(r['seed']), r['arm'])] = r


def load(path):
    return {int(r['seed']): r for r in csv.DictReader(open(path))}


def ci(d):
    rng = np.random.default_rng(20261111)
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    return tuple(float(v) for v in np.percentile(b, [2.5, 97.5]))


def fmt(d):
    lo, hi = ci(d)
    return f'{d.mean():+.2f} [{lo:.2f}, {hi:.2f}]'


def arch(module, cell, seeds, arm, key='TCR'):
    return np.array([float(E[(module, cell, s, arm)][key]) for s in seeds])


def col(rows, seeds, key):
    return np.array([float(rows[s][key]) for s in seeds])


def calls(rows, seeds):
    return np.array([t for s in seeds for t in json.loads(rows[s]['call_seconds'])])


print('== R2: content of the futures ==')
snap_pr, snap_da, full_pr = {}, {}, {}
for path in sorted(glob.glob(os.path.join(out, '*_rows.csv'))):
    name = os.path.basename(path)
    if name.startswith('._'):
        continue
    mode, module, cell = name.replace('_rows.csv', '').split('__')
    if mode not in ('snapshot', 'nodrift', 'noarr'):
        continue
    rows = load(path)
    seeds = sorted(rows)
    x = col(rows, seeds, 'TCR') * 100
    da = arch(module, cell, seeds, 'o_obs-cbba') * 100
    pr = arch(module, cell, seeds, 'cbba-pr') * 100
    total = col(rows, seeds, 'total').sum()
    c = calls(rows, seeds)
    print(f'{mode:9s} {module:10s} {cell:22s} n={len(seeds)} TCR {x.mean():.2f} DA {da.mean():.2f} PR {pr.mean():.2f}'
          f' | vs DA {fmt(x - da)} | vs PR {fmt(x - pr)} | share {(x - pr).mean() / (da - pr).mean():.2f}'
          f' | never {col(rows, seeds, "n_never_assigned").sum() / total:.3f}'
          f' anc {col(rows, seeds, "n_assigned_not_completed").sum() / total:.3f}'
          f' PST {col(rows, seeds, "PST").mean() * 100:.2f}'
          f' | call p50 {np.percentile(c, 50):.2f} p95 {np.percentile(c, 95):.2f} max {c.max():.2f}'
          f' per-tick {col(rows, seeds, "decision_seconds").mean() / 100:.2f}')
    if mode == 'snapshot' and module == 'main':
        snap_pr[cell] = x - pr
        snap_da[cell] = x - da
        full_pr[cell] = da - pr

print('\n== snapshot pooled over the 12 main cells (stratified, 50,000 resamples) ==')
for label, cells in (('snapshot - CBBA-PR', snap_pr), ('snapshot - DA-CBBA', snap_da), ('DA-CBBA - CBBA-PR', full_pr)):
    assert len(cells) == 12
    r = paired_stratified_bootstrap(dict(sorted(cells.items())), replicates=50000, seed=20261111)
    print(f'{label}: {r["estimate"]:+.2f} [{r["lower"]:.2f}, {r["upper"]:.2f}]')
for n in (80, 120, 180):
    sub = {k: v for k, v in snap_da.items() if f'_n{n}_' in k}
    r = paired_stratified_bootstrap(dict(sorted(sub.items())), replicates=50000, seed=20261111)
    sp = {k: v for k, v in snap_pr.items() if f'_n{n}_' in k}
    q = paired_stratified_bootstrap(dict(sorted(sp.items())), replicates=50000, seed=20261111)
    print(f'N={n}: snapshot - DA {r["estimate"]:+.2f} [{r["lower"]:.2f}, {r["upper"]:.2f}];'
          f' snapshot - PR {q["estimate"]:+.2f} [{q["lower"]:.2f}, {q["upper"]:.2f}]')
allc = []
for path in sorted(glob.glob(os.path.join(out, 'snapshot__main__*_rows.csv'))):
    rows = load(path)
    allc.append(calls(rows, sorted(rows)))
allc = np.concatenate(allc)
print(f'snapshot calls, 12 main cells: n={len(allc)} p50 {np.percentile(allc, 50):.2f}'
      f' p95 {np.percentile(allc, 95):.2f} max {allc.max():.2f} mean {allc.mean():.2f}')

print('\n== R3: budget, anytime and delay ==')
for cell in ('u6_n120_high', 'u6_n180_high', 'u4_n120_high'):
    for mode, folder in (('a3f', out), ('anytime', out), ('snapshot', out), ('a3f_d1', ref),
                         ('snapshot_d1', out), ('snapshot_d1a', out), ('full_d1a', out),
                         ('a3f_d1', out), ('anytime_d1', out)):
        path = os.path.join(folder, f'{mode}__main__{cell}_rows.csv')
        if not os.path.exists(path):
            continue
        rows = load(path)
        seeds = sorted(rows)
        x = col(rows, seeds, 'TCR') * 100
        da = arch('main', cell, seeds, 'o_obs-cbba') * 100
        pr = arch('main', cell, seeds, 'cbba-pr') * 100
        c = calls(rows, seeds)
        n_calls = int(col(rows, seeds, 'calls').sum())
        line = (f'{cell:13s} {mode:13s} {"ref" if folder == ref else "   "} n={len(seeds)} TCR {x.mean():.2f}'
                f' PST {col(rows, seeds, "PST").mean() * 100:.2f} | vs PR {fmt(x - pr)} wins {(x > pr).sum()}'
                f' | vs DA {fmt(x - da)} | share {(x - pr).mean() / (da - pr).mean():.2f}'
                f' | events {n_calls} overrun {col(rows, seeds, "overruns").sum() / n_calls:.3f}'
                f' fallback {col(rows, seeds, "fallbacks").sum() / n_calls:.3f}'
                f' >25s {np.mean(c > 25):.3f} >50s {np.mean(c > 50):.3f} p95 {np.percentile(c, 95):.1f} max {c.max():.1f}')
        dpath = os.path.join(ref, f'a3f_d1__main__{cell}_rows.csv')
        if '_d1' in mode and os.path.exists(dpath) and folder != ref:
            prd = col(load(dpath), seeds, 'TCR') * 100
            line += f' | vs delayed PR {fmt(x - prd)} wins {(x > prd).sum()}'
        print(line)
    print(f'{cell:13s} archive: DA {arch("main", cell, seeds, "o_obs-cbba").mean() * 100:.2f}'
          f' PST {arch("main", cell, seeds, "o_obs-cbba", "PST").mean() * 100:.2f}'
          f' PR {arch("main", cell, seeds, "cbba-pr").mean() * 100:.2f}'
          f' PST {arch("main", cell, seeds, "cbba-pr", "PST").mean() * 100:.2f}'
          f' gain {fmt((arch("main", cell, seeds, "o_obs-cbba") - arch("main", cell, seeds, "cbba-pr")) * 100)}')
