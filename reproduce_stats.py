"""Recompute the pre-registered statistics of the paper from results/episodes.csv.

    python reproduce_stats.py

Prints the primary contrast (DA-CBBA minus CBBA-PR, task completion rate), the
25-test secondary family and the six robustness contrasts, and compares every
number with the archived tables in results/tables/. Uses only numpy; takes
about one minute.

Rule: paired differences per scene, resampled within cells (50,000 PCG64
resamples, seed 690524 for the main matrix and 690525 for robustness), equal
cell weights, percentile intervals, centred two-sided p-values and Holm
correction at 0.05. The code names of the methods are listed in README.md.
"""
from __future__ import annotations

import csv
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

from bootstrap import ROBUSTNESS_SEED, compare_arms, holm, paired_stratified_bootstrap

ROOT = Path(__file__).resolve().parent
CAND, BASE = 'o_obs-cbba', 'cbba-pr'
MAIN_CELLS = [f'u{u}_n{n}_{a}' for u in (4, 6) for n in (80, 120, 180) for a in ('medium', 'high')]
OTHER = ['a', 'b', 'regret-b', 'f', 'cbba', 'msa', 'o_obs']
ENDPOINTS = OrderedDict(delay=('delay_seconds', 1.0), energy=('energy', 1.0), PST=('PST', 100.0),
                        latency=('per_decision_latency_seconds', 1.0))
LEARNED = ['p5-dhgc', 'p5-mappo']
ROBUST_CELLS = ['maritime-wind_u4', 'maritime-wind_u6', 'maritime-anchored_u4',
                'maritime-anchored_u6', 'hycom_u4', 'hycom_u6']
COLUMNS = ['TCR', 'delay_seconds', 'energy', 'PST', 'per_decision_latency_seconds']


def number(value):
    return None if value in ('', None) else float(value)


def load():
    rows = []
    with open(ROOT / 'results' / 'episodes.csv', newline='') as fh:
        for r in csv.DictReader(fh):
            for c in COLUMNS:
                r[c] = number(r[c])
            r['seed'] = int(r['seed'])
            rows.append(r)
    return rows


def pvalue(replicates, estimate):
    return float((1 + np.count_nonzero(np.abs(replicates - estimate) >= abs(estimate)))
                 / (len(replicates) + 1))


def contrast(strata, groups, seed):
    """Bootstrap ordered strata of paired differences; report each group of strata."""
    r = paired_stratified_bootstrap(strata, seed=seed)
    reps = {s: v[r['indices'][:, i, :len(v)]].mean(axis=1) for i, (s, v) in enumerate(strata.items())}
    out = OrderedDict()
    for name, members in groups.items():
        rep = np.stack([reps[s] for s in members], axis=1).mean(axis=1)
        est = float(np.mean([strata[s].mean() for s in members]))
        lo, hi = np.quantile(rep, [.025, .975], method='linear')
        out[name] = dict(estimate=est, lower=float(lo), upper=float(hi), p=pvalue(rep, est))
    return out


def differences(index, cells, seeds, column, cand, base, scale):
    out = OrderedDict()
    for c in cells:
        pairs = [(index[(c, s, cand)][column], index[(c, s, base)][column]) for s in seeds[c]]
        out[c] = np.array([scale * (a - b) for a, b in pairs if a is not None and b is not None])
    return out


def check(label, got, table, key):
    want = table[key]
    ok = all(abs(got[k] - float(want[k])) < 1e-9 for k in ('estimate', 'lower', 'upper'))
    print(f'{label:34s} {got["estimate"]:9.3f} [{got["lower"]:8.3f}, {got["upper"]:8.3f}]  '
          f'p_holm {got.get("p_holm", float("nan")):.2g}  '
          f'{"matches archived table" if ok else "DIFFERS from archived table"}')
    return ok


def read_table(name, key):
    with open(ROOT / 'results' / 'tables' / name, newline='') as fh:
        return {r[key]: r for r in csv.DictReader(fh)}


def main():
    rows = load()
    ok = True

    # ---- primary contrast: the frozen function, rows in archive order ----
    main_rows = [r for r in rows if r['module'] == 'main']
    primary = compare_arms(main_rows, candidate=CAND, baseline=BASE)
    order = primary['cell_order']
    print('Primary contrast, DA-CBBA minus CBBA-PR, TCR in percentage points')
    print(f'  {100 * primary["estimate"]:.2f} [{100 * primary["lower"]:.2f}, {100 * primary["upper"]:.2f}], '
          f'p = {primary["p_centered"]:.2g}, 360 scenes in 12 cells\n')

    # ---- 25-test secondary family ----
    index = {(r['cell'], r['seed'], r['arm']): r for r in main_rows}
    learned = defaultdict(list)
    for r in rows:
        if r['module'] == 'learned':
            learned[(r['cell'], r['seed'], r['arm'])].append(r['TCR'])
    for key, values in learned.items():  # three frozen initializations averaged within scene
        assert len(values) == 3
        index[key] = dict(TCR=float(np.mean(values)))
    seeds = {c: sorted({s for (cc, s, a) in index if cc == c}) for c in order}
    groups = OrderedDict((c, [c]) for c in MAIN_CELLS)
    groups['pooled'] = list(order)
    tests = []
    cellwise = contrast(differences(index, order, seeds, 'TCR', CAND, BASE, 100.), groups, 690524)
    for c in MAIN_CELLS:
        tests.append((f'cell:{c}', cellwise[c]))
    for arm in OTHER:
        d = differences(index, order, seeds, 'TCR', CAND, arm, 100.)
        tests.append((f'pooled_TCR:{arm}', contrast(d, groups, 690524)['pooled']))
    for name, (column, scale) in ENDPOINTS.items():
        d = differences(index, order, seeds, column, CAND, BASE, scale)
        tests.append((f'pooled_{name}:{BASE}', contrast(d, groups, 690524)['pooled']))
    for arm in LEARNED:
        d = differences(index, order, seeds, 'TCR', CAND, arm, 100.)
        tests.append((f'pooled_TCR:{arm}', contrast(d, groups, 690524)['pooled']))
    adjusted = holm([t['p'] for _, t in tests])
    table = read_table('main_secondary_family.csv', 'id')
    print('Secondary family (25 tests, Holm): estimate [95% interval]')
    for (name, t), p, reject in zip(tests, adjusted['adjusted'], adjusted['reject']):
        t['p_holm'] = p
        ok &= check(name, t, table, name)
        ok &= str(reject) == table[name]['reject_holm_05']

    # ---- robustness family: HYCOM cells are stratified by current panel ----
    index = {(r['cell'], r['seed'], r['arm']): r for r in rows if r['module'] == 'robustness'}
    strata, groups = OrderedDict(), OrderedDict()
    for c in ROBUST_CELLS:
        cell_seeds = sorted({s for (cc, s, a) in index if cc == c})
        parts = [cell_seeds[10 * i:10 * i + 10] for i in range(6)] if c.startswith('hycom') else [cell_seeds]
        groups[c] = []
        for i, part in enumerate(parts):
            key = f'{c}/panel{i}' if len(parts) > 1 else c
            strata[key] = np.array([100. * (index[(c, s, CAND)]['TCR'] - index[(c, s, BASE)]['TCR'])
                                    for s in part])
            groups[c].append(key)
    robust = contrast(strata, groups, ROBUSTNESS_SEED)
    adjusted = holm([robust[c]['p'] for c in ROBUST_CELLS])
    table = read_table('robustness_family.csv', 'cell')
    print('\nRobustness family (6 tests, Holm), DA-CBBA minus CBBA-PR, TCR in percentage points')
    for c, p in zip(ROBUST_CELLS, adjusted['adjusted']):
        robust[c]['p_holm'] = p
        ok &= check(c, robust[c], table, c)

    print('\nAll recomputed values match the archived tables.' if ok else
          '\nSome recomputed values differ from the archived tables.')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
