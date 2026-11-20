"""Protocol v1.2 analysis, exactly as pre-specified in p5_protocol_v1_2.md.

Usage: python v12_analyze.py <out dir> <bootstrap dir>
"""
import csv
import json
import sys

import numpy as np

out, bdir = sys.argv[1:3]
sys.path.insert(0, bdir)
from bootstrap import paired_stratified_bootstrap, holm  # noqa: E402

SEED = 20261112
ARMS = ['pr', 'snap', 'pr_d1', 'snap_d1a']
R = {}
for arm in ARMS:
    for r in csv.DictReader(open(f'{out}/v12_{arm}__all_rows.csv')):
        R[(arm, r['cell'], int(r['seed']))] = r
cells = sorted({k[1] for k in R}, key=lambda c: (c[:2], int(c.split('_')[1][1:]), c.split('_')[2]))


def diff(a, b):
    d = {}
    for c in cells:
        seeds = sorted(s for (arm, cc, s) in R if arm == a and cc == c and (b, c, s) in R)
        d[c] = np.array([float(R[(a, c, s)]['TCR']) - float(R[(b, c, s)]['TCR']) for s in seeds]) * 100
    return d


def test(a, b):
    d = diff(a, b)
    r = paired_stratified_bootstrap(d, replicates=50000, seed=SEED)
    centred = r['replicates'] - r['estimate']
    p = float((1 + np.count_nonzero(np.abs(centred) >= abs(r['estimate']))) / 50001)
    return d, r, p


res = {}
tests = [('H1', 'snap', 'pr'), ('H2', 'snap_d1a', 'pr_d1')]
for name, a, b in tests:
    d, r, p = test(a, b)
    res[name] = dict(contrast=f'{a} - {b}', estimate=r['estimate'], lower=r['lower'], upper=r['upper'], p=p,
                     n=sum(len(v) for v in d.values()))
adj = holm([res[n]['p'] for n, _, _ in tests])
for i, (n, _, _) in enumerate(tests):
    res[n].update(p_holm=adj['adjusted'][i], reject=adj['reject'][i])
d, r, _ = test('snap_d1a', 'pr')
res['D_snap_d1a_vs_pr'] = dict(estimate=r['estimate'], lower=r['lower'], upper=r['upper'])
d, r, _ = test('pr_d1', 'pr')
res['D_pr_d1_vs_pr'] = dict(estimate=r['estimate'], lower=r['lower'], upper=r['upper'])

rng_cells = {}
for name, a, b in (('snap-pr', 'snap', 'pr'), ('snap_d1a-pr_d1', 'snap_d1a', 'pr_d1')):
    for c, v in diff(a, b).items():
        g = np.random.Generator(np.random.PCG64(SEED))
        boot = v[g.integers(0, len(v), (10000, len(v)))].mean(1)
        rng_cells.setdefault(c, {})[name] = (v.mean(), *np.quantile(boot, [.025, .975]))
res['cells'] = {c: {k: [round(float(x), 2) for x in t] for k, t in v.items()} for c, v in rng_cells.items()}

arms = {}
for arm in ARMS:
    rows = [r for (a, _, _), r in R.items() if a == arm]
    tcr = np.mean([np.mean([float(r['TCR']) for r in rows if r['cell'] == c]) for c in cells]) * 100
    tot = sum(int(r['total']) for r in rows)
    secs = [x for r in rows for x in json.loads(r['call_seconds'])]
    arms[arm] = dict(scenes=len(rows), TCR=tcr,
                     never=100 * sum(int(r['n_never_assigned']) for r in rows) / tot,
                     lost=100 * sum(int(r['n_assigned_not_completed']) for r in rows) / tot,
                     calls=sum(int(r['calls']) for r in rows), fallbacks=sum(int(r['fallbacks']) for r in rows),
                     median=float(np.median(secs)) if secs else 0., p95=float(np.quantile(secs, .95)) if secs else 0.,
                     max=float(max(secs, default=0.)), cpu=rows[0]['cpu'])
res['arms'] = arms
print(json.dumps(res, indent=1))
json.dump(res, open(f'{out}/v12_results.json', 'w'), indent=1)
