"""S3: fixed revision rules versus CBBA-PR and the proposed method (main matrix).

Usage: python fixed_tables.py <episodes.csv> <fixed_out> <bootstrap_dir>
Pooled intervals: cell-stratified paired bootstrap, 50,000 resamples, seed 20261111.
"""
import csv, glob, os, sys
import numpy as np
episodes, out, bdir = sys.argv[1:4]
sys.path.insert(0, bdir)
from bootstrap import paired_stratified_bootstrap
E = {}
for r in csv.DictReader(open(episodes)):
    if r['status'] == 'complete' and r['module'] == 'main':
        E[(r['cell'], int(r['seed']), r['arm'])] = r
def pooled(diff):
    r = paired_stratified_bootstrap(diff, replicates=50000, seed=20261111)
    return f"{r['estimate']:+.2f} [{r['lower']:.2f}, {r['upper']:.2f}]"
order = ['pr2', 'pr4', 'prall', 'deadline', 'regret', 'slack']
rows = {}
for rule in order:
    p = os.path.join(out, f'fixed_{rule}__main_rows.csv')
    if os.path.exists(p):
        rows[rule] = list(csv.DictReader(open(p)))
for rule, R in rows.items():
    by = {}
    for r in R:
        by.setdefault(r['cell'], []).append(r)
    dpr, dda, tcr = {}, {}, []
    never = anc = tot = 0
    for cell, rr in sorted(by.items()):
        x = np.array([float(r['TCR']) for r in rr]) * 100
        pr = np.array([float(E[(cell, int(r['seed']), 'cbba-pr')]['TCR']) for r in rr]) * 100
        da = np.array([float(E[(cell, int(r['seed']), 'o_obs-cbba')]['TCR']) for r in rr]) * 100
        dpr[cell], dda[cell] = x - pr, x - da
        tcr.append(x.mean())
        never += sum(int(r['n_never_assigned']) for r in rr); anc += sum(int(r['n_assigned_not_completed']) for r in rr)
        tot += sum(int(r['total']) for r in rr)
    better = sum(dpr[c].mean() > 0 for c in dpr)
    print(f'{rule:9s} n={len(R)} cells={len(by)} TCR {np.mean(tcr):.2f} | vs PR {pooled(dpr)} | vs DA {pooled(dda)} '
          f'| cells above PR {better}/12 | never {100*never/tot:.1f} lost-after-assign {100*anc/tot:.1f}')
    for N in ('n80', 'n120', 'n180'):
        sub = {c: v for c, v in dpr.items() if f'_{N}_' in c}
        print(f'     {N}: vs PR {np.mean([v.mean() for v in sub.values()]):+.2f}')
