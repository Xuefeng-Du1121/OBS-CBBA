"""SDs over scenes for the post hoc tables (deployment, futures, fixed rules); verifies every mean."""
import csv, json, statistics as st
from common import BASE, EP
X = f'{BASE}/results/posthoc'
E = {}
for r in csv.DictReader(open(EP)):
    if r['status'] == 'complete' and r['module'] in ('main', 'robustness'):
        E[(r['module'], r['cell'], int(r['seed']), r['arm'])] = float(r['TCR']) * 100

def rows(path):
    return {int(r['seed']): r for r in csv.DictReader(open(path))}

def diff(path, module, cell, ref='cbba-pr', refrows=None):
    R = rows(path)
    s = sorted(R)
    b = [float(refrows[k]['TCR']) * 100 for k in s] if refrows else [E[(module, cell, k, ref)] for k in s]
    a = [float(R[k]['TCR']) * 100 for k in s]
    d = [x - y for x, y in zip(a, b)]
    return st.mean(a), st.mean(d), st.stdev(d), st.mean(float(R[k]['PST']) * 100 for k in s)

def arch(module, cell, a, b):
    s = sorted(k[2] for k in E if k[0] == module and k[1] == cell and k[3] == a)
    d = [E[(module, cell, k, a)] - E[(module, cell, k, b)] for k in s]
    return st.mean(E[(module, cell, k, a)] for k in s), st.mean(d), st.stdev(d)

out = {'deploy': {}, 'futures': {}, 'fixed': {}}
O, REF = f'{X}/futures_delay', f'{X}/futures_delay_ref'
for cell in ('u6_n120_high', 'u6_n180_high', 'u4_n120_high'):
    out['deploy'][(cell, 'full')] = arch('main', cell, 'o_obs-cbba', 'cbba-pr')
    spec = {'keep': f'{X}/a3_rows.csv' if cell == 'u6_n120_high' else None,
            'fallback': f'{O}/a3f__main__{cell}_rows.csv',
            'anytime': f'{O}/anytime__main__{cell}_rows.csv',
            'snapshot': f'{O}/snapshot__main__{cell}_rows.csv',
            'clair': f'{X}/a7_rows.csv' if cell == 'u6_n120_high' else None,
            'pr_d1': f'{REF}/a3f_d1__main__{cell}_rows.csv',
            'snap_d1': f'{O}/snapshot_d1__main__{cell}_rows.csv',
            'snap_d1a': f'{O}/snapshot_d1a__main__{cell}_rows.csv',
            'full_d1a': f'{O}/full_d1a__main__{cell}_rows.csv'}
    import os
    for k, p in spec.items():
        if p and os.path.exists(p):
            out['deploy'][(cell, k)] = diff(p, 'main', cell)
# futures
for n in (80, 120, 180, None):
    cells = [f'u{u}_n{nn}_{a}' for u in (4, 6) for nn in ((n,) if n else (80, 120, 180)) for a in ('medium', 'high')]
    dfull, dsnap = [], []
    for c in cells:
        R = rows(f'{O}/snapshot__main__{c}_rows.csv')
        for k in sorted(R):
            dfull.append(E[('main', c, k, 'o_obs-cbba')] - E[('main', c, k, 'cbba-pr')])
            dsnap.append(float(R[k]['TCR']) * 100 - E[('main', c, k, 'o_obs-cbba')])
    out['futures'][('pool', n)] = dict(full=(st.mean(dfull), st.stdev(dfull)), snapshot=(st.mean(dsnap), st.stdev(dsnap)))
for module, cell in (('main', 'u6_n120_high'), ('main', 'u6_n80_medium'),
                     ('robustness', 'maritime-wind_u4'), ('robustness', 'maritime-anchored_u6')):
    e = {'full': arch(module, cell, 'o_obs-cbba', 'cbba-pr')[1:]}
    for mode in ('nodrift', 'noarr', 'snapshot'):
        p = f'{O}/{mode}__{module}__{cell}_rows.csv'
        if os.path.exists(p):
            e[mode] = diff(p, module, cell, ref='o_obs-cbba')[1:3]
    out['futures'][(module, cell)] = e
# fixed rules
F = f'{X}/fixed_rules'
for rule in ('pr2', 'pr4', 'prall', 'deadline', 'regret', 'slack'):
    R = list(csv.DictReader(open(f'{F}/fixed_{rule}__main_rows.csv')))
    a = [float(r['TCR']) * 100 for r in R]
    d = [float(r['TCR']) * 100 - E[('main', r['cell'], int(r['seed']), 'cbba-pr')] for r in R]
    out['fixed'][rule] = (st.mean(a), st.stdev(a), st.mean(d), st.stdev(d))
s = sorted(k for k in E if k[0] == 'main' and k[3] == 'o_obs-cbba')
d = [E[k] - E[(k[0], k[1], k[2], 'cbba-pr')] for k in s]
a = [E[k] for k in s]
out['fixed']['ours'] = (st.mean(a), st.stdev(a), st.mean(d), st.stdev(d))
a = [E[(k[0], k[1], k[2], 'cbba-pr')] for k in s]
out['fixed']['cbba-pr'] = (st.mean(a), st.stdev(a), 0, 0)
for g, v in out.items():
    for k, x in v.items():
        print(g, k, json.dumps(x, default=float) if isinstance(x, dict) else [round(y, 2) for y in x])
json.dump({g: {str(k): v for k, v in d.items()} for g, d in out.items()}, open('sd_extra.json', 'w'), indent=1)
