"""SDs over scenes for main-matrix tables (main_arm_pooled, secondary_holm, robustness); verifies means."""
import json, statistics as st
from collections import defaultdict
from common import episodes, cellmean
M = episodes('main'); RB = episodes('robustness')
F = {'TCR': ('TCR', 100), 'PST': ('PST', 100), 'Delay': ('delay_seconds', 1), 'Energy': ('energy', 1),
     'Latency': ('per_decision_latency_seconds', 1)}
out = {'pooled': {}, 'holm': {}, 'robust': {}}
for a, d in M.items():
    out['pooled'][a] = {k: (cellmean(d, f, s), st.stdev(m[f] * s for m in d.values())) for k, (f, s) in F.items()}

def pdiff(E, a, b, f, s, cell=None):
    ks = sorted(k for k in E[a] if (cell is None or k[0] == cell) and k in E[b])
    dd = defaultdict(list)
    for k in ks:
        dd[k[0]].append((E[a][k][f] - E[b][k][f]) * s)
    allv = [x for v in dd.values() for x in v]
    return st.mean(st.mean(v) for v in dd.values()), st.stdev(allv), len(allv)
cells = sorted({k[0] for k in M['cbba-pr']})
for c in cells:
    out['holm'][(c, 'cbba-pr', 'TCR')] = pdiff(M, 'o_obs-cbba', 'cbba-pr', 'TCR', 100, c)
for b in ('a', 'b', 'regret-b', 'f', 'cbba', 'msa', 'o_obs', 'p5-dhgc', 'p5-mappo'):
    out['holm'][('pooled', b, 'TCR')] = pdiff(M, 'o_obs-cbba', b, 'TCR', 100)
for k, (f, s) in F.items():
    if k != 'TCR':
        out['holm'][('pooled', 'cbba-pr', k)] = pdiff(M, 'o_obs-cbba', 'cbba-pr', f, s)
for c in sorted({k[0] for k in RB['cbba-pr']}):
    out['robust'][c] = pdiff(RB, 'o_obs-cbba', 'cbba-pr', 'TCR', 100, c)
for g, v in out.items():
    for k, x in v.items():
        print(g, k, {kk: [round(y, 4) for y in vv] for kk, vv in x.items()} if isinstance(x, dict) else [round(y, 3) for y in x])
json.dump({g: {str(k): v for k, v in d.items()} for g, d in out.items()}, open('sd_main.json', 'w'), indent=1)
