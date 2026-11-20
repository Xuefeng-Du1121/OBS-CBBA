import pandas as pd
F = pd.read_csv('../20261024/p5_results_final/results/main_secondary_family.csv')
NAME = {'cbba-pr': 'CBBA-PR', 'cbba': 'CBBA', 'o_obs': 'OBS-LA', 'f': 'Fixed-F', 'b': 'Defer-B', 'regret-b': 'Regret-B',
        'msa': 'MSA', 'p5-dhgc': 'HRL-2L', 'a': 'Persist-A', 'p5-mappo': 'MAPPO'}
def n(x):
    s = f'{abs(x):.2f}'
    return ('$-$' if x < 0 else '$+$') + s
def lohi(x):
    s = f'{abs(x):.2f}'
    return ('$-$' if x < 0 else '') + s
def cl(c):
    u, nn, a = c.split('_'); return f'{u.upper()}-{nn.upper()}-{"M" if a == "medium" else "H"}'
rows = []
assert F.reject_holm_05.all() and len(F) == 25
cells = [f'u{u}_n{k}_{a}' for u in (4, 6) for k in (80, 120, 180) for a in ('medium', 'high')]
def line(lab, ep, r):
    return f"{lab} & {ep} & {n(r.estimate)} & [{lohi(r.lower)}, {lohi(r.upper)}]\\\\"
for c in cells:
    r = F[(F.component == 'cellwise_primary') & (F.cell == c)].iloc[0]
    rows.append(line(f'{cl(c)} vs CBBA-PR', 'TCR', r))
rows.append(r'\midrule')
for b in ('a', 'b', 'regret-b', 'f', 'cbba', 'msa', 'o_obs', 'p5-dhgc', 'p5-mappo'):
    r = F[(F.contrast == f'o_obs-cbba minus {b}') & (F.endpoint == 'TCR') & (F.component != 'cellwise_primary')].iloc[0]
    rows.append(line(f'Pooled vs {NAME[b]}', 'TCR', r))
rows.append(r'\midrule')
for ep, lab in (('delay', 'Delay (s)'), ('energy', 'Energy'), ('PST', 'PST'), ('latency', 'Latency (s)')):
    q = F[(F.contrast == 'o_obs-cbba minus cbba-pr') & (F.component != 'cellwise_primary') & (F.endpoint.str.lower() == ep.lower())]
    if q.empty:
        print('missing', ep, F[F.contrast == 'o_obs-cbba minus cbba-pr'].endpoint.unique()); continue
    rows.append(line('Pooled vs CBBA-PR', lab, q.iloc[0]))
tex = r"""\begin{table}[pos=tb]
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{The pre-specified 25-test Holm family: \method minus the comparator, with 95\% CIs. All 25 tests reject at $\alpha=0.05$ (Holm-adjusted $p<5\times10^{-4}$).}
\label{tab:p5-secondary-holm}
\begin{tabular}{@{}llrr@{}}
\toprule
Test & Endpoint & Estimate & 95\% CI\\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open('tables/secondary_holm.tex', 'w').write(tex)
print(tex)
