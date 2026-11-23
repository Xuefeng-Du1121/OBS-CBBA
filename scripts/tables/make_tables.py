"""Write the SD-subscript, single-column versions of the manuscript tables.

Run sd_main.py and sd_extra.py first (they write sd_main.json / sd_extra.json and
verify every mean against the per-scene source data). Subscripts are standard
deviations over scenes (per-scene paired differences for Delta columns).
"""
import json
T = 'tables'
M = json.load(open('sd_main.json')); X = json.load(open('sd_extra.json'))
HW = 'one core of a 9-vCPU virtual machine on an Intel Core Ultra 7 265K'


def num(x, d=2, sign=False):
    s = f'{abs(x):.{d}f}'
    if x < 0 and float(s) != 0:
        return '$-$' + s
    return ('$+$' if sign else '') + s


def ms(m, s, d=2, sign=False, bold=False):
    core = f'{abs(m):.{d}f}'
    pre = '-' if (m < 0 and float(core) != 0) else ('+' if sign else '')
    core = rf'\mathbf{{{core}}}' if bold else core
    return f'${pre}{core}_{{\\pm {s:.{d}f}}}$'


def write(name, body):
    open(f'{T}/{name}.tex', 'w').write(body)


NAME = {'o_obs-cbba': r'\textbf{\method}', 'cbba-pr': 'CBBA-PR', 'cbba': 'CBBA', 'o_obs': 'OBS-LA',
        'f': 'Fixed-F', 'b': 'Defer-B', 'regret-b': 'Regret-B', 'msa': 'MSA', 'p5-dhgc': 'HRL-2L',
        'a': 'Persist-A', 'p5-mappo': 'MAPPO'}
ORDER = ['o_obs-cbba', 'cbba-pr', 'cbba', 'o_obs', 'f', 'b', 'regret-b', 'msa', 'p5-dhgc', 'a', 'p5-mappo']

# ---------------------------------------------------------------- T5 main_arm_pooled
P = M['pooled']
rows = []
for a in ORDER:
    p = P[a]
    rows.append(f"{NAME[a]} & {ms(*p['TCR'], bold=a == 'o_obs-cbba')} & {p['PST'][0]:.2f} & {p['Delay'][0]:.1f}"
                f" & {p['Energy'][0]:.1f} & {p['Latency'][0]:.4f}\\\\")
write('main_arm_pooled', r"""\begin{table}[pos=t]
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{Pooled main-matrix endpoints (360 paired scenes; equal-cell means). Delay in s, energy in simulator units, latency in s per tick.}
\label{tab:p5-main-arm-pooled}
\begin{tabular}{@{}lccccc@{}}
\toprule
Method & TCR (\%) & PST (\%) & Delay & Energy & Latency\\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- T6 fixed_rules
F = X['fixed']
lab = [('pr2', 'CBBA-PR', '--', '32.1', '6.1'), ('pr4', 'Reopen 4', '8/12', '29.6', '8.2'),
       ('prall', 'Reopen all', '12/12', '28.7', '8.6'), ('deadline', 'Always deadline', '0/12', '34.2', '16.5'),
       ('regret', 'Always regret', '1/12', '12.9', '38.6'), ('slack', 'Always slack', '0/12', '20.0', '40.7')]
rows = []
for k, name, above, nev, q in lab:
    m, s, dm, dsd = F[k]
    d = '--' if k == 'pr2' else ms(dm, dsd, sign=True)
    rows.append(f'{name} & {ms(m, s)} & {d} & {above} & {nev} & {q}\\\\')
m, s, dm, dsd = F['ours']
rows.append(r'\midrule')
rows.append(rf'\method & {ms(m, s)} & {ms(dm, dsd, sign=True)} & 12/12 & 18.5 & 16.3\\')
write('fixed_rules', r"""\begin{table}[pos=t]
\centering\footnotesize
\setlength{\tabcolsep}{3.2pt}
\caption{Fixed revision rules on the 360 main scenes (post hoc). $\Delta$: TCR minus CBBA-PR; \emph{Above}: cells with $\Delta>0$; \emph{Never}, \emph{Queue}: tasks never assigned, lost in queue (\%).}
\label{tab:fixed}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccccc@{}}
\toprule
Rule & TCR (\%) & $\Delta$ (pp) & Above & Never & Queue\\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table}
""")

# ---------------------------------------------------------------- T9 deployment
D = X['deploy']
def drow(cell, key, label, events, over, share, pst=None):
    v = D[str((cell, key))]
    tcr, dm, dsd = v[0], v[1], v[2]
    p = v[3] if len(v) > 3 else pst
    return f' & {label} & {tcr:.2f} & {p:.2f} & {ms(dm, dsd, sign=True)} & {share} & {over}\\\\'
blk = []
blk.append(r'\multirow{7}{*}{\shortstack[l]{U6-\\N120-H}} & CBBA-PR & 76.75 & 90.21 & -- & -- & --\\')
blk.append(drow('u6_n120_high', 'full', 'No time limit', 1731, '26.3', '100\\%', 87.11))
blk.append(drow('u6_n120_high', 'keep', '25 s, keep previous', 1360, '12.6', '--'))
blk.append(drow('u6_n120_high', 'fallback', '25 s, fallback', 1749, '11.8', '89\\%'))
blk.append(drow('u6_n120_high', 'anytime', '25 s, anytime', 1732, '9.5', '110\\%'))
blk.append(drow('u6_n120_high', 'snapshot', 'Snapshot ($K=1$)', 1711, '0', '97\\%'))
blk.append(drow('u6_n120_high', 'clair', 'Clairvoyant', 1738, '0', '115\\%'))
blk.append(r'\cmidrule(l){2-7}')
blk.append(r'\multirow{4}{*}{\shortstack[l]{U6-\\N180-H}} & CBBA-PR & 60.07 & 90.37 & -- & -- & --\\')
blk.append(drow('u6_n180_high', 'full', 'No time limit', 1824, '47.7', '100\\%', 75.34))
blk.append(drow('u6_n180_high', 'fallback', '25 s, fallback', 1818, '44.3', '81\\%'))
blk.append(drow('u6_n180_high', 'snapshot', 'Snapshot ($K=1$)', 1801, '0', '61\\%'))
blk.append(r'\cmidrule(l){2-7}')
blk.append(r'\multirow{4}{*}{\shortstack[l]{U4-\\N120-H}} & CBBA-PR & 58.92 & 86.82 & -- & -- & --\\')
blk.append(drow('u4_n120_high', 'full', 'No time limit', 1579, '3.8', '100\\%', 75.12))
blk.append(drow('u4_n120_high', 'fallback', '25 s, fallback', 1579, '0', '100\\%'))
blk.append(drow('u4_n120_high', 'snapshot', 'Snapshot ($K=1$)', 1565, '0', '87\\%'))
blk.append(r'\midrule')
blk.append(r'\multicolumn{7}{@{}l}{\emph{One-tick execution delay}}\\')
blk.append(r'\multirow{4}{*}{\shortstack[l]{U6-\\N120-H}}' + drow('u6_n120_high', 'pr_d1', 'CBBA-PR', 0, '0', '--'))
blk.append(drow('u6_n120_high', 'snap_d1', 'Snapshot, naive', 0, '0', '--'))
blk.append(drow('u6_n120_high', 'snap_d1a', 'Snapshot, aware', 0, '0', '31\\%'))
blk.append(drow('u6_n120_high', 'full_d1a', r'\method, aware', 0, '9.7', '73\\%'))
blk.append(r'\cmidrule(l){2-7}')
blk.append(r'\multirow{4}{*}{\shortstack[l]{U6-\\N180-H}}' + drow('u6_n180_high', 'pr_d1', 'CBBA-PR', 0, '0', '--'))
blk.append(drow('u6_n180_high', 'snap_d1', 'Snapshot, naive', 0, '0', '--'))
blk.append(drow('u6_n180_high', 'snap_d1a', 'Snapshot, aware', 0, '0', '--'))
blk.append(drow('u6_n180_high', 'full_d1a', r'\method, aware', 0, '43.3', '42\\%'))
write('deployment', r"""\begin{table}[pos=t]
\centering\footnotesize
\setlength{\tabcolsep}{2.2pt}
\caption{Real-time replays (post hoc; 30 scenes per cell). $\Delta$: TCR minus undelayed CBBA-PR (pp); \emph{Share}: of the uncapped gain; \emph{Late}: calls over 25~s (\%).}
\label{tab:deploy}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llccccc@{}}
\toprule
Cell & Allocator & TCR & PST & $\Delta$ (pp) & Share & Late\\
\midrule
""" + '\n'.join(blk) + r"""
\bottomrule
\multicolumn{7}{@{}p{\columnwidth}@{}}{\scriptsize In the delayed U6-N180-H block, CBBA-PR and the aware snapshot agree in mean TCR to two decimals (58.558 vs 58.559\%) but differ in 26 of 30 scenes; their PST differs (89.31 vs 71.05\%).}\\
\end{tabular*}
\end{table}
""")

# ---------------------------------------------------------------- T10 futures
U = X['futures']
def fm(v):
    return ms(v[0], v[1], sign=True) if v else '--'
rows = []
for key, label in (("('pool', None)", 'Main'), ("('pool', 80)", r'\quad $N=80$'),
                   ("('pool', 120)", r'\quad $N=120$'), ("('pool', 180)", r'\quad $N=180$')):
    e = U[key]
    share = 1 + e['snapshot'][0] / e['full'][0]
    rows.append(f"{label} & {fm(e['full'])} & -- & -- & {fm(e['snapshot'])} & {share * 100:.0f}\\%\\\\")
rows.append(r'\midrule')
for key, label in (("('main', 'u6_n120_high')", 'U6-N120-H'), ("('main', 'u6_n80_medium')", 'U6-N80-M'),
                   ("('robustness', 'maritime-wind_u4')", 'Wind U4'),
                   ("('robustness', 'maritime-anchored_u6')", 'Anch.\\ U6')):
    e = U[key]
    share = 1 + e['snapshot'][0] / e['full'][0]
    rows.append(f"{label} & {fm(e['full'])} & {fm(e.get('nodrift'))} & {fm(e.get('noarr'))} & {fm(e['snapshot'])}"
                f" & {share * 100:.0f}\\%\\\\")
rows.append(r'\midrule')
rows.append(r'Call (s) & 12.3 / 52.3 & -- & -- & 0.86 / 3.99 & \\')
write('futures', r"""\begin{table}[pos=t]
\centering\footnotesize
\setlength{\tabcolsep}{1.2pt}
\caption{Forecast-content ablation (post hoc; pp). First column: \method minus CBBA-PR; others: variant minus \method. \emph{Share}: snapshot share of the gain.}
\label{tab:futures}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccccc@{}}
\toprule
Cell & \method$-$PR & No drift & No arr. & Snapshot & Share\\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table}
""")

# ---------------------------------------------------------------- T12 robustness
R = M['robust']
lab = [('maritime-wind_u4', 'Maritime wind', 4), ('maritime-wind_u6', 'Maritime wind', 6),
       ('maritime-anchored_u4', 'Anchored drift', 4), ('maritime-anchored_u6', 'Anchored drift', 6),
       ('hycom_u4', 'HYCOM panels', 4), ('hycom_u6', 'HYCOM panels', 6)]
rows = [f'{name} & {u} & {ms(R[c][0], R[c][1], sign=True)} & {R[c][2]}\\\\' for c, name, u in lab]
write('robustness', r"""\begin{table}[pos=t]
\centering\footnotesize
\setlength{\tabcolsep}{8pt}
\caption{Robustness family at $N=120$, high arrivals: \method minus CBBA-PR. All six tests reject after Holm correction (adjusted $p=1.2\times10^{-4}$).}
\label{tab:p5-robustness}
\begin{tabular}{@{}lccc@{}}
\toprule
Forcing & $U$ & $\Delta$TCR (pp) & Scenes\\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- T13 secondary_holm
H = M['holm']
def cl(c):
    u, n, a = c.split('_')
    return f'{u.upper()}-{n.upper()}-{"M" if a == "medium" else "H"}'
cells = [f'u{u}_n{n}_{a}' for u in (4, 6) for n in (80, 120, 180) for a in ('medium', 'high')]
rows = [f"{cl(c)} vs CBBA-PR & TCR & {ms(*H[str((c, 'cbba-pr', 'TCR'))][:2], sign=True)}\\\\" for c in cells]
rows.append(r'\midrule')
for b in ('a', 'b', 'regret-b', 'f', 'cbba', 'msa', 'o_obs', 'p5-dhgc', 'p5-mappo'):
    rows.append(f"Pooled vs {NAME[b]} & TCR & {ms(*H[str(('pooled', b, 'TCR'))][:2], sign=True)}\\\\")
rows.append(r'\midrule')
for k in ('Delay', 'Energy', 'PST', 'Latency'):
    rows.append(f"Pooled vs CBBA-PR & {k} & {ms(*H[str(('pooled', 'cbba-pr', k))][:2], sign=True)}\\\\")
write('secondary_holm', r"""\begin{table}[pos=t]
\centering\footnotesize
\setlength{\tabcolsep}{7pt}
\caption{The pre-specified 25-test Holm family: \method minus the comparator. All 25 tests reject at $\alpha=0.05$ (Holm-adjusted $p<5\times10^{-4}$).}
\label{tab:p5-secondary-holm}
\begin{tabular}{@{}llc@{}}
\toprule
Test & Endpoint & Estimate\\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
print('written')
