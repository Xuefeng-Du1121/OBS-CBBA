"""Write paper_p5/manuscript/tables/v12.tex from out/v12_results.json and the row files.

Means are the pre-specified estimates; subscripts are standard deviations over scenes
(of the per-scene paired difference for Delta columns).
"""
import json, csv, statistics as st
r = json.load(open('out/v12_results.json'))
R = {}
for a in ['pr', 'pr_d1', 'snap', 'snap_d1a']:
    R[a] = {(x['cell'], int(x['seed'])): float(x['TCR']) * 100 for x in csv.DictReader(open(f'out/v12_{a}__all_rows.csv'))}
order = [f'u{u}_n{n}_{a}' for u in (4, 6) for n in (80, 120, 180) for a in ('medium', 'high')]


def stats(a, b, cell=None):
    ks = sorted(k for k in R[a] if cell is None or k[0] == cell)
    d = [R[a][k] - R[b][k] for k in ks]
    return st.mean(d), st.stdev(d)


def ms(m, s):
    core = f'{abs(m):.2f}'
    pre = '-' if (m < 0 and float(core) != 0) else '+'
    return f'${pre}{core}_{{\\pm {s:.2f}}}$'


L = []
for c in order:
    u, n, a = c.split('_')
    lab = f'{u.upper()}-{n.upper()}-{"M" if a == "medium" else "H"}'
    m1, s1 = stats('snap', 'pr', c)
    m2, s2 = stats('snap_d1a', 'pr_d1', c)
    assert abs(m1 - r['cells'][c]['snap-pr'][0]) < 0.006 and abs(m2 - r['cells'][c]['snap_d1a-pr_d1'][0]) < 0.006
    L.append(f'{lab} & {ms(m1, s1)} & {ms(m2, s2)}\\\\')
A, H1, H2 = r['arms'], r['H1'], r['H2']
m1, s1 = stats('snap', 'pr'); m2, s2 = stats('snap_d1a', 'pr_d1')
assert abs(m1 - H1['estimate']) < 1e-9 and abs(m2 - H2['estimate']) < 1e-9
nl = lambda a: f"{A[a]['never']:.1f} / {A[a]['lost']:.1f}"
tm = lambda a: f"{A[a]['median']:.2f} / {A[a]['p95']:.2f} / {A[a]['max']:.1f}"
tex = r"""\begin{table}[pos=t]
\centering\footnotesize
\setlength{\tabcolsep}{6pt}
\caption{Confirmatory test on 360 reserved scenes (v1.2): snapshot minus CBBA-PR TCR (pp), without (H1) and with (H2) a one-tick delay.}
\label{tab:v12}
\begin{tabular}{@{}lcc@{}}
\toprule
Cell & No delay (H1) & One-tick delay (H2)\\
\midrule
""" + "\n".join(L) + "\n\\midrule\n" + \
    f"Pooled & {ms(m1, s1)} & {ms(m2, s2)}\\\\\n" + \
    f"95\\% CI & [{H1['lower']:.2f}, {H1['upper']:.2f}] & [{H2['lower']:.2f}, {H2['upper']:.2f}]\\\\\n" + \
    "Holm-adjusted $p$ & $<10^{-4}$ & $<10^{-4}$\\\\\n\\midrule\n" + \
    f"CBBA-PR TCR (\\%) & {A['pr']['TCR']:.2f} & {A['pr_d1']['TCR']:.2f}\\\\\n" + \
    f"Never / lost, PR (\\%) & {nl('pr')} & {nl('pr_d1')}\\\\\n" + \
    f"Never / lost, snap.~(\\%) & {nl('snap')} & {nl('snap_d1a')}\\\\\n" + \
    f"Call (s), med./p95/max & {tm('snap')} & {tm('snap_d1a')}\\\\\n" + \
    f"Calls / fallbacks & {A['snap']['calls']:,} / {A['snap']['fallbacks']} & {A['snap_d1a']['calls']:,} / {A['snap_d1a']['fallbacks']}\\\\\n" + \
    "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
open('../../paper_p5/manuscript/tables/v12.tex', 'w').write(tex)
print(tex)
