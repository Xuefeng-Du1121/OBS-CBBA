"""Analysis and LaTeX tables of the revision studies (2026-11-19). Run after rev_run.py outputs exist."""
import json, os
import numpy as np, pandas as pd
from boot import strat_boot

HERE = os.path.dirname(os.path.abspath(__file__))
T = 'tables'
E = pd.read_csv(f'{HERE}/../20261024/p5_results_final/episodes.csv')
E = E[(E.module == 'main') & (E.status == 'complete')]
SNAP = pd.concat([pd.read_csv(f'{HERE}/../20261110/p5_eswa_extra/r2r3_out/snapshot__main__{c}_rows.csv')
                  for c in sorted(E.cell.unique())])
res = {}


def load(arm, tag='main', d='out'):
    p = f'{HERE}/{d}/{arm}__{tag}_rows.csv'
    return pd.read_csv(p) if os.path.exists(p) else None


def per_scene(df, tcr='TCR'):
    return df[['cell', 'seed']].assign(v=100 * df[tcr].values)


PR = E[E.arm == 'cbba-pr'][['cell', 'seed', 'TCR']]
REF = {'pr': PR, 'snap': SNAP[['cell', 'seed', 'TCR']]}


def contrast(a, b):
    m = a.merge(b, on=['cell', 'seed'], suffixes=('_a', '_b'))
    m['d'] = 100 * (m.TCR_a - m.TCR_b)
    est, lo, hi, p = strat_boot(m, 'd')
    return dict(est=est, lo=lo, hi=hi, p=p, n=len(m), wins=int((m.d > 1e-9).sum()), ties=int((abs(m.d) <= 1e-9).sum()))


def calls(df):
    s = [x for c in df.call_seconds for x in json.loads(c)]
    e = [x for c in df.call_evals for x in json.loads(c)] if 'call_evals' in df else []
    out = dict(med_s=float(np.median(s)) if s else 0., p95_s=float(np.percentile(s, 95)) if s else 0.,
               max_s=float(max(s)) if s else 0.)
    if e:
        out.update(med_e=float(np.median(e)), p95_e=float(np.percentile(e, 95)))
    return out


# ------------------------------------------------ reproduction checks
chk = {}
snap = load('snap')
for name, df, ref in [('snap718', load('snap', 'main718', 'out718'), SNAP), ('pr718', load('pr', 'main718', 'out718'), PR)]:
    if df is not None:
        m = df.merge(ref[['seed', 'TCR']], on='seed', suffixes=('', '_ref'))
        chk[name] = dict(n=len(m), equal=int((m.TCR == m.TCR_ref).sum()))
res['repro'] = chk

# ------------------------------------------------ decomposition
ARMS = [('snap', 'Paper set'), ('amount', 'Amount only'), ('order', 'Re-insertion only'),
        ('static', 'Paper set, static score'), ('ms32', 'Multi-start, static score'),
        ('ms128', 'Multi-start, static score'), ('apr', 'APR adapter'), ('snapapr', 'Paper set and APR'),
        ('msa64', 'MSA, 64 scenarios')]
dec = {}
for arm, label in ARMS:
    if arm == 'snap':
        s718 = load('snap', 'main718', 'out718')
        df = s718 if (s718 is not None and len(s718) == 360) else SNAP
    else:
        df = load(arm)
    if df is None:
        continue
    if len(df) < 360:
        continue
    r = dict(label=label, tcr=100 * df.groupby('cell').TCR.mean().mean())
    r['vs_pr'] = contrast(df[['cell', 'seed', 'TCR']], PR)
    if arm != 'snap':
        r['vs_snap'] = contrast(df[['cell', 'seed', 'TCR']], SNAP[['cell', 'seed', 'TCR']])
    if 'call_seconds' in df and arm not in ('msa64',):
        r.update(calls(df))
    if 'route_evals' in df:
        r['evals_per_scene'] = float(df.route_evals.mean())
        r['evals_per_call'] = float((df.route_evals / df.calls.clip(lower=1)).mean())
    if 'selected' in df and arm not in ('apr', 'msa64') and len(df.selected.iloc[0]) > 2:
        sel = pd.Series([x for c in df.selected for x in json.loads(c)])
        sel = sel.replace({'cheapest': 'deadline', 'cbba-pr': 'pr2'})
        r['share'] = sel.value_counts(normalize=True).round(4).to_dict()
    rel = df.n_released if 'n_released' in df else df.total
    r['never'] = 100 * (df.n_never_assigned / rel).groupby(df.cell).mean().mean()
    r['failed'] = 100 * (df.n_assigned_not_completed / rel).groupby(df.cell).mean().mean()
    r['decision_s_per_scene'] = float(df.decision_seconds.mean())
    dec[arm] = r
res['decomposition'] = dec
# per-N gains of the decomposition arms
byN = {}
for arm in ('snap', 'amount', 'order', 'union'):
    df = load(arm) if arm != 'snap' else SNAP
    if df is None or len(df) < 360:
        continue
    m = df[['cell', 'seed', 'TCR']].merge(PR, on=['cell', 'seed'], suffixes=('_a', '_b'))
    m['N'] = m.cell.str.extract(r'_n(\d+)_')[0].astype(int)
    byN[arm] = {int(n): round(float((100 * (g.TCR_a - g.TCR_b)).groupby(g.cell).mean().mean()), 2) for n, g in m.groupby('N')}
res['byN'] = byN

# ------------------------------------------------ penalty
def J(df, lam, rel, on, fail):
    return 100 * (on - lam * fail) / rel
PRr = E[E.arm == 'cbba-pr'].assign(rel=lambda d: d.total, on=lambda d: d.completed_on_time,
                                    fail=lambda d: d.total - d.completed_on_time - d.loss_never_assigned)
K8 = E[E.arm == 'o_obs-cbba'].assign(rel=lambda d: d.total, on=lambda d: d.completed_on_time,
                                     fail=lambda d: d.total - d.completed_on_time - d.loss_never_assigned)
def rows_of(df):
    return df.assign(rel=df.n_released, on=df.n_on_time, fail=df.n_assigned_not_completed)
SN = SNAP.assign(rel=SNAP.total, on=(SNAP.TCR * SNAP.total).round(), fail=SNAP.n_assigned_not_completed)
pen = {}
cands = [('k8', K8), ('snap', SN)]
for arm in ('pen05', 'pen10'):
    df = load(arm)
    if df is not None and len(df) == 360:
        cands.append((arm, rows_of(df)))
for name, df in cands:
    out = {}
    for lam in (0, .25, .5, 1.):
        m = df[['cell', 'seed', 'rel', 'on', 'fail']].merge(PRr[['cell', 'seed', 'rel', 'on', 'fail']],
                                                            on=['cell', 'seed'], suffixes=('_a', '_b'))
        m['d'] = J(m, lam, m.rel_a, m.on_a, m.fail_a) - J(m, lam, m.rel_b, m.on_b, m.fail_b)
        est, lo, hi, p = strat_boot(m, 'd')
        out[lam] = [est, lo, hi, p]
    pen[name] = out
res['penalty'] = pen

# ------------------------------------------------ scale
sc = {}
pr_s, sn_s, apr_s = load('pr', 'scale'), load('snap', 'scale'), load('apr', 'scale')
sa_s = load('snapapr', 'scale')
if pr_s is not None and sn_s is not None and len(sn_s) == 90:
    for cell, g in sn_s.groupby('cell'):
        p = pr_s[pr_s.cell == cell]
        m = g.merge(p[['seed', 'TCR']], on='seed', suffixes=('', '_pr'))
        d = 100 * (m.TCR - m.TCR_pr)
        rng = np.random.default_rng(20261119)
        bs = d.values[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        r = dict(pr=100 * p.TCR.mean(), snap=100 * g.TCR.mean(), d=d.mean(), lo=np.percentile(bs, 2.5),
                 hi=np.percentile(bs, 97.5), wins=int((d > 1e-9).sum()), fallbacks=int(g.fallbacks.sum()),
                 calls=int(g.calls.sum()), **calls(g))
        if apr_s is not None and len(apr_s) == 90:
            a = apr_s[apr_s.cell == cell]
            r['apr'] = 100 * a.TCR.mean()
            c3 = a.merge(g[['seed', 'TCR']], on='seed', suffixes=('', '_sn'))
            d3 = 100 * (c3.TCR - c3.TCR_sn)
            b3 = d3.values[rng.integers(0, len(d3), (10000, len(d3)))].mean(1)
            r.update(d_apr_snap=d3.mean(), lo_apr_snap=np.percentile(b3, 2.5), hi_apr_snap=np.percentile(b3, 97.5))
        if sa_s is not None and len(sa_s) == 90:
            b = sa_s[sa_s.cell == cell].merge(p[['seed', 'TCR']], on='seed', suffixes=('', '_pr'))
            db = 100 * (b.TCR - b.TCR_pr)
            bb = db.values[rng.integers(0, len(db), (10000, len(db)))].mean(1)
            r.update(snapapr=100 * b.TCR.mean(), d_sa=db.mean(), lo_sa=np.percentile(bb, 2.5),
                     hi_sa=np.percentile(bb, 97.5), fb_sa=int(sa_s[sa_s.cell == cell].fallbacks.sum()),
                     max_sa=calls(sa_s[sa_s.cell == cell])['max_s'], calls_sa=int(sa_s[sa_s.cell == cell].calls.sum()))
            if apr_s is not None and len(apr_s) == 90:
                c2 = sa_s[sa_s.cell == cell].merge(a[['seed', 'TCR']], on='seed', suffixes=('', '_apr'))
                d2 = 100 * (c2.TCR - c2.TCR_apr)
                b2 = d2.values[rng.integers(0, len(d2), (10000, len(d2)))].mean(1)
                r.update(d_sa_apr=d2.mean(), lo_sa_apr=np.percentile(b2, 2.5), hi_sa_apr=np.percentile(b2, 97.5))
        sc[cell] = r
res['scale'] = sc
json.dump(res, open(f'{HERE}/rev_results.json', 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=lambda x: round(float(x), 3)))


# ================================================ LaTeX tables
def sg(x, d=2):
    s = f'{abs(x):.{d}f}'
    return ('$-$' if x < 0 and float(s) != 0 else '$+$') + s
def iv(lo, hi, d=2):
    f = lambda v: ('$-$' if v < 0 and float(f'{abs(v):.{d}f}') != 0 else '') + f'{abs(v):.{d}f}'
    return f'[{f(lo)}, {f(hi)}]'
def kfmt(x):
    return f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'

rows = []
for arm in ('snap', 'amount', 'order', 'static', 'ms32', 'apr', 'msa64'):
    if arm not in dec:
        continue
    r = dec[arm]; v = r['vs_pr']
    vs = r.get('vs_snap')
    ev = kfmt(r['med_e']) if 'med_e' in r else ('--' if arm != 'snap' else kfmt(res.get('snap_evals', 0)) if res.get('snap_evals') else '--')
    rows.append(f"{r['label']} & {r['tcr']:.2f} & {sg(v['est'])} {iv(v['lo'], v['hi'])} & "
                f"{(sg(vs['est']) + ' ' + iv(vs['lo'], vs['hi'])) if vs else '--'} & {ev}\\\\")
SETS = {'snap': r'\{PR2, deadline, regret, slack\}', 'amount': r'\{PR2, PR4, PR-all\}',
        'order': r'\{PR-all, deadline, regret, slack\}'}
if rows:
    tex = r"""\begin{table*}[pos=tb]
\centering\footnotesize
\caption{Decomposition of the revision set and comparison baselines on the 360 main scenes (post hoc). Scored variants use the snapshot future, a 25-s budget and the CBBA-PR fallback; the APR adapter and MSA run without a budget. Gains in pp with 95\% CIs; evaluations: mean route evaluations per call, millions.}\label{tab:decomp}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{@{}lllrrr@{}}
\toprule
Variant & Candidates & TCR (\%) & vs CBBA-PR & vs paper set & Evaluations (M)\\
\midrule
"""
    lines = []
    for arm in ('snap', 'amount', 'order', 'static', 'ms32', 'ms128', 'apr', 'snapapr', 'msa64'):
        if arm not in dec:
            continue
        r = dec[arm]; v = r['vs_pr']; vs = r.get('vs_snap')
        cand = {'snap': SETS['snap'], 'amount': SETS['amount'], 'order': SETS['order'],
                'static': SETS['snap'] + ', immediate plan', 'ms32': r'paper set $+$ 29 random orders, immediate plan',
                'ms128': r'paper set $+$ 125 random orders, immediate plan', 'snapapr': r'paper set $+$ APR revision',
                'apr': 'protected repair, one-at-a-time release', 'msa64': '64 scenarios, next-task consensus'}[arm]
        ev = f"{r['evals_per_call']/1e6:.2f}"
        lines.append(f"{r['label']} & {cand} & {r['tcr']:.2f} & {sg(v['est'])} {iv(v['lo'], v['hi'])} & "
                     f"{(sg(vs['est']) + ' ' + iv(vs['lo'], vs['hi'])) if vs else '--'} & {ev}\\\\")
    tex += '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    open(f'{T}/decomp.tex', 'w').write(tex)

# penalty table
if pen:
    lab = {'k8': r'\method, $K=8$ (TCR score)', 'snap': 'Snapshot (TCR score)',
           'pen05': r'Snapshot, score $J_{0.5}$', 'pen10': r'Snapshot, score $J_{1}$'}
    lines = []
    for k in ('k8', 'snap', 'pen05', 'pen10'):
        if k not in pen:
            continue
        cells = [f"{sg(pen[k][l][0])} {iv(pen[k][l][1], pen[k][l][2])}" for l in (0, .25, .5, 1.)]
        lines.append(lab[k] + ' & ' + ' & '.join(cells) + r'\\')
    tex = r"""\begin{table*}[pos=tb]
\centering\footnotesize
\caption{Gain over CBBA-PR in the penalized completion $J_\lambda$ (pp of released tasks, 95\% CIs), where each task that was assigned but not completed costs $\lambda$ tasks; 360 main scenes, post hoc. The first two rows score with TCR; the last two put the penalty in the score.}\label{tab:penalty}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{@{}lllll@{}}
\toprule
Planner & $\lambda=0$ (TCR) & $\lambda=0.25$ & $\lambda=0.5$ & $\lambda=1$\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    open(f'{T}/penalty.tex', 'w').write(tex)

# scale table
if sc:
    lines = []
    for cell in ('u10_n200_high', 'u12_n240_high', 'u12_n360_high'):
        if cell not in sc:
            continue
        r = sc[cell]; u, n, _ = cell.split('_')
        sa = f"{r['snapapr']:.2f}" if 'snapapr' in r else '--'
        fb = f"{r['fallbacks']}/{r['fb_sa']}" if 'fb_sa' in r else f"{r['fallbacks']}"
        lines.append(f"{u[1:]} & {n[1:]} & {r['pr']:.2f} & {r.get('apr', float('nan')):.2f} & {r['snap']:.2f} & {sa} & "
                     f"{sg(r['d'])} {iv(r['lo'], r['hi'])} & {r['wins']}/30 & {r['med_s']:.2f} & {r['p95_s']:.2f} & "
                     f"{r['max_s']:.1f} & {kfmt(r['med_e'])} & {fb}\\\\")
    tex = r"""\begin{table*}[pos=tb]
\centering\footnotesize
\caption{Scale study on 90 new scenes (high arrivals, 30 per cell, post hoc): TCR of CBBA-PR, the APR adapter, the snapshot configuration and the paper set with APR added (Set+APR; 25-s budget, CBBA-PR fallback), the snapshot gain over CBBA-PR with 95\% CI, scenes won, per-call time (s) and route evaluations of the snapshot configuration, and budget fallbacks of the snapshot configuration and of Set+APR.}\label{tab:scale}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}rrrrrrlrrrrrr@{}}
\toprule
$U$ & $N$ & CBBA-PR & APR & Snapshot & Set+APR & Gain (pp) & Wins & Median & p95 & Max & Evaluations & Fallbacks\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    open(f'{T}/scale.tex', 'w').write(tex)
print('tables written')
