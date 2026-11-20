"""Commitment-failure penalty: J = (on-time - lam * assigned-not-completed) / released, in pp."""
import json, pandas as pd, numpy as np
from boot import strat_boot
E = pd.read_csv('../20261024/p5_results_final/episodes.csv')
E = E[(E.module == 'main') & (E.status == 'complete')]
def prep(df, released, ontime, failed):
    return pd.DataFrame(dict(cell=df.cell.values, seed=df.seed.values, rel=released.values,
                             on=ontime.values, fail=failed.values))
def arm(a):
    d = E[E.arm == a]
    return prep(d, d.total, d.completed_on_time, d.total - d.completed_on_time - d.loss_never_assigned)
out = {}
lams = [0, .25, .5, 1.]
def contrast(A, B, tag):
    m = A.merge(B, on=['cell', 'seed'], suffixes=('_a', '_b'))
    res = {}
    for lam in lams:
        ja = 100 * (m.on_a - lam * m.fail_a) / m.rel_a
        jb = 100 * (m.on_b - lam * m.fail_b) / m.rel_b
        m['d'] = ja - jb
        est, lo, hi, p = strat_boot(m, 'd')
        res[lam] = [round(est, 2), round(lo, 2), round(hi, 2), p]
    # break-even weight: dTCR / dfail-share, cellwise equal weights
    dt = (100 * (m.on_a / m.rel_a - m.on_b / m.rel_b)).groupby(m.cell).mean().mean()
    df = (100 * (m.fail_a / m.rel_a - m.fail_b / m.rel_b)).groupby(m.cell).mean().mean()
    res['break_even'] = round(dt / df, 3); res['dfail_pp'] = round(df, 2)
    out[tag] = res
contrast(arm('o_obs-cbba'), arm('cbba-pr'), 'main_K8_vs_PR')
# v1.2 snapshot
V = '../20261112_v12/out/'
def v12(a):
    d = pd.read_csv(V + f'v12_{a}__all_rows.csv')
    rel = d.n_never_assigned * 0 + d.total
    on = d.TCR * d.total
    return prep(d, d.total, (d.TCR * d.total).round(), d.n_assigned_not_completed)
contrast(v12('snap'), v12('pr'), 'v12_snap_vs_PR')
json.dump(out, open('lambda_sens.json', 'w'), indent=1); print(json.dumps(out, indent=1))
