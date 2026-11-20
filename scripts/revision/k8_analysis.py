"""Branch-selection distribution and selection stability against K (archived K=8 traces)."""
import json, itertools
import numpy as np, pandas as pd
E = pd.read_csv('k8_events.csv')
E['sel'] = E.selected.replace({'cheapest': 'deadline', 'cbba-pr': 'pr'})
E['U'] = E.cell.str.extract(r'u(\d+)').astype(float)
m = E[E.module == 'main'].copy()
m['N'] = m.cell.str.extract(r'_n(\d+)_').astype(int)
m['arr'] = m.cell.str.split('_').str[-1]
out = {}
out['events'] = len(m)
out['share'] = m.sel.value_counts(normalize=True).round(4).to_dict()
out['share_by_N'] = {int(k): g.sel.value_counts(normalize=True).round(4).to_dict() for k, g in m.groupby('N')}
out['share_by_U'] = {int(k): g.sel.value_counts(normalize=True).round(4).to_dict() for k, g in m.groupby('U')}
out['share_by_arr'] = {k: g.sel.value_counts(normalize=True).round(4).to_dict() for k, g in m.groupby('arr')}
m['tbin'] = pd.cut(m.time, [-1, 0, 30, 60, 100], labels=['t=0', '1-30', '31-60', '61-99'])
out['share_by_time'] = {str(k): g.sel.value_counts(normalize=True).round(4).to_dict() for k, g in m.groupby('tbin', observed=True)}
r = m[m.module == 'robustness'] if False else E[E.module == 'robustness']
out['share_robustness'] = r.sel.value_counts(normalize=True).round(4).to_dict()
# stability against K: subsets of the 8 archived futures
V = np.array([json.loads(c) for c in m.completed])  # events x 5 actions x 8 futures
V = V[:, [0, 1, 3, 4], :]  # drop the duplicate 'deadline' (equals 'cheapest')
full = V.sum(2).argmax(1)  # first maximizer, PR first
stab = {}
for k in (1, 2, 4):
    agree, loss, prk = [], [], []
    for sub in itertools.combinations(range(8), k):
        s = V[:, :, list(sub)].sum(2).argmax(1)
        agree.append((s == full).mean())
        best = V.sum(2)[np.arange(len(V)), full] / 8
        got = V.sum(2)[np.arange(len(V)), s] / 8
        loss.append((best - got).mean())
        prk.append((s == 0).mean())
    stab[k] = dict(agree=round(float(np.mean(agree)), 4), mean_loss_tasks=round(float(np.mean(loss)), 4),
                   keep_pr=round(float(np.mean(prk)), 4))
stab[8] = dict(agree=1.0, mean_loss_tasks=0.0, keep_pr=round(float((full == 0).mean()), 4))
out['stability_vs_K'] = stab
# paired differences: spread of D_k = V(a) - V(PR) within an event, against its range bound M
D = V[:, 1:, :] - V[:, :1, :]
sd = D.std(2, ddof=1); rng = np.abs(D).max(2)
out['paired_sd_median'] = round(float(np.median(sd)), 3)
out['paired_absmax_median'] = round(float(np.median(rng)), 3)
out['value_sd_median'] = round(float(np.median(V.std(2, ddof=1))), 3)
# events where the selection is a revision: margin over PR in tasks per future
rev = full != 0
marg = (V.sum(2)[np.arange(len(V)), full] - V.sum(2)[:, 0]) / 8
out['revision_margin_median'] = round(float(np.median(marg[rev])), 3)
json.dump(out, open('k8_analysis.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
