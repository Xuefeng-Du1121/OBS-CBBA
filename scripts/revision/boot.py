import numpy as np
def strat_boot(df, col, B=10000, seed=20261119, cell='cell'):
    """Cell-stratified paired bootstrap, equal cell weights. df: one row per scene with diff col."""
    rng = np.random.default_rng(seed)
    groups = [g[col].to_numpy() for _, g in df.groupby(cell)]
    est = np.mean([g.mean() for g in groups])
    reps = np.zeros(B)
    for g in groups:
        idx = rng.integers(0, len(g), (B, len(g)))
        reps += g[idx].mean(1)
    reps /= len(groups)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    p = (1 + np.sum(np.abs(reps - est) >= abs(est))) / (B + 1)
    return est, lo, hi, p
