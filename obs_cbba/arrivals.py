"""Public arrival law; realized future counts are hidden in Poisson mode."""
import math
import numpy as np


def release_schedule(cfg, rng=None, after=-1, *, rate_multiplier=1.):
    if not math.isfinite(rate_multiplier) or rate_multiplier <= 0:
        raise ValueError('Positive finite arrival rate multiplier required')
    if rate_multiplier != 1 and cfg.arrival_process != 'poisson':
        raise ValueError('Arrival rate mismatch requires Poisson arrivals')
    interval, count = {'none': (1, 0), 'low': (20, 5), 'medium': (10, 5), 'high': (10, 10)}[cfg.arrival]
    if cfg.arrival_process == 'poisson' and rng is None:
        raise ValueError('Poisson arrivals require an independent RNG')
    release = [0] * cfg.tasks if after < 0 else []
    for t in range(interval, min(cfg.arrival_until, cfg.horizon-1)+1, interval):
        if t <= after:
            continue
        n = int(rng.poisson(count * rate_multiplier)) if cfg.arrival_process == 'poisson' else count
        release.extend([t] * n)
    return np.asarray(release, dtype=np.int64)


def deadlines(cfg, release):
    return release + np.where(release == 0, cfg.initial_window, cfg.arrival_window)
