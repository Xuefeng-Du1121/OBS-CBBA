"""Frozen pretraining/validation percentile scales for regional feedback.

Components are executed step length (normalized spatial units), interaction
deviation (same units; zero when no interaction), and active-task expiry (0/1).
Those measurement choices remain explicit reconstruction assumptions.
"""
from collections import defaultdict, deque
import hashlib
import json
import numpy as np


COMPONENTS = ['path_length', 'deviation', 'infeasibility']


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_calibration(artifact):
    if not isinstance(artifact, dict) or artifact.get('schema') != 'regional-percentile-v1':
        raise ValueError('Unsupported regional calibration artifact')
    if artifact.get('components') != COMPONENTS:
        raise ValueError('Regional component order differs')
    scales = np.asarray(artifact.get('scales'), dtype=float)
    sigma = artifact.get('sigma_ref', float('nan'))
    if scales.shape != (3,) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Positive finite component scales required')
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Positive finite reference scale required')
    if artifact.get('sha256') != digest({k:v for k,v in artifact.items() if k != 'sha256'}):
        raise ValueError('Calibration digest mismatch')
    return artifact


def normalize(components, artifact):
    values = np.asarray(components, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Three nonnegative finite feedback components required')
    return np.clip(values / np.asarray(artifact['scales']), 0., 1.)


def fit_calibration(train_rows, validation_rows, history, provenance, epsilon=1e-8):
    """Fit component q95 on train only, aggregate q95 on validation only.

Rolling means include observations within the last W physical steps, separately
for each episode/UAV/region. Episodes never share feedback history. A zero q95
uses the declared epsilon floor; future positive observations then saturate.
"""
    if history < 1 or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Positive history and epsilon required')
    for rows, split in [(train_rows, 'train'), (validation_rows, 'validation')]:
        if not rows or any(r['split'] != split for r in rows):
            raise ValueError(f'Nonempty {split}-only calibration rows required')
        for r in rows:
            x = np.asarray(r['components'], dtype=float)
            if x.shape != (3,) or not np.isfinite(x).all() or (x < 0).any():
                raise ValueError('Invalid component observations')
            if r['time'] < 0:
                raise ValueError('Negative feedback time')
    train_seeds = {r['scenario_seed'] for r in train_rows}
    val_seeds = {r['scenario_seed'] for r in validation_rows}
    if train_seeds & val_seeds:
        raise ValueError('Calibration splits overlap')
    q95 = np.percentile([r['components'] for r in train_rows], 95, axis=0)
    scales = np.maximum(q95, epsilon)
    recent = defaultdict(deque)
    aggregates = []
    for r in sorted(validation_rows, key=lambda x:(x['scenario_seed'], x['time'], x['uav'], x['region'])):
        key = (r['scenario_seed'], r['uav'], r['region'])
        queue = recent[key]
        while queue and queue[0][0] <= r['time'] - history:
            queue.popleft()
        queue.append((r['time'], np.clip(np.asarray(r['components'])/scales, 0., 1.)))
        aggregates.append(float(np.mean([v for _,v in queue], axis=0).sum()))
    sigma95 = float(np.percentile(aggregates, 95))
    result = dict(schema='regional-percentile-v1', components=COMPONENTS,
        component_q95=q95.tolist(), scales=scales.tolist(), aggregate_q95=sigma95,
        sigma_ref=max(sigma95, epsilon), history=history, epsilon=epsilon,
        zero_q95_components=[COMPONENTS[i] for i,x in enumerate(q95) if x == 0],
        training_records=len(train_rows), validation_records=len(validation_rows),
        training_scenario_seeds=sorted(train_seeds), validation_scenario_seeds=sorted(val_seeds),
        data_sha256=digest(dict(train=train_rows, validation=validation_rows)),
        provenance=provenance, test_used=False)
    result['sha256'] = digest(result)
    return validate_calibration(result)
