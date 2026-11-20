"""Observation features and the branch network used by the event trigger."""
import math
import time
import numpy as np
import torch
from torch import nn
from .commitment import event_now, release_unstarted
from .decision_headroom import ORDERS, allocate
from .maritime_planning import observe_maritime, unpack_maritime, estimate_drift, cell_indices
from .observable_order import select_order
from .policies import Controller

SCHEMA = 'wp3b-observation-set-v1'
TASK_FIELDS = ('x', 'y', 'deadline_remaining', 'assigned', 'ever_assigned',
               'started', 'completed', 'expired', 'active', 'owner_x', 'owner_y',
               'owner_energy', 'available_fraction', 'nearest_uav_distance',
               'drift_x', 'drift_y', 'drift_sd_x', 'drift_sd_y', 'drift_count')
UAV_FIELDS = ('x', 'y', 'energy', 'distance', 'queue_size', 'active', 'sensing',
              'active_x', 'active_y', 'active_deadline', 'available_fraction')
GLOBAL_FIELDS = ('time', 'remaining_time', 'released_count', 'fleet_size',
                 'completed_fraction', 'pending_fraction', 'speed', 'loss',
                 'initial_window', 'arrival_window', 'arrival_medium', 'arrival_high',
                 'poisson', 'drift_x', 'drift_y', 'drift_sd_x', 'drift_sd_y', 'drift_count')


def observation_features(observation):
    """Only immutable telemetry enters here; no env, seed, or truth config."""
    data, _ = unpack_maritime(observation)
    b = data['base']; c = b['config']; t = b['time']
    drift = estimate_drift(observation)
    points = np.asarray(b['task_positions'], dtype=float).reshape(-1, 2)
    fleet = np.asarray(b['position'], dtype=float)
    n = len(points); horizon = max(c['horizon'], 1)
    available = np.asarray(b['available'], dtype=float).reshape(n, len(fleet))
    regions = cell_indices(points, drift['grid']) if n else []
    tasks = []
    for j, p in enumerate(points):
        owner = b['owner'][j]; r = drift['regions'][regions[j]]
        assigned = owner >= 0
        tasks.append([*p, (b['deadline'][j]-t)/horizon, float(assigned),
            float(b['ever_assigned'][j]), float(b['commitment']['started_at'][j] >= 0),
            float(b['completed_at'][j] >= 0), float(b['deadline'][j] <= t),
            float(j in b['active']), *(fleet[owner] if assigned else (0., 0.)),
            b['energy'][owner]/c['energy'] if assigned else 0., available[j].mean(),
            np.linalg.norm(fleet-p, axis=1).min(),
            *np.asarray(r['mean'])*horizon, *np.asarray(r['sd'])*horizon, r['count']/1000.])
    uavs = []
    for u, p in enumerate(fleet):
        j = b['active'][u]
        uavs.append([*p, b['energy'][u]/c['energy'], b['distance'][u]/c['max_distance'],
            len(b['queues'][u])/100., float(j >= 0), b['sensing'][u]/c['sensing_steps'],
            *(points[j] if j >= 0 else (0., 0.)),
            (b['deadline'][j]-t)/horizon if j >= 0 else 0.,
            available[:, u].mean() if n else 0.])
    g = drift['global_stats']
    global_values = [t/horizon, (horizon-t)/horizon, n/100., len(fleet)/10.,
        sum(x >= 0 for x in b['completed_at'])/max(n, 1),
        sum(b['owner'][j] < 0 and b['completed_at'][j] < 0 and b['deadline'][j] > t
            for j in range(n))/max(n, 1), c['speed'], c['loss'],
        c['initial_window']/horizon, c['arrival_window']/horizon,
        float(c['arrival'] == 'medium'), float(c['arrival'] == 'high'),
        float(c.get('arrival_process', 'deterministic') == 'poisson'),
        *np.asarray(g['mean'])*horizon, *np.asarray(g['sd'])*horizon, g['count']/1000.]
    return dict(schema=SCHEMA, tasks=tasks, uavs=uavs, global_features=global_values)


def feature_batch(features, device='cpu'):
    """Pad ragged sets with explicit masks; empty sets are supported."""
    if not features or any(f['schema'] != SCHEMA for f in features):
        raise ValueError('Empty batch or unknown feature schema')
    result = {}
    for key, fields in (('tasks', TASK_FIELDS), ('uavs', UAV_FIELDS)):
        width = max(1, max(len(f[key]) for f in features))
        values = np.zeros((len(features), width, len(fields)), dtype=np.float32)
        mask = np.zeros((len(features), width), dtype=bool)
        for i, f in enumerate(features):
            n = len(f[key])
            if n:
                values[i, :n] = f[key]; mask[i, :n] = True
        result[key] = torch.as_tensor(values, device=device)
        result[key+'_mask'] = torch.as_tensor(mask, device=device)
    result['global_features'] = torch.tensor([f['global_features'] for f in features],
                                             dtype=torch.float32, device=device)
    return result


class BranchNetwork(nn.Module):
    """Deep Sets with mean/max pooling; invariant to task and UAV token order."""
    def __init__(self, hidden=64):
        super().__init__(); self.hidden = hidden
        def encoder(size):
            return nn.Sequential(nn.Linear(size, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.task_encoder = encoder(len(TASK_FIELDS))
        self.uav_encoder = encoder(len(UAV_FIELDS))
        self.trunk = encoder(4*hidden+len(GLOBAL_FIELDS))
        self.logits = nn.Linear(hidden, len(ORDERS))
        self.gaps = nn.Linear(hidden, len(ORDERS))

    @staticmethod
    def pool(tokens, mask):
        count = mask.sum(1, keepdim=True).clamp_min(1)
        mean = (tokens*mask[..., None]).sum(1)/count
        maximum = tokens.masked_fill(~mask[..., None], -torch.inf).amax(1)
        maximum = torch.where(mask.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
        return torch.cat((mean, maximum), dim=-1)

    def forward(self, batch, temperature=1.):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be finite and positive')
        tasks = self.pool(self.task_encoder(batch['tasks']), batch['tasks_mask'])
        uavs = self.pool(self.uav_encoder(batch['uavs']), batch['uavs_mask'])
        h = self.trunk(torch.cat((tasks, uavs, batch['global_features']), dim=-1))
        logits = self.logits(h)
        return dict(logits=logits, confidence=(logits/temperature).softmax(-1).amax(-1),
                    gaps=self.gaps(h))


def distillation_loss(output, labels, mean_scores, gap_weight=.1):
    # Score gaps in units of ten completed tasks; no hidden inputs to the model.
    gaps = (mean_scores.amax(-1, keepdim=True)-mean_scores)/10.
    return (nn.functional.cross_entropy(output['logits'], labels)
            + gap_weight*nn.functional.mse_loss(output['gaps'], gaps))


def teacher_record(row):
    """Keep original teacher provenance/outcomes and add deterministic features."""
    from .maritime_planning import MaritimeObservation
    import json
    observation = MaritimeObservation(json.dumps(row['observation'], sort_keys=True, allow_nan=False))
    if row['observation_sha256'] != observation.fingerprint() or row['orders'] != list(ORDERS):
        raise ValueError('Teacher observation or branch identity mismatch')
    scores = [sum(x)/row['k'] for x in row['completed']]
    winner = max(range(len(ORDERS)), key=lambda i: scores[i])
    if (len(scores) != 4 or any(len(x) != row['k'] for x in row['completed'])
            or row['selected'] != ORDERS[winner]
            or row['sums'] != [sum(x) for x in row['completed']]):
        raise ValueError('Inconsistent teacher outcomes')
    return dict(**row, schema=SCHEMA, features=observation_features(observation),
                mean_scores=scores, label=winner)


class AmortizedSelector:
    def __init__(self, model, threshold=0., temperature=1.):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Confidence threshold must be in [0, 1]')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be positive')
        self.model = model.eval(); self.threshold = threshold; self.temperature = temperature

    def __call__(self, observation, check=lambda: None):
        check(); features = observation_features(observation)
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            output = self.model(feature_batch([features], device), self.temperature)
        logits = output['logits'][0]
        confidence = float(output['confidence'][0].item())
        finite = bool(torch.isfinite(logits).all()) and math.isfinite(confidence)
        # Nonfinite predictions fail closed even in network-only mode.
        fallback = not finite or confidence < self.threshold
        network_order = ORDERS[int(logits.argmax())] if finite else None
        teacher = select_order(observation, 8, check) if fallback else None
        return dict(time=observation.as_dict()['base']['time'], selected=teacher['selected'] if fallback else network_order,
                    network_selected=network_order, confidence=confidence if finite else None,
                    fallback=fallback, teacher=teacher)


def run(env, selector, check=lambda: None):
    """Same event/withdrawal/cheapest non-event loop as observable_order.run."""
    observe_maritime(env)
    planner = Controller(env.cfg, 'cheapest'); previous = []; traces = []; events = 0
    seconds = dict(selection=0., allocation=0., planning=0.)
    while not env.done:
        check()
        if env.time < env.cfg.horizon:
            event = event_now(env, previous); events += int(event)
            if event:
                release_unstarted(env); start = time.perf_counter()
                row = selector(observe_maritime(env), check)
                row['seconds'] = time.perf_counter()-start
                seconds['selection'] += row['seconds']; traces.append(row)
                order = row['selected']
            else:
                order = 'cheapest'
            start = time.perf_counter(); allocate(env, order, check=check)
            seconds['allocation'] += time.perf_counter()-start
            start = time.perf_counter(); planner.plan(env)
            seconds['planning'] += time.perf_counter()-start
        _, previous = env.tick()
    fallbacks = sum(r['fallback'] for r in traces)
    return (dict(events=events, decisions=len(traces), traces=traces),
            dict(seconds=seconds, decision_seconds=seconds['selection'], fallbacks=fallbacks,
                 fallback_rate=fallbacks/len(traces) if traces else 0.))
