from dataclasses import asdict, dataclass, fields
from pathlib import Path
import json
import math


@dataclass
class Config:
    # Manuscript settings.
    tasks: int = 100
    uavs: int = 6
    speed: float = .05
    energy: float = 100.
    sensing_energy: float = 1.
    sensing_steps: int = 1
    initial_window: int = 50
    arrival_window: int = 20
    sigma: float = .005
    grid: int = 5
    history: int = 20
    dim: int = 128
    layers: int = 3
    lr: float = 1e-4
    batch_size: int = 512
    beta: float = 1.
    response_threshold: float = .7
    allocation_weight: float = .6
    pretrain_epochs: int = 60
    joint_epochs: int = 40
    transitions_per_epoch: int = 1280000
    upper_clip: float = 3.
    lower_clip: float = 1.
    upper_decay: float = .995
    # Explicit reconstruction assumptions.
    horizon: int = 100
    arrival_until: int = 60
    arrival_process: str = "deterministic"  # or independent Poisson counts per arrival slot
    arrival: str = "medium"
    return_steps: int = 30
    flight_energy: float = 10.  # per normalized distance, not per tick
    hover_energy: float = .02
    reserve: float = 10.
    max_distance: float = 8.
    sensing_radius: float = .01
    separation: float = .002
    depot_radius: float = .015
    observation_delay: int = 0
    loss: float = 0.
    nominal_speed: float = 0.
    mobility: str = "iid"  # iid, regional, switch
    maritime: dict = None  # opt-in physical forcing; never exported to a planner
    real_drift: dict = None  # explicit, hash-pinned local current subset
    rho: float = .2
    prior_cost: float = 0.
    region_budget: int = 5
    heads: int = 4
    gamma: float = .99
    entropy: float = .01
    distance_weight: float = .1
    target_tau: float = .01
    replay_size: int = 50000
    replay_alpha: float = .6
    replay_importance: float = .4
    update_every: int = 512
    # Fixed physical reference scales. Original percentile scalers are lost.
    deviation_scale: float = .02
    delay_scale: float = 20.
    reward_energy: float = .1
    reward_delay: float = .2
    reward_conflict: float = .2
    reward_length: float = 1.
    reward_deviation: float = .2
    reward_failure: float = 1.
    reward_value: float = 1.
    reward_success: float = 1.
    reward_protocol: str = 'legacy'
    validation_instances: int = 5
    # Explicit v4 reconstruction option: observed assignment completion calibrates
    # the absolute response score. Zero preserves the v3 executor-only objective.
    response_aux_weight: float = 0.
    upper_advantage_normalize: bool = False
    # Opt-in v6 fix: collect multiple complete episodes before optimizer updates.
    rollout_batching: bool = False
    # Opt-in v9: logistic exploration centered on the deployed response threshold.
    response_gate_mode: str = 'bernoulli'
    response_gate_temperature: float = .25
    # Explicit opt-in manuscript percentile protocol; old runs keep fixed scales.
    regional_cost_mode: str = 'fixed-reference'
    regional_calibration: dict = None
    # Physical switching contract only; legacy controllers never invoke it.
    switching_protocol: str = 'nonpreemptive'
    task_order: str = 'deadline'  # opt-in independent learned broadcast-task actor

    def validate(self):
        assert self.tasks > 0 and self.uavs > 0
        assert self.horizon > 0 and self.return_steps >= 0
        assert self.speed > 0 and self.energy > self.reserve >= 0
        assert self.flight_energy >= 0 and self.sensing_energy >= 0
        assert self.sensing_steps >= 1 and self.max_distance > 0
        assert 0 <= self.loss <= 1 and self.observation_delay >= 0
        assert 0 < self.rho <= 1 and self.grid >= 1
        assert 0 <= self.allocation_weight <= 1
        assert 0 <= self.response_threshold <= 1
        assert self.dim % self.heads == 0
        assert self.arrival in ("none", "low", "medium", "high")
        assert self.arrival_process in ("deterministic", "poisson")
        assert self.mobility in ("iid", "regional", "switch", "maritime-current", "maritime-wind", "maritime-anchored", "real-current")
        if self.mobility == 'real-current':
            from .real_drift import RealDriftParameters
            RealDriftParameters.from_config(self)
        elif self.real_drift is not None:
            raise ValueError('Real drift parameters require explicit real-current mode')
        if self.mobility.startswith('maritime-'):
            from .maritime import MaritimeParameters
            MaritimeParameters.from_config(self)
        elif self.maritime is not None:
            raise ValueError('Maritime parameters require an explicit maritime mode')
        assert self.deviation_scale > 0 and self.delay_scale > 0
        assert self.response_aux_weight >= 0
        assert self.reward_protocol in ('legacy', 'event-components-v1')
        assert self.switching_protocol in ('nonpreemptive', 'restart-service-v1')
        assert self.task_order in ('deadline', 'learned')
        if self.task_order=='learned':
            assert self.reward_protocol=='event-components-v1', 'Learned task order requires explicit reward components'
        for name in ('reward_value','reward_success','reward_energy','reward_delay',
                     'reward_conflict','reward_length','reward_deviation','reward_failure'):
            assert math.isfinite(getattr(self,name)) and getattr(self,name) >= 0
        assert self.batch_size > 0 and self.update_every > 0
        assert self.response_gate_mode in ('bernoulli', 'threshold-logistic')
        assert 0 < self.response_gate_temperature < float('inf')
        assert self.regional_cost_mode in ('fixed-reference', 'percentile')
        if self.regional_cost_mode == 'percentile':
            from .calibration import validate_calibration
            artifact = validate_calibration(self.regional_calibration)
            if artifact['history'] != self.history:
                raise ValueError('Calibration and deployment history differ')
            context = artifact.get('provenance', {}).get('collection_config')
            if context:
                for name in ('tasks','uavs','speed','energy','sigma','arrival','arrival_until',
                             'horizon','return_steps','grid','mobility','nominal_speed','loss',
                             'sensing_steps','sensing_radius','initial_window','arrival_window',
                             'flight_energy','hover_energy','sensing_energy','reserve','max_distance',
                             'observation_delay','separation','depot_radius'):
                    if context.get(name) != getattr(self,name):
                        raise ValueError(f'Calibration physical context differs: {name}')
        return self

    def dict(self):
        values = asdict(self)
        # Preserve legacy config serialization, scenario hashes and state IDs.
        if self.arrival_process == 'deterministic':
            values.pop('arrival_process')
        if self.real_drift is None:
            values.pop('real_drift')
        else:
            from .real_drift import RealDriftParameters
            values['real_drift'] = asdict(RealDriftParameters.from_config(self))
        if self.maritime is None:
            values.pop('maritime')
        else:
            from .maritime import MaritimeParameters
            # Persist all physical defaults, using JSON-stable vector lists.
            values['maritime'] = json.loads(json.dumps(asdict(MaritimeParameters.from_config(self)), allow_nan=False))
        return values

    @classmethod
    def load(cls, path=None, **overrides):
        values = json.loads(Path(path).read_text()) if path else {}
        values.update({k: v for k, v in overrides.items() if v is not None})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown settings: {unknown}")
        return cls(**values).validate()
