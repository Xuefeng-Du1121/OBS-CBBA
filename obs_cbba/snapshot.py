"""In-process simulator checkpoints, NOT observations for an online policy.

Snapshots contain hidden exogenous futures. An online planner must construct its
own observed-information future sampler; restoring a snapshot is not sampling.
External controllers, models and Python/NumPy/Torch global RNGs are not owned by
the environment and are deliberately outside this API.
"""
from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvSnapshot:
    """Opaque checkpoint; callers must not inspect or mutate ``_state``.

    Each capture/restore makes an independent deep copy, including scenario,
    config, nested bandit queues, reward ledger and event history. Object aliases
    inside a checkpoint (e.g. bandit.cfg and env.cfg) are preserved.
    """

    version: int
    _state: dict

    @classmethod
    def capture(cls, env):
        return cls(1, deepcopy(vars(env)))

    def materialize(self):
        if self.version != 1:
            raise ValueError('Unsupported environment snapshot version')
        return deepcopy(self._state)
