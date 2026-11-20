"""Explicit event-based reconstruction of the manuscript RA/RP/RJ formulas.

Unit task value is realized on completion. Delay means an assigned task expiring
at its deadline (unit normalized penalty). Energy is actual fleet consumption,
including sensing, waiting and return. Each task incurs RP infeasibility once,
at its first reserve abort or expiry. These are declared reconstruction choices.
"""
import numpy as np


def reward_step(cfg, time, raw):
    values = {k:np.asarray(v, dtype=float) for k,v in raw.items()}
    if any(v.shape != (cfg.uavs,) or not np.isfinite(v).all() or (v < -1e-10).any()
           for v in values.values()):
        raise ValueError('Invalid per-UAV reward observations')
    allocation = dict(
        allocation_value=cfg.reward_value*values['completed'],
        allocation_energy=-cfg.reward_energy*values['energy']/cfg.energy,
        allocation_delay=-cfg.reward_delay*values['expired'],
        allocation_conflict=-cfg.reward_conflict*values['conflicts'])
    path = dict(
        path_success=cfg.reward_success*values['completed'],
        path_length=-cfg.reward_length*values['length'],
        path_deviation=-cfg.reward_deviation*np.minimum(values['deviation']/cfg.deviation_scale, 1.),
        path_infeasibility=-cfg.reward_failure*values['infeasible'])
    ra = sum(allocation.values())
    rp = sum(path.values())
    return dict(time=time, raw={k:v.tolist() for k,v in values.items()},
        allocation_terms={k:v.tolist() for k,v in allocation.items()},
        path_terms={k:v.tolist() for k,v in path.items()},
        allocation=float(ra.mean()), path=rp.tolist(),
        joint=float(cfg.allocation_weight*ra.mean()+(1-cfg.allocation_weight)*rp.mean()))


def transition_terms(ledger, cfg, layer, uav, stage):
    """Both joint actors optimize the same fleet-mean RJ time stream."""
    terms = {}
    if layer in ('upper','proposer') or stage == 'joint':
        for key in ledger[0]['allocation_terms']:
            a=np.array([r['allocation_terms'][key] for r in ledger]).mean(axis=1)
            terms[key]=a*(cfg.allocation_weight if stage=='joint' else 1.)
    if layer == 'lower' or stage == 'joint':
        for key in ledger[0]['path_terms']:
            a=np.array([r['path_terms'][key] for r in ledger])
            terms[key]=((1-cfg.allocation_weight)*a.mean(axis=1) if stage=='joint' else a[:,uav])
    return terms
