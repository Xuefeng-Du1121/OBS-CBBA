"""Opt-in restart-service switching contract, not a recovered trained Q policy.

Continue defers the broadcast task at this decision; switch accepts it and puts
the interrupted task immediately behind it. Both values must refer to this same
state and these two actions. A legacy task-allocation Q is not such a critic.
"""
from copy import deepcopy
import hashlib
import json
import math
import numpy as np


def fingerprint(state):
    return hashlib.sha256(json.dumps(state,sort_keys=True,allow_nan=False).encode()).hexdigest()


def switch_decision(env, uav, task):
    """Return an observed-state/action pair, or None if switching is unavailable.

    No future positions, outcomes or random draws enter this representation.
    Full-route feasibility is an estimate at current observations, not a promise
    against later mobility, conflicts or hover delays.
    """
    c,s=env.cfg,env.scenario
    if c.switching_protocol!='restart-service-v1':return None
    if not 0<=uav<c.uavs or task not in env.pending():return None
    old=int(env.active[uav])
    if (old<0 or env.owner[old]!=uav or env.completed_at[old]>=0 or
        s.deadline[old]<=env.time or not env.queues[uav] or env.queues[uav][0]!=old):return None
    if env.last_switch_step[uav]==env.time:return None
    available=s.response_uniform[env.time,task]>=c.loss
    if not available[uav]:return None
    route=[int(task),*env.queues[uav]]
    estimate=env.route_estimate(uav,route,reset_service=True)
    if not estimate['feasible']:return None
    positions=env.observed_positions()
    known=np.flatnonzero(s.release<=env.time)
    state=dict(protocol=c.switching_protocol,config=c.dict(),time=env.time,
        task=int(task),uav=int(uav),current_task=old,
        tasks=[dict(id=int(j),position=positions[j].tolist(),release=int(s.release[j]),
                    deadline=int(s.deadline[j]),owner=int(env.owner[j]),
                    assigned_at=int(env.assigned_at[j]),completed_at=int(env.completed_at[j])) for j in known],
        position=env.position.tolist(),energy=env.energy.tolist(),distance=env.distance.tolist(),
        queues=deepcopy(env.queues),active=env.active.tolist(),sensing=env.sensing.tolist(),
        planned_interaction=env.planned_interaction.tolist(),history=deepcopy(env.history),
        regional_mean=env.bandit.mean.tolist(),regional_count=env.bandit.count.tolist(),
        available=available.tolist(),last_switch_step=env.last_switch_step.tolist())
    actions=[dict(kind='continue',route=list(env.queues[uav]),accept_new=False,reset_service=False,
                  estimate=env.route_estimate(uav,env.queues[uav])),
             dict(kind='switch',route=route,accept_new=True,reset_service=True,estimate=estimate)]
    # Include action semantics in identity: a value for another action pair is stale too.
    identity=fingerprint(dict(state=state,actions=actions))
    return dict(state_id=identity,state=state,actions=actions)


def score_switch(decision, critic):
    """Call one action-conditioned critic twice on identical observed state.

    Caller supplies a critic trained for this decision process. This function
    intentionally does not adapt the legacy allocation critic by changing task
    features, and does not supply an untrained default network.
    """
    if fingerprint(dict(state=decision['state'],actions=decision['actions']))!=decision['state_id']:
        raise ValueError('Modified switch decision')
    values=[float(critic(deepcopy(decision['state']),deepcopy(action))) for action in decision['actions']]
    if not all(math.isfinite(x) for x in values):raise ValueError('Nonfinite switch values')
    return dict(state_id=decision['state_id'],q_continue=values[0],q_switch=values[1])


def apply_switch(env, decision, values=None, willingness=None, *, diagnostic=False):
    """Atomically commit an eligible action or leave physical state unchanged.

    diagnostic=True is an explicit offline intervention, never a learned gate.
    It bypasses Q/willingness only; every physical/communication check remains.
    """
    u=decision['state']['uav'];j=decision['state']['task']
    fresh=switch_decision(env,u,j)
    if fresh is None or fresh['state_id']!=decision['state_id']:
        raise ValueError('Switch decision is stale or infeasible')
    if diagnostic:
        if values is not None or willingness is not None:raise ValueError('Intervention must not claim Q values')
        gate=dict(mode='counterfactual-intervention',q_continue=None,q_switch=None,willingness=None)
    else:
        if values is None or values['state_id']!=fresh['state_id']:
            raise ValueError('Q values must describe the same state/action pair')
        q0=float(values['q_continue']);q1=float(values['q_switch'])
        if willingness is None or not all(math.isfinite(x) for x in (q0,q1,float(willingness))):
            raise ValueError('Finite Q values and willingness required')
        if not 0<=willingness<=1:raise ValueError('Willingness must be a probability')
        gate=dict(mode='value-gate',q_continue=q0,q_switch=q1,willingness=float(willingness))
        accepted=q1>q0 and willingness>=env.cfg.response_threshold
        env.log('switch_gate',uav=u,task=j,state_id=fresh['state_id'],accepted=accepted,**gate)
        if not accepted:return False
    old=int(env.active[u]);lost=int(env.sensing[u]);offer=fresh['actions'][1]
    # Record before assignment/select events so an independent state machine can
    # verify the interruption rather than infer it from final task counters.
    env.log('task_switch',uav=u,task=j,interrupted_task=old,lost_sensing=lost,
            state_id=fresh['state_id'],route=offer['route'],estimate=offer['estimate'],**gate)
    env.active[u]=-1;env.sensing[u]=0
    delta=offer['estimate']['length']-fresh['actions'][0]['estimate']['length']
    env.assign(j,u,(delta,0,offer['estimate']))
    env.select(u,j)
    env.last_switch_step[u]=env.time
    return True
