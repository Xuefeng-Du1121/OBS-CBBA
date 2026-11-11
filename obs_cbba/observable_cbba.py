"""DA-CBBA: sampled-future evaluation of commitment revisions with a CBBA-PR continuation.

Decision state is BEFORE withdrawal. Actions include reset, allocation and first
route dispatch. Full = CBBA-PR + all four original o_obs event actions; compact =
CBBA-PR + cheapest/regret. Non-event ticks and every evaluation suffix use CBBA-PR.
No existing module is modified. The immutable input plus acquisition bundles is
the complete planner interface; a true environment is accepted only by execution.
"""
import hashlib
import time
from .baselines_p5 import Parameters, cbba_allocate, install_routes, reset_tail
from .commitment import commitment_planning_env, release_unstarted
from .decision_headroom import ORDERS, allocate
from .maritime_planning import observe_maritime, unpack_maritime, sample_maritime_future
from .planning import planning_seeds
from .policies import Controller

BRANCHES = {'full': ('cbba-pr', *ORDERS), 'compact': ('cbba-pr', 'cheapest', 'regret')}
NAMESPACE = 'wp3d-observable-cbba-v1:'


def initial_bundles(env):
    """Same documented resume convention as P5; never used to reset live history."""
    return [sorted(q, key=lambda j: (env.assigned_at[j], j)) for q in env.queues]


def validate_bundles(observation, bundles):
    data, _ = unpack_maritime(observation); b = data['base']
    if len(bundles) != b['config']['uavs']:
        raise ValueError('Wrong bundle count')
    for bundle, queue in zip(bundles, b['queues']):
        if len(bundle) != len(set(bundle)) or set(bundle) != set(queue):
            raise ValueError('Bundles must contain exactly the current queued tasks')
    return data


def dispatch(env):
    """Route-order execution; no between-event auctions or reordering."""
    for u, q in enumerate(env.queues):
        if env.active[u] < 0 and q and q[0] in env.route_candidates(u):
            env.select(u, q[0])


def apply_action(env, bundles, action, parameters=Parameters(), check=lambda: None,
                 *, arrival_rate_multiplier=1., deferred_commitment=True):
    """Execute the complete event action; return its acquisition history.

    Alternative branches withdraw all unstarted work as o_obs does. Their new
    acquisition sequence records actual assign calls (NOT route/first-ever order).
    """
    p = parameters.validate()
    if action not in BRANCHES['full']:
        raise ValueError(action)
    bundles = [[j for j in q if j in env.queues[u]] for u,q in enumerate(bundles)]
    if action == 'cbba-pr':
        for u in range(env.cfg.uavs):
            tail=(reset_tail(bundles[u],env.started_at,count=p.reset_count,fraction=p.reset_fraction)
                  if deferred_commitment else [])
            for j in tail: env.withdraw(j,'cbba-pr-reset')
            bundles[u]=[j for j in bundles[u] if j not in tail]
        result=cbba_allocate(observe_maritime(env, arrival_rate_multiplier=arrival_rate_multiplier),bundles,p,check)
        install_routes(env,result['routes']); bundles=result['bundles']; dispatch(env)
    else:
        if deferred_commitment:
            release_unstarted(env)
        bundles=[[j for j in q if j in env.queues[u]] for u,q in enumerate(bundles)]
        original=env.assign; had_override='assign' in env.__dict__
        def record(j,u,insertion,compute_seconds=0.):
            original(j,u,insertion,compute_seconds); bundles[u].append(int(j))
        env.assign=record
        try: allocate(env,action,check=check)
        finally:
            if had_override: env.assign=original
            else: del env.assign
        Controller(env.cfg,'cheapest').plan(env)
    return bundles


def cbba_suffix(env, bundles, check=lambda: None, *, prepared=True, parameters=Parameters(), on_event=None,
                arrival_rate_multiplier=1., deferred_commitment=True):
    """Exact continuation with caller-supplied bundles; no reconstructed history.

    prepared=True means this event's complete action was already executed. Do
    not reset/auction again before the first physical tick. Subsequent events
    are newly observed task IDs or previous-tick completions, exactly as P5.
    """
    bundles=[list(q) for q in bundles]; seen=set(); previous=[]
    if prepared and env.time < env.cfg.horizon:
        seen.update(observe_maritime(env, arrival_rate_multiplier=arrival_rate_multiplier).as_dict()['base']['known'])
    while not env.done:
        check()
        if env.time < env.cfg.horizon:
            obs=observe_maritime(env, arrival_rate_multiplier=arrival_rate_multiplier)
            known=set(obs.as_dict()['base']['known'])
            event=bool(known-seen) or bool(previous); seen.update(known)
            if event and not prepared:
                bundles=apply_action(env,bundles,'cbba-pr',parameters,check,
                                     arrival_rate_multiplier=arrival_rate_multiplier,
                                     deferred_commitment=deferred_commitment)
                if on_event is not None: on_event(env,bundles)
            dispatch(env)
        _,previous=env.tick(); prepared=False
    return int((env.completed_at>=0).sum())


def select_action(observation, bundles, k=8, check=lambda: None, *, variant='full', parameters=Parameters(),
                  arrival_rate_multiplier=1.):
    if type(k) is not int or k != 8: raise ValueError('this entry point fixes K=8')
    if variant not in BRANCHES: raise ValueError(variant)
    data=validate_bundles(observation,bundles); parameters.validate()
    # Only public observation supplies seeds; all actions share each future.
    anchor=int.from_bytes(hashlib.sha256((NAMESPACE+observation.fingerprint()).encode()).digest()[:8],'little')
    seeds=planning_seeds(anchor,k); actions=BRANCHES[variant]
    values=[[] for _ in actions]; hashes=[]; signatures=[None for _ in actions]
    for seed in seeds:
        check(); scenario=sample_maritime_future(observation,seed)
        fingerprint=scenario.fingerprint(); hashes.append(fingerprint)
        for i,action in enumerate(actions):
            check(); env=commitment_planning_env(observation.base(),scenario,log_events=False)
            next_bundles=apply_action(env,bundles,action,parameters,check,
                                       arrival_rate_multiplier=arrival_rate_multiplier)
            signature=dict(queues=[list(q) for q in env.queues],active=env.active.tolist(),bundles=next_bundles)
            if signatures[i] is None: signatures[i]=signature
            elif signatures[i] != signature: raise RuntimeError('Current action depends on a sampled hidden future')
            values[i].append(cbba_suffix(env,next_bundles,check,prepared=True,parameters=parameters,
                                         arrival_rate_multiplier=arrival_rate_multiplier))
            if scenario.fingerprint()!=fingerprint: raise RuntimeError('Shared future mutated')
    sums=[sum(v) for v in values]; winner=max(range(len(actions)),key=lambda i:sums[i])
    return dict(time=data['base']['time'],observation=data,observation_sha256=observation.fingerprint(),
        bundles=[list(q) for q in bundles],k=k,variant=variant,actions=list(actions),seeds=seeds,
        sample_hashes=hashes,completed=values,sums=sums,signatures=signatures,selected=actions[winner],
        suffix='cbba-pr',parameters=vars(parameters))


def run(env,k=8,check=lambda:None,*,variant='full',parameters=Parameters(),on_decision=None,
        arrival_rate_multiplier=1.):
    if variant not in BRANCHES or type(k) is not int or k != 8: raise ValueError('Invalid policy')
    observe_maritime(env, arrival_rate_multiplier=arrival_rate_multiplier); parameters.validate()
    bundles=initial_bundles(env); seen=set(); previous=[]; traces=[]
    seconds=dict(selection=0.,allocation=0.,dispatch=0.)
    while not env.done:
        check()
        if env.time < env.cfg.horizon:
            obs=observe_maritime(env, arrival_rate_multiplier=arrival_rate_multiplier)
            known=set(obs.as_dict()['base']['known'])
            event=bool(known-seen) or bool(previous); seen.update(known)
            if event:
                bundles=[[j for j in q if j in env.queues[u]] for u,q in enumerate(bundles)]
                t=time.perf_counter(); row=select_action(
                    obs,bundles,k,check,variant=variant,parameters=parameters,
                    arrival_rate_multiplier=arrival_rate_multiplier)
                row['seconds']=time.perf_counter()-t;seconds['selection']+=row['seconds']
                env.compute_seconds+=row['seconds']
                t=time.perf_counter();bundles=apply_action(
                    env,bundles,row['selected'],parameters,check,
                    arrival_rate_multiplier=arrival_rate_multiplier)
                seconds['allocation']+=time.perf_counter()-t
                chosen=row['actions'].index(row['selected'])
                assert row['signatures'][chosen]==dict(queues=[list(q) for q in env.queues],active=env.active.tolist(),bundles=bundles)
                traces.append(row)
                if on_decision is not None: on_decision(row)
            t=time.perf_counter();dispatch(env);seconds['dispatch']+=time.perf_counter()-t
        _,previous=env.tick()
    return dict(events=len(traces),decisions=len(traces),traces=traces),dict(seconds=seconds,
        decision_seconds=seconds['selection'],sampled_suffixes=len(BRANCHES[variant])*k*len(traces))
