import numpy as np

TASK_DIM, UAV_DIM, REGION_DIM, HISTORY_DIM = 10, 10, 4, 4


def proposer_features(env, candidates):
    """Observed pending-task features and exact current insertion eligibility.

    IDs are action metadata, never numeric network features. No future positions
    or task outcomes enter the observation. Recompute after every assignment.
    """
    c,s=env.cfg,env.scenario;p=env.observed_positions();tasks=[];mask=[]
    for j in candidates:
        feasible=[env.insertion(u,j) for u in range(c.uavs)]
        valid=[u for u,x in enumerate(feasible) if x is not None and s.response_uniform[env.time,j,u]>=c.loss]
        region=int(env.bandit.region(p[j]));mask.append(bool(valid))
        tasks.append([*p[j],(s.deadline[j]-env.time)/c.initial_window,
            (env.time-s.release[j])/c.horizon,len(valid)/c.uavs,
            min((feasible[u][0] for u in valid),default=0.),
            min((float(np.linalg.norm(p[j]-env.position[u])) for u in valid),default=0.),
            env.bandit.mean[region],env.bandit.count[region]/100,1.])
    uavs=[]
    for u in range(c.uavs):
        estimate=env.route_estimate(u,env.queues[u])
        uavs.append([*env.position[u],env.energy[u]/c.energy,
            len(env.queues[u])/max(c.tasks/c.uavs,1),env.distance[u]/c.max_distance,
            float(env.active[u]>=0),env.sensing[u]/c.sensing_steps,
            estimate['length']/c.max_distance,(estimate['finish']-env.time)/c.horizon,1.])
    history=np.zeros((c.history,HISTORY_DIM),dtype=np.float32)
    if env.history:history[-len(env.history):]=env.history
    regions=np.array([[((g%c.grid)+.5)/c.grid,((g//c.grid)+.5)/c.grid,
        env.bandit.mean[g],env.bandit.count[g]/100] for g in range(c.grid**2)],dtype=np.float32)
    return dict(tasks=np.array(tasks,dtype=np.float32),uavs=np.array(uavs,dtype=np.float32),
                history=history,regions=regions,mask=np.array(mask,dtype=bool),task_ids=list(candidates))


def upper_features(env, j, insertions):
    c, s = env.cfg, env.scenario
    p = env.observed_positions()
    regions = env.bandit.region(p)
    released = np.flatnonzero((s.release <= env.time) & (s.deadline > env.time) & (env.completed_at < 0))
    if not len(released):
        released = np.array([j])
    tasks = np.array([[*p[k], (s.deadline[k]-env.time)/c.initial_window,
                       float(env.owner[k] >= 0), float(k == j),
                       env.bandit.mean[regions[k]], env.bandit.count[regions[k]]/100,
                       (env.time-s.release[k])/c.horizon,
                       float(s.release[k] > 0), 1.] for k in released], dtype=np.float32)
    uavs = []
    mask = []
    for u, ins in enumerate(insertions):
        estimate = env.route_estimate(u, env.queues[u])
        uavs.append([*env.position[u], env.energy[u]/c.energy,
                     len(env.queues[u])/max(c.tasks/c.uavs, 1),
                     estimate['length']/c.max_distance,
                     (estimate['finish']-env.time)/c.horizon,
                     float(np.linalg.norm(p[j]-env.position[u])),
                     ins[0] if ins else 0., float(env.active[u] >= 0), 1.])
        mask.append(ins is not None)
    history = np.zeros((c.history, HISTORY_DIM), dtype=np.float32)
    if env.history:
        history[-len(env.history):] = env.history
    region = np.array([[((g % c.grid)+.5)/c.grid, ((g//c.grid)+.5)/c.grid,
                        env.bandit.mean[g], env.bandit.count[g]/100]
                       for g in range(c.grid**2)], dtype=np.float32)
    current = tasks[np.flatnonzero(released == j)[0]]
    return dict(tasks=tasks, uavs=np.asarray(uavs, dtype=np.float32),
                history=history, regions=region, current=current,
                mask=np.asarray(mask, dtype=bool))


def lower_features(env, u, candidates):
    c, s = env.cfg, env.scenario
    p = env.observed_positions()
    idx = env.bandit.index(u, env.time)
    candidate_regions = env.bandit.region(p[candidates])
    # Candidate budget changes preferences only. All mandatory assigned tasks
    # remain reachable, preventing selective omission from improving metrics.
    present = np.unique(candidate_regions)
    budget = min(c.region_budget, len(present))
    chosen = present[np.argsort(idx[present])[:budget]]
    values = []
    for j, g in zip(candidates, candidate_regions):
        values.append([*p[j], (s.deadline[j]-env.time)/c.initial_window,
                       float(np.linalg.norm(p[j]-env.position[u])),
                       idx[g], env.bandit.mean[g],
                       float(g in chosen), float(j == env.queues[u][0]),
                       env.energy[u]/c.energy, 1.])
    return dict(tasks=np.array(values, dtype=np.float32),
                uav=np.array([*env.position[u], env.energy[u]/c.energy,
                              len(env.queues[u])/max(c.tasks/c.uavs, 1),
                              env.distance[u]/c.max_distance, env.time/c.horizon,
                              0., 0., 0., 1.], dtype=np.float32))
