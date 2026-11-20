import numpy as np


def ratio(n, d):
    return float(n / d) if d else None


def metrics(env):
    c, s = env.cfg, env.scenario
    released = s.release < c.horizon
    assigned = env.ever_assigned & released
    completed = (env.completed_at >= 0) & (env.completed_at <= s.deadline) & released
    nt, na, nc = int(released.sum()), int(assigned.sum()), int(completed.sum())
    assert nc <= na <= nt
    tcr, pst = ratio(nc, nt), ratio(nc, na)
    used = env.distance > 0
    returned = np.linalg.norm(env.position - .5, axis=1) <= 1e-7
    feasible = (env.distance <= c.max_distance + 1e-7) & (env.energy >= c.reserve - 1e-7) & returned
    pfr = float(feasible.mean())
    delay = float(np.mean(env.delays)) if env.delays else 0.
    energy_penalty = float(np.clip((c.reserve - env.energy) / max(c.reserve, 1), 0, 1).mean())
    delay_penalty = min(delay / c.delay_scale, 1.)
    qos = .45 * tcr + .45 * pfr - .05 * delay_penalty - .05 * energy_penalty
    assert -.1 - 1e-8 <= qos <= .45 * (tcr + pfr) + 1e-8
    arrivals = released & (s.release > 0)
    accepted = arrivals & assigned
    latencies = [env.assignment_seconds[i] for i in np.flatnonzero(accepted)]
    result = dict(total=nt, assigned=na, completed=nc, TCR=tcr, PST=pst, PFR=pfr,
                  active_PFR=ratio(int((feasible & used).sum()), int(used.sum())),
                  active_uavs=int(used.sum()), returned_uavs=int(returned.sum()),
                  CRR=ratio(env.conflicts_resolved, env.conflicts_detected),
                  conflicts_detected=env.conflicts_detected, conflicts_resolved=env.conflicts_resolved,
                  response_decisions=env.response_decisions, fallback_decisions=env.fallback_decisions,
                  fallback_rate=ratio(env.fallback_decisions,env.response_decisions),
                  mean_path_km=float(env.distance.mean() * 10),
                  DE_m=float(np.mean(env.deviations) * 10000) if env.deviations else None,
                  interactions=len(env.deviations), QoS=qos, delay_penalty=delay_penalty,
                  energy_penalty=energy_penalty,
                  arrival_tasks=int(arrivals.sum()), accepted_tasks=int(accepted.sum()),
                  Acceptance=ratio(int(accepted.sum()), int(arrivals.sum())),
                  ReallocationSuccess=ratio(int((accepted & completed).sum()), int(accepted.sum())),
                  decision_compute_s=float(env.compute_seconds),
                  mean_assignment_compute_s=float(np.mean(latencies)) if latencies else None,
                  mean_assignment_wait_steps=float(np.mean(env.assigned_at[accepted] - s.release[accepted])) if accepted.any() else None,
                  energy_used=float((c.energy - env.energy).sum()),
                  unfinished=int((released & ~completed).sum()))
    return result

