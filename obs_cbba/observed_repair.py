"""Deterministic repair rules; no sampled or privileged futures."""
import time
from .commitment import (commitment_observe, repair_candidates, release_unstarted,
                         event_now, run_commitment)
from .policies import Controller


def choose_repair(observation, candidates, method):
    if method not in ('e1', 'e2'):
        raise ValueError(method)
    d = observation.as_dict()
    lengths = [len(q) for q in d['queues']]
    durations = [end-start for start, end in zip(d['assigned_at'], d['completed_at'])
                 if 0 <= start <= end <= d['time']]
    history_mean = sum(durations)/len(durations) if durations else None
    eligible = []
    for move in candidates:
        j, src, dst = move['task'], move['source'], move['destination']
        difference = lengths[src]-lengths[dst]
        if method == 'e1':
            if (history_mean is not None and difference > 0
                    and 2*len(durations)*(d['deadline'][j]-d['time']) < 3*sum(durations)):
                eligible.append(move)
        elif difference >= 2:
            eligible.append(move)
    def key(m):
        j, src, dst = m['task'], m['source'], m['destination']
        if method == 'e1':
            return (d['deadline'][j], j, lengths[dst], dst)
        return (-lengths[src], src, d['deadline'][j], j, lengths[dst], dst)
    chosen = min(eligible, key=key) if eligible else None
    return dict(selected=chosen, qualified=len(eligible), queue_lengths=lengths,
                history_count=len(durations), history_mean=history_mean,
                cold_start=method == 'e1' and not durations)


def observed_repair(env, method):
    started = time.perf_counter()
    observation = commitment_observe(env)
    candidates = repair_candidates(env)
    decision = choose_repair(observation, candidates, method)
    env.log('observed_repair', method=method, selected=decision['selected'])
    move = decision['selected']
    if move is not None:
        j, u = move['task'], move['destination']
        # Validate before withdrawing; a failed check cannot leave a half action.
        ins = env.insertion(u, j)
        assert ins is not None and env.scenario.response_uniform[env.time, j, u] >= env.cfg.loss
        env.withdraw(j, 'observed-reassignment-'+method)
        env.assign(j, u, ins)
    return dict(time=env.time, method=method, observation=observation.as_dict(),
                candidates=candidates, eligible=len(candidates), **decision,
                seconds=time.perf_counter()-started)


def run_observed(env, method, check=lambda: None):
    if method in ('b', 'd'):
        return run_commitment(env, method, check)
    if method not in ('e1', 'e2'):
        raise ValueError(method)
    controller = Controller(env.cfg, 'cheapest')
    previous_completed = []; traces = []; events = 0
    while not env.done:
        check()
        if env.time < env.cfg.horizon:
            event = event_now(env, previous_completed)
            events += int(event)
            if event:
                release_unstarted(env)
            controller.allocate(env)
            if event:
                traces.append(observed_repair(env, method))
            controller.plan(env)
        _, previous_completed = env.tick()
    return dict(events=events, decisions=len(traces), traces=traces)
