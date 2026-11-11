import time
import numpy as np
import torch
from .features import upper_features, lower_features, proposer_features
from .models import response_gate_probability, realize_responder_set
from .route_geometry import RouteGeometry

METHODS = ['dhgc', 'no-cdta', 'no-cmab', 'no-joint', 'hard-response',
           'lcb-cumulative', 'ucb-ew', 'ucb-cumulative', 'fixed-cost',
           'nearest', 'cheapest', 'regret', 'local-repair', 'kmeans', 'cmab-only']
LEARNED = METHODS[:9]


class Controller:
    def __init__(self, cfg, method='cheapest', model=None, training=False, proposer_training=None,
                 route_cache=True):
        if method not in METHODS:
            raise ValueError(method)
        if method not in ['nearest', 'cheapest', 'regret', 'local-repair', 'kmeans', 'cmab-only'] and model is None:
            raise ValueError(f'{method} requires a trained checkpoint')
        self.cfg, self.method, self.model, self.training = cfg, method, model, training
        self.records = []
        self.route_cache = route_cache
        self.proposer_training=training if proposer_training is None else proposer_training
        if model is not None and cfg.task_order=='learned' and not hasattr(model,'proposer'):
            raise ValueError('Learned task order requires a proposer checkpoint')

    def allocation_order(self,env,pending):
        if self.cfg.task_order!='learned' or self.model is None:
            yield from pending;return
        remaining=list(pending)
        while remaining:
            started=time.perf_counter();obs=proposer_features(env,remaining)
            if not obs['mask'].any():
                env.compute_seconds+=time.perf_counter()-started
                break
            with torch.no_grad():
                logits,value=self.model.proposer(obs)
                dist=torch.distributions.Categorical(logits=logits)
                action=int(dist.sample()) if self.proposer_training else int(logits.argmax())
                task=remaining[action];logp=float(dist.log_prob(torch.tensor(action,device=logits.device)))
                probabilities=dist.probs.cpu().tolist()
            if logits.is_cuda:torch.cuda.synchronize(logits.device)
            env.compute_seconds+=time.perf_counter()-started
            if self.proposer_training:
                self.records.append(dict(layer='proposer',obs=obs,action=action,time=env.time,
                    task=task,uav=-1,old_logp=logp,old_value=float(value)))
            env.log('task_proposal',candidates=list(remaining),mask=obs['mask'].tolist(),
                    probabilities=probabilities,action=action,task=int(task),logp=logp,
                    training=bool(self.proposer_training))
            remaining.remove(task)
            yield task # caller performs the assignment before the next observation

    @property
    def bandit_variant(self):
        return {'no-cmab': 'fixed', 'fixed-cost': 'fixed', 'lcb-cumulative': 'lcb-cumulative',
                'ucb-ew': 'ucb-ew', 'ucb-cumulative': 'ucb-cumulative'}.get(self.method, 'lcb-ew')

    def allocate(self, env, excluded_tasks=()):
        # Offline action-pair probes can defer a task for this scheduling round.
        # Normal deployment calls preserve the original complete pending list.
        pending = [j for j in env.pending() if j not in excluded_tasks]
        geometry = RouteGeometry(env) if self.route_cache and pending else None
        # Regret insertion prioritizes tasks with few good alternative UAVs.
        if self.method == 'regret':
            def regret(j):
                costs = sorted(ins[0] for u in range(self.cfg.uavs)
                               if (ins := env.insertion(u, j, geometry=geometry)) is not None)
                return costs[1]-costs[0] if len(costs)>1 else (1e6 if costs else -1e6)
            pending.sort(key=regret, reverse=True)
        centers = None
        if self.method == 'kmeans' and pending:
            from scipy.cluster.vq import kmeans2
            points = env.observed_positions()[pending]
            # Deterministic initialization avoids consuming scenario randomness.
            if len(points) >= self.cfg.uavs:
                initial = points[np.linspace(0, len(points)-1, self.cfg.uavs, dtype=int)]
                centers, _ = kmeans2(points, initial, iter=10, minit='matrix', missing='warn')
        for j in self.allocation_order(env,pending):
            started = time.perf_counter()
            insertions = [env.insertion(u, j, geometry=geometry) for u in range(self.cfg.uavs)]
            # Response loss affects all methods in the shared centralized interface.
            available = env.scenario.response_uniform[env.time, j] >= self.cfg.loss
            insertions = [v if available[u] else None for u, v in enumerate(insertions)]
            valid = [u for u, v in enumerate(insertions) if v is not None]
            if not valid:
                env.compute_seconds += time.perf_counter()-started
                continue
            obs, action_record = None, None
            if self.model is not None:
                obs = upper_features(env, j, insertions)
                with torch.no_grad():
                    logits, q, response, proposer = self.model.upper(obs, willingness=self.method not in ('no-cdta','hard-response'),
                                                                   hard_response=self.method=='hard-response')
                    responders = valid if self.method == 'hard-response' else [x for x in valid if float(response[x]) >= self.cfg.response_threshold]
                    fallback = not responders and self.method != 'no-cdta' and not self.training
                    realization = 'legacy_executor_sample' if self.training else 'hard_threshold'
                    gate_probability = None
                    if self.cfg.response_gate_mode == 'threshold-logistic' and self.method not in ('hard-response','no-cdta'):
                        mask = torch.as_tensor(obs['mask'], device=logits.device, dtype=torch.bool)
                        distances = torch.as_tensor(obs['uavs'][:,6], device=logits.device)
                        costs = torch.as_tensor(obs['uavs'][:,7], device=logits.device).masked_fill(~mask, torch.inf)
                        utility = q-self.cfg.distance_weight*distances
                        # Use the same raw scorer as forward(), without a second network call.
                        gate_probability = response_gate_probability(response,
                            self.cfg.response_threshold,self.cfg.response_gate_temperature,self.cfg.response_gate_mode)
                        # responder_distribution uses this numerical clamp as well.
                        gate_probability = gate_probability.clamp(1e-6,1-1e-6)
                        gates = (torch.rand_like(response)<gate_probability if self.training
                                 else response>=self.cfg.response_threshold)
                        u, chosen, fallback = realize_responder_set(gates,utility,mask,costs.argmin())
                        responders = chosen.tolist()
                        realization = 'sampled_threshold_logistic_gates' if self.training else 'hard_threshold'
                    if self.training:
                        dist = torch.distributions.Categorical(logits=logits)
                        if realization != 'sampled_threshold_logistic_gates':u = int(dist.sample())
                        action_record = dict(layer='upper', obs=obs, action=u, time=env.time,
                                             task=j, uav=u, old_logp=float(dist.log_prob(torch.tensor(u,device=logits.device))))
                    elif self.method == 'no-cdta':
                        u = max(valid, key=lambda x: float(proposer[x]))
                    elif gate_probability is None:
                        if responders:
                            pos = env.observed_positions()[j]
                            u = max(responders, key=lambda x: float(q[x])-self.cfg.distance_weight*np.linalg.norm(pos-env.position[x]))
                        else:
                            u = min(valid, key=lambda x: insertions[x][0])
                    env.log('response',task=int(j),feasible=valid,
                            probability=response.detach().cpu().tolist(),
                            q=q.detach().cpu().tolist(),executor=int(u),training=bool(self.training),
                            responders=responders, fallback=bool(fallback),
                            realization=realization,
                            gate_probability=None if gate_probability is None else gate_probability.cpu().tolist(),
                            empty_set_probability=None if gate_probability is None else float((1-gate_probability[mask]).prod()),
                            policy_probability=logits.softmax(-1).cpu().tolist())
                    env.response_decisions += 1
                    env.fallback_decisions += int(fallback)
                if logits.is_cuda:
                    torch.cuda.synchronize(logits.device)
            elif self.method == 'nearest':
                u = min(valid, key=lambda x: np.linalg.norm(env.position[x]-env.observed_positions()[j]))
            elif centers is not None:
                p = env.observed_positions()[j]
                u = min(valid, key=lambda x: np.linalg.norm(centers[x]-p))
            else:
                u = min(valid, key=lambda x: insertions[x][0])
            elapsed = time.perf_counter()-started
            env.compute_seconds += elapsed
            ins = insertions[u]
            env.assign(j, u, ins, elapsed)
            if action_record:
                action_record['local_reward'] = -self.cfg.reward_energy*max(ins[0],0)*self.cfg.flight_energy/self.cfg.energy
                self.records.append(action_record)

    def plan(self, env):
        for u in range(self.cfg.uavs):
            started = time.perf_counter()
            candidates = env.route_candidates(u)
            if not candidates:
                env.compute_seconds += time.perf_counter()-started
                continue
            if self.model is not None:
                obs = lower_features(env, u, candidates)
                if self.method in ('no-cmab','fixed-cost'):
                    obs['tasks'][:, 4:7] = 0.
                with torch.no_grad():
                    logits, _ = self.model.lower(obs, cost_prior=self.method not in ('no-cmab','fixed-cost'))
                    dist = torch.distributions.Categorical(logits=logits)
                    a = int(dist.sample()) if self.training else int(logits.argmax())
                    if self.training:
                        self.records.append(dict(layer='lower', obs=obs, action=a, time=env.time,
                                                 task=candidates[a], uav=u, old_logp=float(dist.log_prob(torch.tensor(a,device=logits.device)))))
                if logits.is_cuda:
                    torch.cuda.synchronize(logits.device)
                j = candidates[a]
            elif self.method == 'cmab-only':
                p = env.observed_positions()
                idx = env.bandit.index(u, env.time)
                j = min(candidates, key=lambda k: np.linalg.norm(p[k]-env.position[u])+idx[env.bandit.region(p[k])])
            elif self.method == 'local-repair':
                p = env.observed_positions()
                j = min(candidates, key=lambda k: np.linalg.norm(p[k]-env.position[u]))
            else:
                j = candidates[0]
            env.select(u, j)
            env.compute_seconds += time.perf_counter()-started
