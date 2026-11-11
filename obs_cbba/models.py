import math
import torch
from torch import nn
from .features import TASK_DIM, UAV_DIM, REGION_DIM, HISTORY_DIM


def tensor(value, device):
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def responder_distribution(response, utility, mask, fallback):
    """Exact executor probabilities for independent Bernoulli responders.

    A responder wins iff it accepts and every higher-utility responder declines.
    The empty-set mass belongs to the feasible insertion fallback. Ranking is a
    discrete action realization; gradients flow through probabilities, not Q.
    """
    valid = torch.where(mask)[0]
    order = valid[torch.argsort(utility[valid].detach(), descending=True, stable=True)]
    p = response[order].clamp(1e-6, 1-1e-6)
    if torch.are_deterministic_algorithms_enabled():
        # CUDA cumprod backward uses a nondeterministic cumsum kernel in torch
        # 2.5.1. At 4/6 UAVs an ordered scalar product is small and differentiable.
        prefixes = [p.new_ones(())]
        for decline in (1-p).unbind():
            prefixes.append(prefixes[-1] * decline)
        preceding = torch.stack(prefixes[:-1])
        empty = prefixes[-1]
    else:
        preceding = torch.cat((p.new_ones(1), (1-p).cumprod(0)[:-1]))
        empty = (1-p).prod()
    probabilities = torch.zeros_like(response).scatter(0, order, p*preceding)
    fallback_mask = torch.arange(len(response), device=response.device) == fallback
    return probabilities + fallback_mask * empty


def response_gate_probability(response, threshold, temperature, mode):
    """Exploration probability from the raw sigmoid willingness score.

    Threshold-logistic adds logistic noise to the willingness logit BEFORE the
    same threshold decision used at deployment. Away from the boundary its
    zero-temperature limit is the deployed hard gate. This is a declared
    reconstruction option, not a claimed recovery of the missing implementation.
    """
    if mode == 'bernoulli':
        return response
    if mode != 'threshold-logistic' or not 0 < temperature < math.inf:
        raise ValueError('Invalid response exploration protocol')
    if threshold == 0:
        return torch.ones_like(response)
    if threshold == 1:
        return torch.zeros_like(response)
    if not 0 < threshold < 1:
        raise ValueError('Invalid response threshold')
    boundary = math.log(threshold / (1-threshold))
    response_logits = torch.logit(response.clamp(1e-7,1-1e-7))
    return ((response_logits-boundary)/temperature).sigmoid()


def realize_responder_set(gates, utility, mask, fallback):
    """Shared deterministic action realization for sampled and deployed gates."""
    responders = torch.where(gates & mask)[0]
    if not bool(mask[fallback]):
        raise ValueError('Fallback must be feasible')
    if len(responders):
        # torch.where preserves ascending UAV order, matching stable ranking ties.
        return int(responders[utility[responders].argmax()]), responders, False
    return int(fallback), responders, True


class CrossBlock(nn.Module):
    """UAV queries against task/history/region keys: no N x N task attention."""
    def __init__(self, d, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d, heads, dropout=0., batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))

    def forward(self, query, context):
        a, _ = self.attention(query, context, context, need_weights=False)
        query = self.norm1(query + a)
        return self.norm2(query + self.ff(query))


class UpperPolicy(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c.dim
        self.task = nn.Linear(TASK_DIM, d)
        self.uav = nn.Linear(UAV_DIM, d)
        self.region = nn.Linear(REGION_DIM, d)
        self.history = nn.Linear(HISTORY_DIM, d)
        self.blocks = nn.ModuleList([CrossBlock(d, c.heads) for _ in range(c.layers)])
        self.actor = nn.Linear(d, 1)
        self.response = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, 1))
        self.q = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.distance_weight = c.distance_weight
        self.response_threshold = c.response_threshold
        self.response_gate_mode = c.response_gate_mode
        self.response_gate_temperature = c.response_gate_temperature

    def forward(self, obs, willingness=True, hard_response=False):
        device = next(self.parameters()).device
        task = self.task(tensor(obs['tasks'], device))[None]
        query = self.uav(tensor(obs['uavs'], device))[None]
        context = torch.cat((task, self.region(tensor(obs['regions'], device))[None],
                             self.history(tensor(obs['history'], device))[None]), dim=1)
        for block in self.blocks:
            query = block(query, context)
        h = query[0]
        current = self.task(tensor(obs['current'], device)).expand_as(h)
        response_logits = self.response(torch.cat((h, current), dim=-1)).squeeze(-1)
        proposer_logits, q = self.actor(h).squeeze(-1), self.q(h).squeeze(-1)
        mask = torch.as_tensor(obs['mask'], device=device, dtype=torch.bool)
        utility = q - self.distance_weight*tensor(obs['uavs'], device)[:, 6]
        if willingness:
            costs = tensor(obs['uavs'], device)[:, 7].masked_fill(~mask, torch.inf)
            gate_probability = response_gate_probability(response_logits.sigmoid(), self.response_threshold,
                self.response_gate_temperature, self.response_gate_mode)
            probabilities = responder_distribution(gate_probability, utility, mask, costs.argmin())
            logits = probabilities.clamp_min(1e-30).log()
        elif hard_response:
            # Q exploration in training, argmax of the same utility at deployment.
            logits = utility
        else:
            logits = proposer_logits
        logits = logits.masked_fill(~mask, -1e9)
        return logits, q, response_logits.sigmoid(), proposer_logits


class LowerPolicy(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c.dim
        self.task = nn.Linear(TASK_DIM, d)
        self.uav = nn.Linear(UAV_DIM, d)
        layer = nn.TransformerEncoderLayer(d, c.heads, 4*d, dropout=0.,
                                           activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, c.layers, enable_nested_tensor=False)
        self.score = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, 1))
        self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.cost_strength = nn.Parameter(torch.tensor(1.))

    def forward(self, obs, cost_prior=True):
        device = next(self.parameters()).device
        features = tensor(obs['tasks'], device)
        tokens = self.task(features)
        u = self.uav(tensor(obs['uav'], device))
        h = self.encoder(torch.cat((u[None], tokens), dim=0)[None])[0]
        logits = self.score(torch.cat((h[1:], h[0].expand_as(h[1:])), dim=-1)).squeeze(-1)
        if cost_prior:
            logits = logits - torch.nn.functional.softplus(self.cost_strength) * features[:, 4]
        return logits, self.value(h[0]).squeeze(-1)


class TaskProposer(nn.Module):
    """Independent actor/value network for the next broadcast TASK.

    Task queries cross-attend to UAV/history/region tokens and a task mean.
    This avoids introducing quadratic task self-attention into the proposer.
    """
    def __init__(self,c):
        super().__init__();d=c.dim
        self.task=nn.Linear(TASK_DIM,d);self.uav=nn.Linear(UAV_DIM,d)
        self.region=nn.Linear(REGION_DIM,d);self.history=nn.Linear(HISTORY_DIM,d)
        self.blocks=nn.ModuleList([CrossBlock(d,c.heads) for _ in range(c.layers)])
        self.actor=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,1))
        self.value=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,1))

    def forward(self,obs):
        device=next(self.parameters()).device
        mask=torch.as_tensor(obs['mask'],device=device,dtype=torch.bool)
        if not bool(mask.any()):raise ValueError('No eligible proposer action')
        query=self.task(tensor(obs['tasks'],device))[None]
        context=torch.cat((self.uav(tensor(obs['uavs'],device))[None],
            self.region(tensor(obs['regions'],device))[None],
            self.history(tensor(obs['history'],device))[None],query.mean(dim=1,keepdim=True)),dim=1)
        for block in self.blocks:query=block(query,context)
        h=query[0];logits=self.actor(h).squeeze(-1).masked_fill(~mask,-torch.inf)
        return logits,self.value(h[mask].mean(dim=0)).squeeze(-1)


class DHGC(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.upper = UpperPolicy(c)
        self.lower = LowerPolicy(c)
        if c.task_order=='learned':self.proposer=TaskProposer(c)
