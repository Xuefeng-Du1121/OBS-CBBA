"""Learned event trigger in front of the unmodified DA-CBBA selector."""
import math
import time
import hashlib
import torch
from torch import nn
from .amortized_order import BranchNetwork, observation_features, feature_batch
from .maritime_planning import observe_maritime
from . import observable_cbba as teacher

SCHEMA='wp3e-cbba-trigger-v1'
FROZEN_CHECKPOINT_SHA256='5a0d0bf090e0d3d0d94b92d1e41afb355959ea5f369f0fd777da2c42d1e5253f'

class TriggerNetwork(BranchNetwork):
    def __init__(self,hidden=64):
        super().__init__(hidden)
        self.logits=nn.Linear(hidden,1)
        del self.gaps

    def forward(self,batch):
        tasks=self.pool(self.task_encoder(batch['tasks']),batch['tasks_mask'])
        uavs=self.pool(self.uav_encoder(batch['uavs']),batch['uavs_mask'])
        return self.logits(self.trunk(torch.cat((tasks,uavs,batch['global_features']),dim=-1))).squeeze(-1)

class TriggerSelector:
    def __init__(self,model,threshold,*,teacher_module=teacher, k=8, variant='full',
                 suffix='cbba-pr', deferred_commitment=True, workers=1):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1: raise ValueError('Invalid threshold')
        if type(k) is not int or k not in (1,4,8): raise ValueError('Invalid trigger K')
        if type(workers) is not int or workers not in (1, 2, 4, 8): raise ValueError('Invalid trigger workers')
        self.model=model.cpu().eval();self.threshold=threshold;self.teacher_module=teacher_module
        self.k=k;self.variant=variant;self.suffix=suffix;self.deferred_commitment=bool(deferred_commitment);self.workers=workers

    def __call__(self,observation,bundles,check=lambda:None):
        check();features=observation_features(observation)
        with torch.inference_mode():
            logit=float(self.model(feature_batch([features]))[0])
        if not math.isfinite(logit): raise ValueError('Nonfinite trigger prediction')
        p=1/(1+math.exp(-max(-700.,min(700.,logit))))
        triggered=p>=self.threshold
        if triggered:
            kwargs = {} if self.teacher_module is teacher else {
                'suffix': self.suffix, 'deferred_commitment': self.deferred_commitment,
                'workers': self.workers}
            full=self.teacher_module.select_action(observation,bundles,self.k,check,
                                                   variant=self.variant,**kwargs)
        else:
            full=None
        return dict(time=observation.as_dict()['base']['time'],observation_sha256=observation.fingerprint(),
            score=p,logit=logit,threshold=self.threshold,triggered=triggered,
            selected=full['selected'] if triggered else 'cbba-pr',teacher=full)


class NeverTriggerSelector:
    """Explicit no-network endpoint used for the never-trigger identity proof."""
    def __init__(self, *, teacher_module=teacher, k=8, variant='base', workers=1):
        if type(k) is not int or k not in (1, 4, 8):
            raise ValueError('Invalid trigger K')
        if type(workers) is not int or workers not in (1, 2, 4, 8):
            raise ValueError('Invalid trigger workers')
        self.teacher_module=teacher_module; self.k=k; self.variant=variant
        self.suffix='cbba-pr'; self.deferred_commitment=True; self.workers=workers

    def __call__(self, observation, bundles, check=lambda:None):
        check()
        return dict(time=observation.as_dict()['base']['time'],
            observation_sha256=observation.fingerprint(), score=0., logit=float('-inf'),
            threshold=float('inf'), triggered=False, selected='cbba-pr', teacher=None)

def load_selector(path, *, expected_sha256=None, **kwargs):
    if expected_sha256 is not None:
        actual=hashlib.sha256(open(path,'rb').read()).hexdigest()
        if actual != expected_sha256: raise ValueError('Trigger checkpoint SHA256 mismatch')
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint['schema']!=SCHEMA or checkpoint['target']!='gain>0': raise ValueError('Trigger contract mismatch')
    model=TriggerNetwork(checkpoint['hidden']);model.load_state_dict(checkpoint['state_dict'])
    return TriggerSelector(model,checkpoint['threshold'],**kwargs),checkpoint

def run(env,selector,check=lambda:None):
    observe_maritime(env);bundles=teacher.initial_bundles(env);seen=set();previous=[];traces=[]
    seconds=dict(selection=0.,allocation=0.,dispatch=0.)
    while not env.done:
        check()
        if env.time<env.cfg.horizon:
            obs=observe_maritime(env);known=set(obs.as_dict()['base']['known'])
            event=bool(known-seen) or bool(previous);seen.update(known)
            if event:
                bundles=[[j for j in q if j in env.queues[u]] for u,q in enumerate(bundles)]
                t=time.perf_counter();row=selector(obs,bundles,check)
                row['seconds']=time.perf_counter()-t;seconds['selection']+=row['seconds'];env.compute_seconds+=row['seconds']
                t=time.perf_counter();bundles=teacher.apply_action(
                    env,bundles,row['selected'],check=check,
                    deferred_commitment=selector.deferred_commitment)
                seconds['allocation']+=time.perf_counter()-t
                if row['triggered']:
                    full=row['teacher'];chosen=full['actions'].index(row['selected'])
                    assert full['signatures'][chosen]==dict(queues=[list(q) for q in env.queues],active=env.active.tolist(),bundles=bundles)
                traces.append(row)
            t=time.perf_counter();teacher.dispatch(env);seconds['dispatch']+=time.perf_counter()-t
        _,previous=env.tick()
    count=sum(r['triggered'] for r in traces)
    return dict(events=len(traces),decisions=len(traces),traces=traces),dict(seconds=seconds,
        decision_seconds=seconds['selection'],triggered=count,trigger_rate=count/len(traces) if traces else 0.,sampled_suffixes=40*count)
