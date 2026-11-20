"""Cell-stratified paired bootstrap and Holm correction (the frozen statistical rule)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np


DEFAULT_BOOTSTRAPS = 50_000
PRIMARY_SEED = 690524
ROBUSTNESS_SEED = 690525


def _validate_cells(cells):
    if not cells: raise ValueError('at least one cell is required')
    out={}
    for key, value in cells.items():
        a=np.asarray(value, dtype=float)
        if a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all(): raise ValueError(f'invalid cell {key}')
        out[str(key)] = a
    return out


def paired_stratified_bootstrap(cells, *, replicates=DEFAULT_BOOTSTRAPS, seed=PRIMARY_SEED,
                                save_indices=None):
    """Resample entire paired scene differences within cells, equal cell weights."""
    cells=_validate_cells(cells)
    if type(replicates) is not int or replicates < 1: raise ValueError('replicates')
    if type(seed) is not int or seed < 0: raise ValueError('seed')
    names=list(cells); rng=np.random.Generator(np.random.PCG64(seed))
    indices=np.empty((replicates,len(names),max(map(len,cells.values()))), dtype=np.int32)
    draws=np.empty((replicates,len(names)), dtype=float)
    for ci,name in enumerate(names):
        values=cells[name]; n=len(values)
        sampled=rng.integers(0,n,size=(replicates,n),endpoint=False,dtype=np.int32)
        indices[:,ci,:n]=sampled
        if n < indices.shape[2]: indices[:,ci,n:]=-1
        draws[:,ci]=values[sampled].mean(axis=1)
    replicate=draws.mean(axis=1)
    estimate=float(np.mean([a.mean() for a in cells.values()]))
    result=dict(estimate=estimate, lower=float(np.quantile(replicate,.025,method='linear')),
                upper=float(np.quantile(replicate,.975,method='linear')), replicates=replicate,
                indices=indices, seed=seed, cell_order=names,
                cell_sizes={name:len(cells[name]) for name in names})
    if save_indices is not None:
        save_indices=Path(save_indices); save_indices.parent.mkdir(parents=True,exist_ok=True)
        np.save(save_indices, indices)
        result['indices_sha256']=hashlib.sha256(save_indices.read_bytes()).hexdigest()
    return result


def centered_two_sided_pvalue(cells, *, replicates=DEFAULT_BOOTSTRAPS, seed=PRIMARY_SEED):
    r=paired_stratified_bootstrap(cells,replicates=replicates,seed=seed)
    centered=r['replicates']-r['estimate']
    p=float((1+np.count_nonzero(np.abs(centered)>=abs(r['estimate'])))/(replicates+1))
    return p


def holm(pvalues, alpha=.05):
    """Holm step-down decisions, retaining original test order."""
    values=np.asarray(pvalues,dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values<0)|(values>1)): raise ValueError('pvalues')
    order=np.argsort(values, kind='stable'); adjusted=np.empty(len(values)); running=0.
    for rank,index in enumerate(order):
        running=max(running, min(1., (len(values)-rank)*values[index])); adjusted[index]=running
    return dict(adjusted=adjusted.tolist(), reject=(adjusted<=alpha).tolist(), alpha=float(alpha))


def compare_arms(rows, *, candidate, baseline, cell_key='cell', seed=PRIMARY_SEED,
                 replicates=DEFAULT_BOOTSTRAPS):
    """Build paired cell differences from endpoint rows and bootstrap them."""
    grouped={}
    for row in rows:
        cell=str(row[cell_key]); arm=row.get('arm')
        if arm not in (candidate,baseline):continue
        if 'seed' not in row:raise ValueError('paired inference requires scene seed')
        value=row.get('TCR')
        if value is None and isinstance(row.get('endpoints'),dict):value=row['endpoints'].get('TCR')
        if value is None:raise ValueError('missing TCR')
        bucket=grouped.setdefault(cell,{}).setdefault(arm,{})
        if row['seed'] in bucket:raise ValueError('duplicate scene/arm (average frozen initializations first)')
        bucket[row['seed']]=float(value)
    cells={}
    for cell,arms in grouped.items():
        if candidate not in arms or baseline not in arms:raise ValueError(f'incomplete paired cell {cell}')
        if set(arms[candidate])!=set(arms[baseline]):raise ValueError(f'unpaired cell {cell}')
        cells[cell]=np.array([arms[candidate][s]-arms[baseline][s] for s in sorted(arms[candidate])])
    result=paired_stratified_bootstrap(cells,replicates=replicates,seed=seed)
    result['candidate']=candidate;result['baseline']=baseline
    result['p_centered']=centered_two_sided_pvalue(cells,replicates=replicates,seed=seed)
    return result


def average_frozen_initializations(rows):
    """Reduce exactly three fixed learned initializations to one paired scene row."""
    groups={};out=[]
    for row in rows:
        if row['arm'] not in ('p5-dhgc','p5-mappo'):
            out.append(dict(row));continue
        key=(row['cell'],row['seed'],row['arm']);bucket=groups.setdefault(key,{})
        init=row.get('init_seed')
        if init not in (33000,33001,33002) or init in bucket:raise ValueError('invalid/duplicate initialization')
        bucket[init]=row
    for (cell,seed,arm),bucket in sorted(groups.items()):
        if set(bucket)!={33000,33001,33002}:raise ValueError('all three frozen initializations required')
        values=[bucket[i]['endpoints']['TCR'] for i in (33000,33001,33002)]
        if not np.isfinite(np.asarray(values,dtype=float)).all():raise ValueError('missing initialization outcome')
        out.append(dict(cell=cell,seed=seed,arm=arm,TCR=float(np.mean(values)),initializations=[33000,33001,33002]))
    return out
