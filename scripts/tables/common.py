import csv, statistics as st
from collections import defaultdict
from pathlib import Path
BASE = str(Path(__file__).resolve().parents[2])                 # repository root
EP = f'{BASE}/results/episodes.csv'

def episodes(module):
    """{arm: {(cell, seed): row-dict of floats}}; learned arms averaged over the 3 inits per scene."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(EP)):
        if r['status'] != 'complete':
            continue
        mod = r['module']
        if mod == 'learned':
            mod = 'main'
        if mod != module:
            continue
        acc[r['arm']][(r['cell'], int(r['seed']))].append(r)
    out = {}
    for a, d in acc.items():
        out[a] = {}
        for k, rows in d.items():
            m = {}
            for f in ('TCR', 'PST', 'delay_seconds', 'energy', 'per_decision_latency_seconds'):
                vals = [float(x[f]) for x in rows if x[f] not in ('', 'nan')]
                m[f] = sum(vals) / len(vals) if vals else float('nan')
            m['n'] = len(rows)
            out[a][k] = m
    return out

def cellmean(d, f, scale=1.0):
    cells = defaultdict(list)
    for (c, s), m in d.items():
        cells[c].append(m[f] * scale)
    return st.mean(st.mean(v) for v in cells.values())

def sd(vals):
    return st.stdev(vals)
