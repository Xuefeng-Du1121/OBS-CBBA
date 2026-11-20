"""Harness check: replay row must equal archived episode on every non-timing metric and every selection."""
import csv, json, sys
from pathlib import Path
archive, rows = Path(sys.argv[1]), sys.argv[2]
skip = {'decision_compute_s', 'mean_assignment_compute_s', 'method'}
ok = True
for r in csv.DictReader(open(rows)):
    d = archive / r['batch'] / 'episodes' / r['episode_dir']
    ref = json.loads((d / 'metrics.json').read_text())
    bad = [k for k in ref if k not in skip and k in r and str(ref[k]) != r[k]
           and not (isinstance(ref[k], float) and abs(ref[k] - float(r[k])) < 1e-9)]
    sel = [t['selected'] for t in json.loads((d / 'trace.json').read_text())['traces']]
    same = sel == json.loads(r['selected'])
    print(r['seed'], 'metrics_mismatch=', bad, 'selections_equal=', same, 'n=', len(sel))
    ok &= (not bad) and same
print('HARNESS_OK' if ok else 'HARNESS_FAIL')
