# CR-CBBA: counterfactual commitment revision for online multi-UAV service of ocean-drifting buoys

Code and result records for the paper

> X. Du, Y. Cheng, L. Yin, F. Li. *Counterfactual commitment revision for online
> multi-UAV service of ocean-drifting buoys.* Manuscript submitted to Expert
> Systems with Applications.

The method was called DA-CBBA in earlier drafts; the code keeps `da-cbba` as an
alias of `cr-cbba`, and the result files use the internal name `o_obs-cbba`.

The repository contains the simulator, the allocators compared in the paper,
the scene definitions, the per-episode records of every reported experiment,
and a script that recomputes the reported statistics from those records.

## Contents

| Path | Content |
|---|---|
| `obs_cbba/` | Simulator, CR-CBBA and the non-learned baselines |
| `run_episode.py` | Run one method on one scene of the paper |
| `reproduce_stats.py`, `bootstrap.py` | Recompute the primary contrast, the 25-test secondary family and the 6 robustness contrasts |
| `data/scenes.json` | Configuration and seed range of each of the 41 cells (850 scenes) |
| `data/scene_digests.csv` | SHA-256 digest of the arrays of each scene |
| `data/hycom/` | Six HYCOM GOFS 3.1 surface-current panels used by the robustness cells, with provenance files |
| `checkpoints/trigger.pt` | Weights of the event trigger used in the trigger ablation |
| `results/episodes.csv` | One row per episode (7,060 rows): all modules, all methods |
| `results/selector_event_times.csv` | Planning time of every CR-CBBA call (74,577 calls) |
| `results/analysis_scene_drift.csv` | Realized drift speed of each kinematic-analysis scene |
| `results/tables/` | Summary tables behind the tables and figures of the paper |
| `results/posthoc/` | Forecast accuracy, real-time budget and clairvoyant bound (rows and summaries) |
| `results/posthoc/fixed_rules/` | Six fixed revision rules on the 360 main scenes |
| `results/posthoc/futures_delay/` | Forecast-content ablation (snapshot, no drift, no arrivals), budget, anytime and one-tick-delay replays |
| `results/v12/` | Confirmatory test of protocol v1.2 on seeds 40600-40959 (four arms, per-episode rows, results, scene index) |
| `results/revision/` | Revision-set decomposition, APR adapter, equal-computation baselines, penalized score, scale study (U = 10, 12), corrected-simulator reproduction, K=8 branch selections |
| `protocol/` | Protocol v1.2 (card and plan) and the SHA-256 digests of every protocol version (v1, v1.1, v1.2) |
| `scripts/` | Scripts of the post hoc studies, of protocol v1.2 and of the revision-round studies (see `scripts/README.md`) |

## Install

Python 3.10 or later.

```
pip install -r requirements.txt
```

`reproduce_stats.py` needs only numpy. `run_episode.py` also needs PyTorch
(CPU is enough) and SciPy.

## Recompute the statistics

```
python reproduce_stats.py
```

The script reads `results/episodes.csv`, recomputes each contrast and compares
it with the archived table in `results/tables/`. It takes about one minute and
exits with code 0 when every value matches. The first lines of the output are

```
Primary contrast, CR-CBBA minus CBBA-PR, TCR in percentage points
  4.00 [3.72, 4.28], p = 2e-05, 360 scenes in 12 cells
```

Statistical rule: paired differences per scene, resampled within cells
(50,000 PCG64 resamples; seed 690524 for the main matrix, 690525 for
robustness), equal cell weights, percentile intervals, centred two-sided
p-values, Holm correction at 0.05. HYCOM cells are stratified by current panel.
Learned baselines are averaged over their three initializations within each
scene.

## Run one episode

```
python run_episode.py --list                              # cells and seed ranges
python run_episode.py --seed 40030 --method cbba-pr
python run_episode.py --seed 40030 --method cr-cbba --workers 4
```

The script generates the scene from its seed, checks the scene arrays against
`data/scene_digests.csv`, runs the method, and prints the archived row of
`results/episodes.csv` for the same scene and method next to the new result.

Method names:

| Paper | `--method` | Name in `results/` |
|---|---|---|
| Persist-A | `persist-a` | `a` |
| Defer-B | `defer-b` | `b` |
| Regret-B | `regret-b` | `regret-b` |
| Fixed-F | `fixed-f` | `f` |
| CBBA | `cbba` | `cbba` |
| CBBA-PR | `cbba-pr` | `cbba-pr` |
| MSA | `msa` | `msa` |
| OBS-LA | `obs-la` | `o_obs` |
| CR-CBBA | `cr-cbba` | `o_obs-cbba` |
| HRL-2L (learned) | not included | `p5-dhgc` |
| MAPPO (learned) | not included | `p5-mappo` |

Ablation variants of CR-CBBA (seeds 31400–31439): `full_k8`, `full_k4`,
`full_k1` (number of sampled futures), `compact_k8` (three-branch set),
`suffix_b` (Defer-B continuation), `commitment_off` (no deferred commitment),
`base_only` (CBBA-PR), and `trigger_0.05`, `trigger_0.12456826310046251`,
`trigger_0.25` (event trigger with that threshold). `defer-b-forecast`
(`b_fc`) is Defer-B with the drift forecast; it appears in the kinematic
analysis records only.

Running time: the fixed-order and CBBA-family baselines take seconds per
episode. CR-CBBA takes minutes per episode on one core (the archived runs used
one core per episode); `--workers` evaluates the sampled futures in parallel
and does not change the decisions.

## Scenes

All scenes are generated before any method acts, from the seed and the cell
configuration in `data/scenes.json`. Every method sees the same scene, so all
comparisons are paired.

| Module | Seeds | Cells |
|---|---|---|
| main | 40000–40359 | 12 cells: U ∈ {4, 6} × N ∈ {80, 120, 180} × {medium, high} arrivals, 30 scenes each |
| robustness | 40360–40599 | wind, anchored and HYCOM forcing at U ∈ {4, 6}; 30 scenes per cell, 60 for HYCOM |
| ablations | 31400–31439 | U = 6, N = 120, medium and high arrivals, 20 scenes each |
| availability | 31460–31489 | UAV loss probability 0, 0.1, 0.3; 10 scenes each |
| analysis | 31700–31879 | domain side × vehicle speed × N, 10 scenes each |
| v1.2 confirmatory | 40600–40959 | the 12 main cells, 30 new scenes each; 40960–40999 sealed |
| scale (post hoc) | 41000–41089 | U = 10, N = 200; U = 12, N = 240 and 360; high arrivals; 30 scenes each |

`data/scene_digests.csv` holds the SHA-256 digest of the four scene arrays
(`positions`, `release`, `deadline`, `response_uniform`; data type, shape and
bytes). The digests were computed from the scenes of the original runs (Linux,
x86-64, numpy 1.26). Regeneration on macOS (arm64, numpy 2.3) matches all 850
of them. If a platform produces different
floating-point values, `run_episode.py` reports the mismatch and the result is
then not comparable row by row.

The `scenario_sha256` column of `results/episodes.csv` is the hash of the
stored scene file of the original run, not the array digest above.

## Code version

The confirmatory runs (main, robustness, ablations, availability, learned
baselines) and the kinematic analysis were produced by two revisions of the
simulator that differ in one rule: the kinematic-analysis revision corrects the
separation check for a pair of UAVs leaving the depot zone (`obs_cbba/env.py`,
`violates` inside `tick`). This repository contains the later revision. The
correction was introduced for the 10-km analysis scenes. We re-ran all 360
main scenes with CBBA-PR and with the snapshot configuration of CR-CBBA under
this code, and all 720 episodes equal their archived rows in completion, PST and
energy (`results/revision/pr__main718_rows.csv`, `snap__main718_rows.csv`);
`run_episode.py` shows the comparison for any scene. The v1.2 test and the
main-scene arms of the revision round used the earlier revision, the scale
study the later one.

The strings `NAMESPACE` and `SCHEMA` in `obs_cbba/` seed the sampled futures
and identify the trigger checkpoint. Changing them changes the results.

Software used for the archived runs: Python 3.12.3, numpy 1.26.4,
PyTorch 2.5.1, SciPy 1.14.1 (Linux, x86-64).

## Not included

The training code and weights of the two learned baselines (HRL-2L, internal name `p5-dhgc`, and MAPPO),
the stored scene files, the full per-episode archive (event logs and audits)
and the figure scripts are not part of this repository. The episode records of
the learned baselines are in `results/episodes.csv`. The other material is
available from the corresponding author on reasonable request.

## Data source

`data/hycom/` holds subsets of the HYCOM GOFS 3.1 global analysis
(GLBy0.08, experiment 93.0), surface level, three North-Atlantic locations and
two dates. Each `.nc.json` file records the request and the SHA-256 of the
panel. The HYCOM data are public (DoD Distribution A, approved for public
release). The fields drive synthetic buoy trajectories; they are not measured
buoy tracks.

## Licence

MIT, see `LICENSE`.
