# Scripts of the post hoc and confirmatory studies

These are the scripts as they were run, with the package name changed from the
internal `dhgc` to `obs_cbba`. They read and write files in the layout of the
original working directory; the table below maps that layout to this
repository. Run them from their own folder after adjusting the input paths.

| Path in the scripts | In this repository |
|---|---|
| `../20261024/p5_results_final/episodes.csv` | `results/episodes.csv` |
| `../20261024/p5_results_final/results/*.csv` | `results/tables/*.csv` |
| `../20261110/p5_eswa_extra/fixed_out/` | `results/posthoc/fixed_rules/` |
| `../20261110/p5_eswa_extra/r2r3_out/` | `results/posthoc/futures_delay/` |
| `../20261112_v12/out/` | `results/v12/` |
| `out/`, `out718/` (revision round) | `results/revision/` |
| `tables/` | LaTeX tables of the paper (written, not read) |

| Folder | Study |
|---|---|
| `posthoc/` | Fixed revision rules, forecast-content ablation, real-time budget, anytime and one-tick-delay replays, drift-estimator accuracy, clairvoyant bound |
| `v12/` | Confirmatory test of protocol v1.2 (runner, analysis, table) |
| `revision/` | Revision-set decomposition, static and multi-start scores, APR adapter (`apr_event` in `rev_run.py`), MSA with 64 scenarios, penalized score, scale study, corrected-simulator reproduction, K=8 branch selections, failure-cost sensitivity |

`rev_run.py gen` regenerates the scenes from `data/scenes.json` and checks each
against `data/scene_digests.csv`; `rev_run.py run` runs one or more arms
(`--arms`) and writes one row per episode with the route-evaluation count of
every planner call. Post hoc intervals use a cell-stratified paired bootstrap
with equal cell weights (`boot.py`, 10,000 resamples).

The episode rows of the main-scene arms were produced with the simulator
revision used for the main study; `pr__main718_rows.csv` and
`snap__main718_rows.csv` repeat CBBA-PR and the snapshot configuration with the
revision in this repository and equal the archived rows in all 720 episodes.
