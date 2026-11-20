# P5 protocol v1.2: confirmatory replication of the real-time configurations

Amendment to `p5_formal_protocol_v1.md` and `p5_formal_protocol_v1_1.md`.
Frozen on 2026-11-12, before any scene with seed 40600-40999 was generated.
Nothing in v1 or v1.1 changes: the primary study, its estimand, its tests and
its 360 + 240 confirmation scenes (seeds 40000-40599) stay as reported.

## Why

All real-time results in the first submission (25-s budget, snapshot
configuration, one-tick execution delay) were post hoc replays on scenes whose
outcomes were already known. This amendment tests the two deployable
configurations on scenes that nobody has seen, with estimands, tests and
analysis fixed in advance.

## Seeds and scenes

- Main-matrix cells exactly as in v1.1 (12 cells: U in {4,6}, N in {80,120,180},
  medium and high arrivals; configs copied byte for byte from
  `p5_formal_protocol_v1_1.plan.json`).
- 30 scenes per cell, 360 scenes. Cell i (v1.1 order) uses seeds
  40600 + 30 i ... 40629 + 30 i, so seeds 40600-40959.
- Seeds 40960-40999 stay sealed and unused. No replacement, no re-run, no exclusion.
- Each scene is generated once with `Scenario.generate(Config(**config), seed)`
  (the same generator that produced the archived main scenes; checked to
  reproduce archived seeds 40061, 40212 and 40335 bit for bit), written once with
  exclusive creation, made read-only and hashed. All arms read the same bytes.

## Arms (all on every scene)

| Arm | Meaning |
|---|---|
| `pr` | CBBA-PR (reopen the last 2 unstarted tasks per bundle), decisions applied at once |
| `snap` | Snapshot configuration: the proposed branch set and CBBA-PR continuation scored on one future with no drift and no arrivals (K=1). 25-s budget per call; a call that ends after 25 s applies CBBA-PR instead (fallback) |
| `pr_d1` | CBBA-PR, every decision applied one tick (25 s) after its observation |
| `snap_d1a` | Snapshot configuration under the one-tick delay, branches scored as they will execute (delay-aware), 25-s budget with CBBA-PR fallback |

Code: `v12_run.py` (this folder), which imports the frozen samplers and the
delay-aware evaluator from `../20261110/p5_eswa_extra/r2_r3_extra.py`, and the
planner from the formal release `09ee2cb`. SHA-256 values are recorded in
`p5_protocol_v1_2.plan.json`. No parameter of the method changes.

## Confirmatory family (Holm, alpha 0.05, two tests)

All on pooled TCR, cell-stratified paired bootstrap with equal cell weights,
50,000 resamples, PCG64 seed 20261112, centred two-sided p values as in v1.

- H1: `snap` minus `pr` (deployable configuration, no delay).
- H2: `snap_d1a` minus `pr_d1` (deployable configuration, one-tick delay).

Expected direction for both: positive. A test that cannot be computed gets p = 1.

## Pre-specified descriptive quantities (no tests)

- Per-cell paired differences with 95% scene bootstrap intervals (10,000 resamples, seed 20261112).
- The full K=8 method is not re-run (about 15 h of single-machine compute). Its
  gain on the v1 main matrix (+4.00 pp) is quoted as a reference only, and no
  share of it is computed on these scenes.
- `snap_d1a` minus `pr` (the cost of the delay plus the gain of the method).
- Never-assigned and lost-after-assignment shares per arm.
- Per-call wall time (median, 95th percentile, maximum) and the number of
  fallback calls for `snap` and `snap_d1a`, with the hardware used.

## Hardware and timing

The run uses one machine; its CPU model is recorded in the results. Decisions
are deterministic given the scene; the wall clock enters only through the 25-s
fallback. The number of calls that hit the fallback is reported, so a reader can
see whether timing changed any decision.

## What is reported

All four arms, all 360 scenes, whatever the outcome. If a scene fails, the
failure is reported and its arms are missing for every contrast; nothing is
re-run.
