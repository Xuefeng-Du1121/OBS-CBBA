# P5 formal protocol v1.1 — explicit amendment to v1

Release candidate, **NOT AUTHORIZED TO RUN**. Read together with immutable
`p5_formal_protocol_v1.md`; only replacements below apply. No method, physical
setting, seed allocation, primary estimand, or primary testing rule changes.
The companion `p5_formal_protocol_v1_1.plan.json` is the executable plan.
Authority: user decisions R1/R2 in `review_claude_20261024_p5_protocol_r1.md`
and frozen training review `review_claude_20261024_p5_drl_r2.md`.

## §2 ablations: replace the variant list and cost

- REMOVE K16 and executed thresholds 0.5 and 1.0.
- RETAIN K1/K4/K8, base-only, compact K8, b scoring suffix, commitment-off,
  thresholds 0.05, 0.12456826310046251 and 0.25: ten unique variants.
- ALIAS threshold zero to full K8 and never-trigger to base-only. Execute each
  identity once; endpoints exclude network overhead. All three executed positive
  thresholds include inference and all scheduling work. No threshold selection.
- REPLACE 520 ablation episodes with 400 (40 paired scenes × ten variants).

## §1 secondary family: replace 23 with exactly 25

Keep the 12 cellwise primary contrasts, seven other pooled non-learned TCR
contrasts, and four pooled endpoints versus CBBA-PR. Add exactly two tests:
pooled TCR, proposed minus p5-dhgc; pooled TCR, proposed minus p5-mappo.
The same fixed Holm family at alpha .05 covers all 25. No initialization-level
or additional learned endpoint tests enter this family. Missing/unavailable
contrasts receive p=1; never shrink the family. Primary inference is unchanged.
For each learned adapter, average the three frozen initialization outcomes
within each scene before paired, cell-stratified bootstrap inference. Report
all initializations descriptively; inference is conditional on these models.

## §§1,3,4 learned timing: replace pre-main checkpoint requirement (R2)

Option B is adopted: p5-dhgc and p5-mappo, each U in {4,6}, initialization seeds
{33000,33001,33002}. Evaluate AFTER the main run, using its exact sealed scenario
bytes. Never regenerate or replace a scene. All 360 main scenes and all three
initializations are mandatory: 1,080 episodes/adapter, 2,160 total. No selection
by main results, checkpoint quality, missing outcomes, or initialization.

Frozen inputs are `p5_drl_training_card.md` and `p5_drl_training_spec.json`.
Spec SHA-256: `00918210679d80bde16c5b754fd83d4e9755c25da61a20ed6769ea0114c64b59`.
The exact card SHA is bound in the plan and release checklist. CPU; DHGC LR=1e-4,
MAPPO LR=3e-4; final episode E=6600; six final checkpoints per adapter, three per U.
The training card's older authorization wording is superseded by the accepted
r2 training review, not rewritten here. Training is separately authorized and
currently occupies ubuntu-b; this release preparation starts no server work.

Append the twelve checkpoint SHA-256 values, completion records, and their
initialization/U identities before learned evaluation. This is allowed after
main launch because the checkpoint production rule and all evaluation choices
are already frozen. The append-only checkpoint supplement must bind this plan,
training spec and main sealed-store receipts; it cannot revise their rules.
Incomplete training remains incomplete; no earlier/best checkpoint is eligible.

Main-run analysis tables for the non-learned arms may be produced once all main
arms and their audits finish, before learned evaluation. No partial-main effect
inspection. Early secondary tables retain the full 25-test family (pending tests
p=1) and are labeled provisional; recompute Holm once all tests are available.
Learned evaluation has no remaining degrees of freedom, even after those tables
are seen. Its episodes load the sealed main bytes and frozen final weights only.

## §4 budget: replace stale summary AND detailed assignments

Repack all retained jobs using the unchanged measured rates, nine single-core
lanes, replay allowance, 50% margin, and 300 s/batch overhead. The exact plan
contains 4,900 non-learned episodes, 56 batches, 305.3667561547879 projected
server hours and 336 reserved cap-hours. Ablations account for 400 episodes,
19 batches and 105.44847358068199 projected hours. Other module budgets are
unchanged. This supersedes the prerequisite's approximate 319.9 h/59 batches;
its detailed assignments still included removed variants and are not executable.

Learned evaluation adds twelve batches, 13 projected server hours; combined
projection 318.3667561547879 hours. Its 72 reserved cap-hours yield 408 total
cap-hours, which exceeds the previously discussed 390 h upper budget if every
cap is consumed. The authorization must therefore bind an explicit aggregate
compute ceiling within the approved scope; exhaustion means incomplete, never
unapproved extra compute. Training is separately accounted CPU time, not GPU
hours. Learned inference timings remain provisional, a release limitation.

All original one-shot, collision scan, independent audit, dual archive, and
no-relaunch rules remain. HYCOM has no substitution rule; an unavailable panel
blocks that module. Neither this amendment nor any included manifest authorizes
scenario generation. No real authorization file is supplied.

## R2 source identity clarification

The authorized training snapshot is commit
`a17ed6498dbaa6492f7062cb9ca1401e9d59ef06`, frozen by
`p5_drl_training_source.json` (per-file SHA-256 plus the original training
source-digest algorithm). The prerequisite merge changes five files in the
loader's broad source hash, although no learned policy code changes. Therefore
learned checkpoint validation AND evaluation execute the original, checksum-
verified training snapshot in an isolated CPU subprocess. The original loader
still verifies source/card/spec, E=6600, final validation count, method, U and
initialization. No guard is bypassed or weakened. Main scenarios remain the
sealed main-run bytes; only protocol endpoint extraction uses the release code.
The source supplement is frozen now, before any confirmation generation.
