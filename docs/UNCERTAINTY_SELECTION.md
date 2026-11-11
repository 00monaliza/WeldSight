# Choosing the uncertainty score on validation (pre-registered)

Written and committed **before** any validation-split uncertainty numbers were computed.

## Why

Stage 1 found on the **test** split that mutual information (the pre-registered score) flags
errors with AUROC 0.54, while predictive entropy reaches 0.82. Switching the score because of
that would tune on test. The score must be chosen on val, then applied to test once.

## Rule

- Model: the `riawelc_cpu` run (seed 42, data_version `2c33eedecb1697b9`, best epoch by val
  macro-F1). MC Dropout T = 10, threshold 0.5. Nothing is retrained for this.
- Candidates: `predictive_entropy`, `expected_entropy`, `mutual_information`, `max_prob`
  (radiograph score = max over its patches, as in Stage 1).
- **Primary criterion:** area under the radiograph-level referral curve (all classes) on
  **val**. This is the operational target: recall after sending the most uncertain
  radiographs to an inspector.
- Tie (difference < 0.005 in area): pick the higher patch-level error-detection AUROC on val.
- The chosen score is then evaluated **once** on test. Test numbers for all four scores are
  also reported, labelled as post-hoc, so the reader can see whether the val choice transferred.

## Known limitation, stated in advance

Val has 6 weld specimens. Stage 1 already showed a large val→test gap (macro-F1 0.89 → 0.61)
caused by specimen-to-specimen variance. A score chosen on 6 specimens may not transfer.
The proper fix is specimen-level k-fold CV (ROADMAP), not this single split.

## Result (computed 2026-11-11, after this rule was committed)

Model: `riawelc_cpu`, best epoch 5, retrained from scratch; val/test metrics identical to the
README run. Training was stopped after epoch 5 (the 2026-10-28 rerun shows epochs 6-9 never
beat it), outputs in `docs/runs/uncertainty_selection_20261111/`.

**Val (24 radiographs, 6 specimens)** — the selection:

| score | referral area | error AUROC |
|---|---|---|
| **max_prob (chosen)** | **0.961** | 0.843 |
| predictive_entropy | 0.953 | 0.795 |
| expected_entropy | 0.953 | 0.788 |
| mutual_information (pre-registered) | 0.934 | 0.731 |
| random / oracle | 0.932 / 0.973 | |

**Test (21 radiographs, 6 specimens)** — applied once, all scores shown post hoc:

| score | referral area | error AUROC |
|---|---|---|
| predictive_entropy | 0.852 | 0.819 |
| expected_entropy | 0.852 | 0.820 |
| **max_prob (chosen on val)** | **0.837** | 0.804 |
| mutual_information (pre-registered) | 0.832 | 0.537 |
| random / oracle | 0.801 / 0.875 | |

**Reading.**
1. On val, mutual information is no better than random referral (0.934 vs 0.932). The
   pre-registered score is the weakest on *both* splits, so dropping it is now justified
   without looking at test.
2. The val choice (max_prob) beats mutual information on test (0.837 vs 0.832), but it is
   not the best score on test: predictive entropy is (0.852). The val ranking of the three
   entropy-like scores did not transfer. With 6 specimens per split, differences of ~0.01
   in area are within specimen-to-specimen noise.
3. Conclusion for the next stages: use a total-uncertainty score (predictive entropy or
   max-prob), not the epistemic part (MI), and settle the choice with specimen-level k-fold
   CV, not with one val split.
