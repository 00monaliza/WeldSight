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
