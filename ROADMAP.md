# WeldSight roadmap

Stage 1 (done): data pipeline with a specimen-level split, EDA, U-Net baseline (Dice + Focal),
per-class metrics, MC-Dropout referral curve, tests. See README.

## Protocol shared by stages 2 and 3

These rules exist so that the three arms (none / copy-paste / diffusion) differ in exactly one thing.

- **Same split, same test set.** Split by weld specimen (`data_version` in `dataset_card.json`).
  Val and test are always 100 % real radiographs; synthetic patches exist only in train and are
  tagged `source=synthetic` in the manifest.
- **Synthesis only from train.** Defect crops, masks and the diffusion fine-tuning set come from
  train-split specimens only. Otherwise test defects leak into train through the generator.
- **Same budget.** Equal number of optimisation steps and the same basic photometric
  augmentation in every arm. "No augmentation" means no *defect* augmentation.
- **Metrics that cannot be gamed by a threshold.** Adding rare-class examples shifts calibration,
  and a lower threshold raises recall for free. Primary metric per rare class: **recall at a fixed
  false-positive rate** (threshold chosen on val so that specificity = 95 %), plus PR-AUC. Plain
  recall@0.5 is reported only as a secondary number.
- **Uncertainty over seeds and specimens.** ≥ 3 training seeds per arm, and a paired cluster
  bootstrap over test radiographs for the arm difference. Only 2–4 test specimens contain each
  defect class, so a single-seed difference of a few points is noise.
- **Rare classes are defined up front** from train counts (by specimens, not patches), before any
  arm is run.

## Stage 2: copy-paste augmentation of rare classes

1. Build a defect bank from train masks (SWRD/GDXray): crop each connected component with a
   margin, store class, area and local background statistics.
2. Paste onto defect-free train patches: random flip/rotation (±10°, welds are oriented), scale
   0.8–1.2, placement restricted to the weld bead (bead mask from an intensity profile or a
   Stage-1 model), blending by Poisson or alpha-feathering + local intensity matching.
   Radiographs are (approximately) multiplicative in transmitted intensity, so blend in log-intensity.
3. Oversampling ratio for rare classes: 1× / 2× / 4× of the original count.
4. Compare with baseline under the protocol above. Also report common-class recall to catch
   regressions.

Label-only data (RIAWELC) cannot provide defect masks for cut-outs. There, the only honest
copy-paste variant is patch-level mixing (e.g. CutMix of defect patches onto no-defect patches,
with the label of the pasted patch). This is a weaker baseline and is reported separately.

## Stage 3: diffusion inpainting of defects

1. **Model.** Small latent-free DDPM/UNet (≈ 30–60 M params) at 256 × 256, grayscale, conditioned
   on (masked image, binary mask, class embedding). Alternative: fine-tune
   `stable-diffusion-inpainting` with LoRA. It is heavier and pre-trained on RGB photos, so the
   domain gap is larger. Pick one after a 1-day feasibility test on T4 (memory, it/s).
2. **Training data.** Train-split patches with defects, and masks from SWRD/GDXray. The mask comes
   from the annotation, so the generated defect has a known segmentation label for free.
3. **Generation.** Sample masks from the empirical shape distribution of the class (or reuse real
   train masks with random affine transforms) and place them on defect-free train patches. Use a
   fixed seed per synthetic patch and store the generator checkpoint hash in the manifest.
4. **Ablation.** Share of synthetic rare-class patches in train: **0 / 25 / 50 / 100 %** (relative
   to the number of real rare-class patches). Also a "synthetic only for rare classes, no real
   rare examples" point, which shows whether synthetic data alone carries signal. Three seeds each.
5. **Did the segmenter learn generation artefacts?**
   - *Real-vs-synthetic probe.* Train a small classifier to separate real and synthetic defect
     patches. AUROC ≈ 0.5 is good. High AUROC means artefacts exist and the next checks matter.
   - *Frequency check.* Compare the radially averaged power spectra of real and synthetic
     patches; diffusion upsampling often leaves high-frequency peaks.
   - *Shortcut test.* On a held-out synthetic set, compare recall on synthetic vs real defects.
     Much higher recall on synthetic data means the model keys on generation traces.
   - *Grad-CAM / occlusion* inside vs outside the mask. Attention on the blend boundary or
     background texture is a red flag.
   - *Counterfactual.* Take real test patches, inpaint a region **without** a defect (empty mask
     class). If the model now predicts a defect there, it has learnt "inpainted ⇒ defect".
   - Report FID/KID between real and synthetic defect patches (Inception features are
     poorly suited to radiographs, so treat it as a sanity check, not the main evidence).
6. **Outcome table.** Arms × rare classes, recall@FPR=5 % with CIs, PR-AUC, the effect on common
   classes, and the referral curve from Stage 1 for every arm. Did the synthetic data also make
   the model's uncertainty better, or only more confident?

## Known risks / open questions

- **Data.** RIAWELC has no masks, and GDXray has 10 annotated radiographs with a single class.
  SWRD is the only public multi-class polygon dataset found; its licence is unclear
  (cite + ask the maintainers). Everything in stages 2–3 that needs masks depends on it.
- **Tiny test sets.** 6 specimens in test. Consider k-fold cross-validation by specimen
  (e.g. 5 folds) for the final comparison instead of a single split.
- **Upper-bound referral.** The referral curve assumes the inspector finds all defects on referred
  radiographs. Real inspectors miss defects, so treat the curve as an upper bound.
