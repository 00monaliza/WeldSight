"""Synthetic defect generation (stages 2-3, not implemented yet; see ROADMAP.md).

Planned modules:
    copy_paste.py  - Poisson/alpha blending of real defect crops into defect-free patches
    diffusion.py   - fine-tuned inpainting diffusion model conditioned on a defect mask + class
    detect_artifacts.py - real-vs-synthetic probe to check the segmenter did not learn
                          generation artefacts
Contract: every synthetic patch must be generated only from TRAIN-split radiographs and
be written with `source=synthetic` in the manifest; val/test stay 100% real.
"""
