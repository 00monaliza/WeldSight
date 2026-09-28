# Public weld radiograph datasets: review (checked 2026-09-28)

| | **RIAWELC** | **GDXray+ Welds** | **SWRD** |
|---|---|---|---|
| Source | [github.com/stefyste/RIAWELC](https://github.com/stefyste/RIAWELC) | [github.com/computervision-xray-testing/GDXray](https://github.com/computervision-xray-testing/GDXray) (Dropbox link to `Welds.zip`, 209 MB) | [github.com/bit628/RapidX-Annotator](https://github.com/bit628/RapidX-Annotator) (Google Drive link) |
| Paper | Totino, Spagnolo, Perri, ICMECE 2022; Perri et al., Manufacturing Letters 2023 | Mery et al., J. Nondestruct. Eval. 34(4), 2015 | Zhao et al., J. Nondestruct. Eval. 44, 2025 |
| Content | 24,407 PNG patches 227×227, 8-bit | W0001: 10 radiographs (~4k px long); W0002: binary masks for W0001; W0003: 67 radiographs (BAM round-robin), mostly unlabelled | ~3,600 seam-weld radiographs (standard + T-joint) |
| Classes | crack (CR), porosity (PO), lack of penetration (LP), no defect (ND) | one class: "defect" (no type) | porosity, inclusion, crack, undercut, lack of fusion, lack of penetration |
| Annotation | **class label per patch only**, no masks, no boxes | **pixel masks** (binary) for 10 images | **polygons** (LabelMe-style JSON / VOC XML), which can be rasterised to masks |
| Licence | none stated; README says "released freely", cite 2 papers | research and education only; **redistribution and commercial use prohibited**; cite paper | not stated; maintainers ask to cite and to contact them before redistribution or commercial use |
| Reachable from this dev container | yes (git clone) | no (Dropbox blocked) | no (Google Drive blocked) |

## RIAWELC facts we verified

- Folder → class: `Difetto1` = crack (7,635 files), `Difetto2` = porosity (6,320),
  `Difetto4` = lack of penetration (4,452), `NoDifetto` = no defect (6,000). These counts match
  the paper, and visual inspection agrees (round blobs in Difetto2, thin jagged lines in
  Difetto1, wide straight bands in Difetto4).
- File names encode the source: `RRT-26R_Img2_A80_S4_[12][84].png` → weld specimen `RRT-26R`
  (BAM round-robin test film series, the same origin as GDXray W0003), film `Img2`, grid
  position. That gives **29 specimens / 109 radiographs** in total.
- **The published split leaks.** 108 of 109 radiographs appear in both `training` and `testing`,
  and 2,443 files are byte-identical copies across splits (including 30 `... - Copia.png`).
  Results reported on the published split measure memorisation.
- **Selection bias.** No-defect patches are much flatter (median std 7 vs ~25 grey levels).
  Patch std alone separates defect from no-defect with AUROC 0.91.
- Images are 227×227, not 224×224 as stated in the README.

## Implication for the research question

The study needs per-class *segmentation masks* (IoU, and diffusion **inpainting** needs a mask).

| Option | What you get | Cost / risk |
|---|---|---|
| **A. SWRD polygons → masks** (recommended main dataset) | real multi-class segmentation, 6 classes incl. rare ones | manual download; licence must be confirmed with the authors; label names need verifying (`--inspect`) |
| B. GDXray W0001/W0002 | real pixel masks | 10 radiographs, **one class**: no per-class recall and no rare classes; too small for a study |
| C. RIAWELC as patch classification (implemented, used for the stage-1 run here) | class recall per patch, MC-Dropout referral | no IoU; diffusion inpainting has no masks to condition on |
| D. RIAWELC + weak masks (CAM / thresholding → pseudo-masks) | a "segmentation" on RIAWELC | pseudo-masks are not ground truth; IoU against them is circular; not recommended |
| E. SWRD boxes → box masks | works if only boxes are available | IoU vs. box masks overstates the defect area of thin cracks |
