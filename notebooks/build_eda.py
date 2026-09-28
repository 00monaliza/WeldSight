"""Builds notebooks/01_eda.ipynb (run: uv run python notebooks/build_eda.py, then execute)."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# WeldSight — EDA

Dataset: the prepared manifest in `data/processed/<name>` (default RIAWELC, re-split by weld specimen).
Sections: provenance · audit of the published split · class balance · effective sample size ·
defect size (mask datasets only) · examples per class.""")
code("""import json, hashlib, re
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, cv2

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROC = ROOT / "data/processed/riawelc_v1"
RAW = ROOT / "data/raw/riawelc"
FIG = ROOT / "notebooks/figures"; FIG.mkdir(exist_ok=True)
df = pd.read_csv(PROC / "manifest.csv", keep_default_na=False)
card = json.loads((PROC / "dataset_card.json").read_text())
CLS = card["class_names"]
df["label"] = "no_defect"
for c in CLS:
    df.loc[df[f"cls_{c}"] == 1, "label"] = c
{k: card[k] for k in ["name", "source", "license", "supervision", "n_records_raw", "n_records",
                      "duplicates_dropped", "n_patches", "patches_per_split", "radiographs_per_split",
                      "data_version"]}""")
md("""## 1. Audit of the split published with RIAWELC

RIAWELC ships `training/validation/testing` folders. File names encode the source radiograph
(`RRT-26R_Img2_...`), so we can check whether those splits are independent.""")
code("""pat = re.compile(r"^(?P<spec>.+?)_Img(?P<img>\\d+)_A\\d+_S\\d+_\\[\\d+\\]\\[\\d+\\].*\\.png$")
rows = []
if RAW.exists():
    for f in RAW.rglob("*.png"):
        m = pat.match(f.name)
        rows.append({"orig_split": f.parent.parent.name, "folder": f.parent.name, "spec": m["spec"],
                     "radiograph": f"{m['spec']}_Img{m['img']}",
                     "md5": hashlib.md5(f.read_bytes()).hexdigest()})
raw = pd.DataFrame(rows)
if len(raw):
    rs = {s: set(raw.loc[raw.orig_split == s, "radiograph"]) for s in ["training", "validation", "testing"]}
    dup = raw.groupby("md5")["orig_split"].nunique()
    audit = {
        "files": len(raw),
        "radiographs total": raw.radiograph.nunique(),
        "weld specimens total": raw.spec.nunique(),
        "radiographs in testing": len(rs["testing"]),
        "... of which also in training": len(rs["testing"] & rs["training"]),
        "byte-identical files": int((raw.groupby("md5").size() - 1).sum()),
        "... md5 groups spanning >1 split": int((dup > 1).sum()),
    }
    display(pd.Series(audit, name="published RIAWELC split"))
else:
    print("raw RIAWELC not found — skip audit")""")
md("""**Conclusion.** The published split is patch-level: almost every radiograph appears in both
training and testing, and thousands of files are exact copies across splits. Accuracies reported on
it measure memorisation of radiographs, not generalisation. We drop duplicates and re-split by
**weld specimen** (all films of a specimen go to one split).""")
md("## 2. Class balance after the specimen-level split")
code("""ct = pd.crosstab(df["label"], df["split"])[["train", "val", "test"]]
display(ct)
ax = (ct / ct.sum()).T.plot.bar(stacked=True, figsize=(6, 3.5), colormap="tab10")
ax.set_ylabel("share of patches"); ax.set_title("Patch label share per split"); ax.legend(bbox_to_anchor=(1, 1))
plt.tight_layout(); plt.savefig(FIG / "class_balance.png", dpi=120); plt.show()""")
md("""## 3. Effective sample size: patches ≠ independent samples

What matters for generalisation is how many *specimens* show each defect class.""")
code("""spec = pd.crosstab(df["group_id"], df["label"])
display(pd.DataFrame({
    "specimens with class": (spec > 0).sum(),
    "patches": spec.sum(),
    "max share from one specimen": (spec.max() / spec.sum()).round(2),
}))
per_split = df[df.label != "no_defect"].groupby(["split", "label"])["group_id"].nunique().unstack()
display(per_split.rename_axis(columns="specimens per split"))
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(np.log1p(spec[["no_defect"] + CLS].values), aspect="auto", cmap="viridis")
ax.set_yticks(range(len(spec))); ax.set_yticklabels([f"{g} ({df.loc[df.group_id==g,'split'].iat[0]})" for g in spec.index], fontsize=7)
ax.set_xticks(range(len(CLS) + 1)); ax.set_xticklabels(["no_defect"] + CLS, rotation=20)
plt.colorbar(im, label="log(1 + patches)"); ax.set_title("Patches per specimen and class")
plt.tight_layout(); plt.savefig(FIG / "specimen_heatmap.png", dpi=120); plt.show()""")
md("""**Conclusion.** ~21k patches come from only 29 weld specimens. Lack of penetration occurs in
~10 specimens, so the test set sees it in only 2–3 specimens: per-class recall has wide confidence
intervals (we report cluster-bootstrap CIs over radiographs).""")
md("## 4. Defect size")
code("""px_cols = [f"px_{c}" for c in CLS if f"px_{c}" in df]
if px_cols:
    areas = []
    for p in df.loc[df.has_mask == 1, "mask"]:
        m = cv2.imread(str(PROC / p), cv2.IMREAD_GRAYSCALE)
        for k, c in enumerate(CLS, 1):
            n, lab, st, _ = cv2.connectedComponentsWithStats((m == k).astype(np.uint8))
            areas += [{"class": c, "area": a, "w": w, "h": h} for (_, _, w, h, a) in st[1:]]
    areas = pd.DataFrame(areas)
    display(areas.groupby("class")["area"].describe())
    areas.boxplot("area", by="class", figsize=(6, 3.5)); plt.yscale("log"); plt.show()
else:
    print("No pixel masks in this dataset (label-only supervision) -> defect size cannot be measured.")""")
md("""For label-only data we show a *proxy*: intensity contrast of the patch (std and the share of
pixels more than 2σ darker than the patch median). It is **not** a defect size measurement.""")
code("""rng = np.random.default_rng(0)
stats = []
for lab, g in df.groupby("label"):
    for p in g.sample(min(400, len(g)), random_state=0)["image"]:
        x = cv2.imread(str(PROC / p), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        s = x.std() + 1e-6
        stats.append({"label": lab, "std": s, "dark_share": float((x < np.median(x) - 2 * s).mean()),
                      "mean": x.mean()})
stats = pd.DataFrame(stats)
display(stats.groupby("label").median().round(3))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
stats.boxplot("std", by="label", ax=ax[0]); stats.boxplot("dark_share", by="label", ax=ax[1])
for a in ax: a.tick_params(axis="x", rotation=20)
plt.suptitle(""); plt.tight_layout(); plt.savefig(FIG / "contrast_proxy.png", dpi=120); plt.show()""")
md("""**Warning — shortcut.** No-defect patches have a much lower intensity std (~7 vs ~25 grey
levels) than any defect class: they were cropped from flat regions. A trivial std threshold nearly
separates *defect vs no defect*, so high "any-defect" recall on RIAWELC is not evidence of a good
detector. The informative part is discrimination **between** defect classes and recall per class.""")
code("""from sklearn.metrics import roc_auc_score
print("AUROC of patch std for defect-vs-no-defect:",
      round(roc_auc_score(stats["label"] != "no_defect", stats["std"]), 3))""")
md("## 5. Examples per class (one patch per specimen where possible)")
code("""n = 8
fig, axes = plt.subplots(len(CLS) + 1, n, figsize=(n * 1.6, (len(CLS) + 1) * 1.7))
for r, lab in enumerate(["no_defect"] + CLS):
    g = df[df.label == lab].groupby("group_id").sample(1, random_state=0)
    g = g.sample(min(n, len(g)), random_state=0)
    for c in range(n):
        ax = axes[r, c]; ax.axis("off")
        if c < len(g):
            ax.imshow(cv2.imread(str(PROC / g.iloc[c]["image"]), cv2.IMREAD_GRAYSCALE), cmap="gray")
            ax.set_title(g.iloc[c]["group_id"], fontsize=7)
    axes[r, 0].text(-40, 128, lab, rotation=90, va="center", ha="right", fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "examples.png", dpi=110); plt.show()""")
md("""## Take-aways for modelling
1. Evaluate only on the specimen-level split; never on the published one.
2. Report per-class recall with specimen/radiograph-level CIs — test has 6 specimens.
3. RIAWELC has no masks: segmentation IoU requires SWRD (polygons) or GDXray (binary masks).
4. Brightness/contrast differ strongly between specimens -> per-patch normalisation.
5. No-defect patches are trivially separable by contrast; do not use "defect vs no defect" accuracy
   as the headline metric.""")

nb = nbf.v4.new_notebook(
    cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}}
)
nbf.write(nb, "notebooks/01_eda.ipynb")
print("written")
