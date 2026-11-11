"""Builds notebooks/03_swrd_gpu.ipynb: SWRD segmentation baseline on a Colab T4.

uv run python notebooks/build_swrd_colab.py
"""

import nbformat as nbf

c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
code = lambda s: c.append(nbf.v4.new_code_cell(s))

md("""# WeldSight: SWRD segmentation baseline on a single GPU (Colab T4)

SWRD (Zhao et al., J. Nondestruct. Eval. 2025): seam-weld radiographs, polygon labels for 6 defect classes.
Licence is not stated: research use only, cite the paper, do **not** redistribute the images.

**How to run:** Runtime → Change runtime type → **T4 GPU**, then Runtime → **Run all**.

* Subset v1 (2000 weld crops) is already on your Drive in `MyDrive/weldsight/swrd_raw`; the notebook checks
  that it matches the frozen list `MyDrive/weldsight/subset_v1.txt` and only touches the 116 GB archive if it does not.
* `prepare` must reproduce `data_version aa609714f665f727` (the version the rare classes were pre-registered on);
  the notebook stops otherwise.
* Runs are written to `MyDrive/weldsight/runs`, with a checkpoint after every epoch. **If Colab disconnects,
  just Run all again**: an unfinished run is resumed from its `last.pt`, not restarted.""")

code("""BRANCH = "main"
EXPECTED_DATA_VERSION = "aa609714f665f727"
from google.colab import drive
drive.mount('/content/drive')
DRIVE = "/content/drive/MyDrive/weldsight"
RAW, RUNS = f"{DRIVE}/swrd_raw", f"{DRIVE}/runs"
%cd /content
!test -d WeldSight || git clone -b {BRANCH} https://github.com/00monaliza/WeldSight.git
%cd /content/WeldSight
!git fetch -q origin {BRANCH} && git checkout -q {BRANCH} && git pull -q origin {BRANCH} && git log --oneline -1
!pip -q install uv && uv sync -q --extra dev
!mkdir -p "{RAW}" "{RUNS}" data/raw && ln -sfn "{RAW}" data/raw/swrd
!nvidia-smi --query-gpu=name,memory.total --format=csv""")

md("""## 1. Is subset v1 already on Drive?
Compares `swrd_raw/_subset.txt` with the frozen `subset_v1.txt` and counts the image files.""")
code("""import os
from pathlib import Path
frozen = Path(f"{DRIVE}/subset_v1.txt")
local = Path(RAW) / "_subset.txt"
want = frozen.read_text().split() if frozen.exists() else []
have = local.read_text().split() if local.exists() else []
n_img = sum(1 for p in Path(RAW).rglob("*") if p.suffix.lower() in {".png", ".tif", ".tiff"})
n_json = sum(1 for _ in Path(RAW).rglob("*.json"))
print(f"frozen list: {len(want)} | _subset.txt: {len(have)} | images on Drive: {n_img} | json: {n_json}")
NEED_EXTRACT = not want or sorted(want) != sorted(have) or n_img < len(want)
print("-> extraction needed (section 1b)" if NEED_EXTRACT else "-> subset v1 is complete, skipping the archive")""")

md("""### 1b. Only if section 1 said extraction is needed
Reads the archive through the **Drive shortcut** `MyDrive/SWXD_Data.zip` (Add shortcut to My Drive on
https://drive.google.com/file/d/1Hc9_de5YAdXg46F-GfjcRKMEBRF9XRQi/view). Finished files are skipped, so it resumes.
If Google answers with the quota page (`DownloadBlocked`), wait a few hours: the quota is per file and resets within 24 h.""")
code("""ZIP = "/content/drive/MyDrive/SWXD_Data.zip"
if NEED_EXTRACT:
    if want:  # restore the frozen list first: the subset only grows from _subset.txt
        local.write_text(frozen.read_text())
    !uv run weldsight-remote-subset --zip "$ZIP" --max-images 2000 --seed 0 --include crop_weld_images/ --ann-include crop_weld_jsons/ --prefer-labels 咬边,未熔合,未焊透
    if not want:
        !cp "{RAW}/_subset.txt" "{frozen}"
    have = local.read_text().split()
    assert not want or sorted(want) == sorted(have), "subset differs from subset_v1.txt"
else:
    print("skipped")""")

md("""## 2. Prepare patches (local disk, a few minutes) and check the data version
Expected in `--inspect`: ignored labels 焊缝, 伪缺陷, 焊瘤, 内凹 and **no** `WARNING: labels not in class_names`.""")
code("""!uv run weldsight-prepare --config configs/experiment/swrd_baseline.yaml --inspect
!uv run weldsight-prepare --config configs/experiment/swrd_baseline.yaml
import json
card = json.load(open("data/processed/swrd_v1/dataset_card.json"))
print("data_version", card["data_version"], "| supervision", card["supervision"])
print("train groups per class:", card.get("groups_with_class_per_split", {}).get("train"))
assert card["data_version"] == EXPECTED_DATA_VERSION, (
    f"data_version {card['data_version']} != {EXPECTED_DATA_VERSION}: the subset or the prepare code changed. "
    "Do not train: the pre-registered rare classes were chosen on " + EXPECTED_DATA_VERSION)""")
code("""!uv run pytest -q -x""")

md("""## 3. Train U-Net (ResNet-34 / ImageNet, Dice + Focal), then evaluate on the test split
Resumes the newest unfinished `swrd_baseline_*` run on Drive if there is one. 30 epochs, early stop after 7
without a better `val/mean_iou`. Keep the tab open: free Colab drops idle sessions.""")
code("""import glob, torch
def state(run):
    p = f"{run}/last.pt"
    if not os.path.exists(p):
        return "no-checkpoint"
    st = torch.load(p, map_location="cpu", weights_only=False)
    return f"done@{st['epoch']}" if st.get("done") else f"epoch {st['epoch']}"
runs = sorted(glob.glob(f"{RUNS}/swrd_baseline_*"))
for r in runs:
    print(os.path.basename(r), state(r), "| test metrics" if os.path.exists(f"{r}/metrics_test.json") else "")
todo = [r for r in runs if state(r).startswith("epoch")]
if todo:
    run = todo[-1]
    print("resuming", run)
    !uv run weldsight-train --resume "{run}"
elif runs and state(runs[-1]).startswith("done") and not os.path.exists(f"{runs[-1]}/metrics_test.json"):
    run = runs[-1]
    print("training finished, test evaluation missing:", run)
    !uv run weldsight-eval --run "{run}" --split test
else:
    !uv run weldsight-train --config configs/experiment/swrd_baseline.yaml train.num_workers=2 output_dir="{RUNS}"
    run = sorted(glob.glob(f"{RUNS}/swrd_baseline_*"))[-1]
print("run:", run)""")

md("## 4. Results")
code("""from IPython.display import Image, Markdown, display
import pandas as pd
h = pd.read_csv(f"{run}/history.csv")
display(h[["epoch", "train/loss", "val/total", "val/mean_iou", "val/macro_recall", "sec"]].round(4))
!uv run weldsight-eval --run "{run}" --split val --uncertainty predictive_entropy
display(Markdown(open(f"{run}/results_test.md").read()))
display(Image(f"{run}/referral_test.png"))""")

md("""The whole run folder (config, provenance with git commit + data version + seed, history, metrics,
figures, checkpoints) is on Drive in `MyDrive/weldsight/runs/`. The zip below leaves out the `.pt` files.""")
code("""RUN_NAME = os.path.basename(run)
!cd "{RUNS}" && zip -qr /content/swrd_run.zip "{RUN_NAME}" -x '*.pt' '*.tmp'
from google.colab import files; files.download('/content/swrd_run.zip')""")

nb = nbf.v4.new_notebook(
    cells=c,
    metadata={"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"}},
)
nbf.write(nb, "notebooks/03_swrd_gpu.ipynb")
print("wrote notebooks/03_swrd_gpu.ipynb")
