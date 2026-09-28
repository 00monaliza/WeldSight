"""Builds notebooks/02_train_gpu.ipynb: full run on Kaggle/Colab (T4)."""

import nbformat as nbf

c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
code = lambda s: c.append(nbf.v4.new_code_cell(s))
md("""# WeldSight: baseline training on a single GPU (Kaggle / Colab T4)
Runtime → GPU. On Kaggle enable *Internet*. Takes ~1–1.5 h for RIAWELC with resnet34/ImageNet.""")
code("""!git clone -b claude/weldsight-stage-1-setup-eeev49 https://github.com/00monaliza/WeldSight.git
%cd WeldSight
!pip -q install uv && uv sync --extra dev
!apt-get -qq install -y unrar > /dev/null""")
md(
    "## 1. Data (RIAWELC is downloaded automatically; SWRD/GDXray: put files into `data/raw/<name>`)"
)
code("""!uv run weldsight-prepare --config configs/experiment/riawelc_baseline.yaml --download
!uv run pytest -q""")
md("## 2. Train + evaluate on the real test split (MC Dropout, T=20)")
code(
    """!uv run weldsight-train --config configs/experiment/riawelc_baseline.yaml train.num_workers=4"""
)
code("""import glob, json
from IPython.display import Image, Markdown, display
run = sorted(glob.glob("runs/riawelc_baseline_*"))[-1]
display(Markdown(open(f"{run}/results_test.md").read()))
display(Image(f"{run}/referral_test.png"))""")
md("""## 3. SWRD (segmentation, 6 classes)
Download the Google Drive folder linked from https://github.com/bit628/RapidX-Annotator into
`data/raw/swrd`, then:""")
code("""# !uv run weldsight-prepare --config configs/experiment/swrd_baseline.yaml --inspect
# !uv run weldsight-prepare --config configs/experiment/swrd_baseline.yaml
# !uv run weldsight-train --config configs/experiment/swrd_baseline.yaml""")
md(
    "Keep the whole `runs/<run>` directory: it has config, provenance (git commit, data version, seed), history and metrics."
)
nbf.write(
    nbf.v4.new_notebook(
        cells=c,
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU",
        },
    ),
    "notebooks/02_train_gpu.ipynb",
)
print("written")
