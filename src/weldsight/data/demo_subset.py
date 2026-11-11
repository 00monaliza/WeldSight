"""Build a small demo copy of a prepared dataset for the web UI's dataset browser.

    uv run weldsight-demo-subset --src data/processed/riawelc_v1 --out build/demo --per-group 40

Takes up to `per_group` patches per (split, label) cell, copies them with a manifest of
the same format, and keeps the full dataset's label counts and specimen lists in the card
(`demo_subset`), so the UI shows the real sizes and says that it shows a sample. For
browsing only: never train or evaluate on it.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from weldsight.data.dataset import load_manifest


def label_column(df, classes):
    pos = df[[f"cls_{c}" for c in classes]].to_numpy().astype(bool)
    return ["none" if not r.any() else "multi" if r.sum() > 1 else classes[r.argmax()] for r in pos]


def build(src: Path, out: Path, per_group: int = 40, seed: int = 0) -> Path:
    df, card = load_manifest(src)
    df["label"] = label_column(df, card["class_names"])
    sub = (
        df.sample(frac=1.0, random_state=seed)
        .groupby(["split", "label"])
        .head(per_group)
        .sort_values(["split", "patch_id"])
    )
    dst = out / src.name
    shutil.rmtree(dst, ignore_errors=True)
    for rel in sub["image"]:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, dst / rel)
    sub.drop(columns="label").to_csv(dst / "manifest.csv", index=False)
    card["demo_subset"] = {
        "n_patches": len(sub),
        "of": len(df),
        "per_group": per_group,
        "seed": seed,
        "label_counts": {
            sp: {lab: int(n) for lab, n in g["label"].value_counts().items()}
            for sp, g in df.groupby("split")
        },
        "specimens": {sp: sorted(g["group_id"].unique().tolist()) for sp, g in df.groupby("split")},
    }
    (dst / "dataset_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
    print(f"[demo] {len(sub)} of {len(df)} patches -> {dst}")
    return dst


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", default="data/processed/riawelc_v1")
    ap.add_argument("--out", default="build/demo")
    ap.add_argument("--per-group", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    build(Path(a.src), Path(a.out), a.per_group, a.seed)


if __name__ == "__main__":
    main()
