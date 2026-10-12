"""Audit a group split before training: how many groups carry each class, and how many of
them land in val/test.

Two steps, so the slow one runs once:

    # 1. read every annotation json of the crops (HTTP Range, parallel; no images) -> csv
    uv run weldsight-split-audit collect --url "$URL" --include 'crop_weld_data/crop_weld_jsons/' \\
        --out data/raw/swrd/_labels.csv
    # 2. report for the split that `weldsight-prepare` would make, plus other seeds and k-fold
    uv run weldsight-split-audit report --labels data/raw/swrd/_labels.csv \\
        --config configs/experiment/swrd_baseline.yaml

Records are annotation files (one per crop image). Groups come from `source_opts.group_regex`,
exactly as in the SWRD adapter. Deduplication needs pixels and is skipped here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from tqdm import tqdm

from weldsight.config import load_config
from weldsight.data.split import SPLITS, stratified_group_split


def to_md(df: pd.DataFrame) -> str:
    """Markdown table without the optional `tabulate` dependency of DataFrame.to_markdown."""
    cols = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
    lines = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, row in df.iterrows():
        lines.append(f"| {idx} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def polygon_area(pts) -> float:
    p = np.asarray(pts, float)
    if p.ndim != 2 or len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)


def parse_labelme(member: str, raw: bytes) -> dict:
    d = json.loads(raw)
    row = {
        "member": member,
        "stem": PurePosixPath(member).stem,
        "image_path": d.get("imagePath"),
        "width": d.get("imageWidth"),
        "height": d.get("imageHeight"),
    }
    n, area = Counter(), Counter()
    for s in d.get("shapes", []):
        lab = str(s.get("label"))
        n[lab] += 1
        area[lab] += polygon_area(s.get("points", []))
    row["shapes"] = json.dumps({k: [n[k], round(area[k], 1)] for k in n}, ensure_ascii=False)
    return row


def collect(
    url: str | None,
    zip_path: str | None,
    include: str,
    out: Path,
    token: str | None = None,
    workers: int = 16,
) -> pd.DataFrame:
    """Read all matching json members into a csv (resumable: rows already in `out` are kept)."""
    from weldsight.data.remote_subset import open_zip

    local = threading.local()

    def zf():
        if not hasattr(local, "zf"):  # one connection per thread: zipfile is not thread-safe
            local.zf = open_zip(url, zip_path, token)
        return local.zf

    members = sorted(
        i.filename
        for i in zf().infolist()
        if not i.is_dir() and i.filename.endswith(".json") and re.search(include, i.filename)
    )
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["member"])
    todo = sorted(set(members) - set(done["member"]))
    print(f"{len(members)} json match {include!r}; {len(todo)} to read")
    rows = []
    if todo:
        with ThreadPoolExecutor(workers) as ex:
            futs = ex.map(lambda m: parse_labelme(m, zf().read(m)), todo)
            rows = list(tqdm(futs, total=len(todo), desc="json"))
    df = pd.concat([done, pd.DataFrame(rows)], ignore_index=True).sort_values("member")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def records_from_labels(df: pd.DataFrame, source_opts: dict, class_names: list[str]):
    """Label csv -> (record table with group_id and presence columns, label matrix)."""
    label_map = source_opts.get("label_map", {})
    group_regex = source_opts.get("group_regex")
    rec = pd.DataFrame({"stem": df["stem"].astype(str)})
    if group_regex:
        rec["group_id"] = [
            (m.group(1) if (m := re.match(group_regex, s)) else s) for s in rec["stem"]
        ]
    else:
        rec["group_id"] = rec["stem"]
    present = np.zeros((len(df), len(class_names)), np.uint8)
    for i, js in enumerate(df["shapes"].fillna("{}")):
        for lab in json.loads(js):
            c = label_map.get(lab, lab)
            if c in class_names:
                present[i, class_names.index(c)] = 1
    for k, c in enumerate(class_names):
        rec[c] = present[:, k].astype(np.int64)  # uint8 would overflow in group sums
    return rec, present


def class_group_table(rec: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Per class: records, groups, effective number of groups, share of the largest group."""
    rows = []
    for c in classes:
        per_g = rec.groupby("group_id")[c].sum()
        per_g = per_g[per_g > 0]
        tot = per_g.sum()
        rows.append(
            {
                "class": c,
                "records": int(tot),
                "groups": len(per_g),
                # (sum n)^2 / sum n^2: groups of equal weight that would give the same spread
                "eff_groups": round(float(tot**2 / (per_g**2).sum()), 1) if tot else 0.0,
                "top_group_share": round(float(per_g.max() / tot), 2) if tot else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("class").astype({"records": int, "groups": int})


def split_table(rec: pd.DataFrame, classes: list[str], split: dict[str, str]) -> pd.DataFrame:
    r = rec.assign(split=rec["group_id"].map(split))
    out = {}
    for s in SPLITS:
        rs = r[r.split == s]
        out[(s, "groups")] = {c: rs.loc[rs[c] > 0, "group_id"].nunique() for c in classes}
        out[(s, "records")] = {c: int(rs[c].sum()) for c in classes}
    t = pd.DataFrame(out)
    t.loc["(all records)"] = [
        r[r.split == s]["group_id"].nunique() if k == "groups" else int((r.split == s).sum())
        for s, k in t.columns
    ]
    return t


def kfold_groups(rec: pd.DataFrame, classes: list[str], k: int, seed: int) -> pd.DataFrame:
    """Greedy class-stratified group k-fold: per fold, how many groups carry each class."""
    g = rec.assign(n=1).groupby("group_id")[classes + ["n"]].sum()
    stats = g.to_numpy(float)
    rng = np.random.default_rng(seed)
    # rarest-first: place groups holding rare classes before common ones
    rarity = (stats[:, :-1] > 0) / np.maximum((stats[:, :-1] > 0).sum(0), 1)
    order = np.lexsort((rng.random(len(g)), -rarity.max(1)))
    filled = np.zeros((k, stats.shape[1]))
    fold = np.zeros(len(g), int)
    tot = np.maximum(stats.sum(0), 1)
    for gi in order:
        # the fold whose shares of this group's classes (and of all records) are lowest
        f = int(np.argmin((filled / tot)[:, stats[gi] > 0].sum(1)))
        fold[gi] = f
        filled[f] += stats[gi]
    has = pd.DataFrame(stats[:, :-1] > 0, index=g.index, columns=classes).assign(fold=fold)
    return has.groupby("fold")[classes].sum().T.rename(columns=lambda f: f"fold{f}")


def report(labels: Path, config: str, seeds: int = 20, k: int = 5) -> str:
    cfg = load_config(config)
    d = cfg["data"]
    classes = d["class_names"]
    df = pd.read_csv(labels)
    rec, present = records_from_labels(df, d.get("source_opts", {}), classes)
    sc = d["split"]
    fr = tuple(sc["fractions"])
    lines = [f"# Split audit: {labels} with {config}", ""]

    sizes = rec.groupby("group_id").size()
    lines += [
        f"{len(rec)} records, {len(sizes)} groups; records per group: "
        f"median {sizes.median():.0f}, max {sizes.max()} ({sizes.idxmax()}), "
        f"top-5 groups hold {sizes.nlargest(5).sum() / len(rec):.0%} of records",
        f"records without any of {classes}: {int((present.sum(1) == 0).sum())}",
        "",
        "## Classes over groups",
        "`eff_groups` = (sum n)^2 / sum n^2: how many equal-sized groups this class is really spread over.",
        "",
        to_md(class_group_table(rec, classes)),
        "",
    ]

    def split_of(seed):
        return stratified_group_split(
            rec["group_id"].tolist(), present, fr, seed=seed, n_trials=sc.get("n_trials", 5000)
        )

    seed0 = sc.get("seed", 0)
    lines += [
        f"## Split used by prepare (seed {seed0}, fractions {fr})",
        "",
        to_md(split_table(rec, classes, split_of(seed0))),
        "",
    ]

    test_groups = []
    for s in range(seeds):
        t = split_table(rec, classes, split_of(s))
        test_groups.append(t[("test", "groups")].drop("(all records)"))
    tg = pd.DataFrame(test_groups)
    lines += [
        f"## Test groups per class over {seeds} split seeds",
        "How much the test set depends on the seed: min / median / max groups with the class.",
        "",
        to_md(tg.agg(["min", "median", "max"]).T),
        "",
        f"## {k}-fold by group: groups with the class in each fold",
        "",
        to_md(kfold_groups(rec, classes, k, seed0)),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="read annotation json from a (remote) zip into a csv")
    src = c.add_mutually_exclusive_group(required=True)
    src.add_argument("--url")
    src.add_argument("--zip")
    c.add_argument("--include", default="crop_weld_data/crop_weld_jsons/")
    c.add_argument("--out", default="data/raw/swrd/_labels.csv")
    c.add_argument("--workers", type=int, default=16)
    c.add_argument("--token-env", default="GDRIVE_TOKEN")
    r = sub.add_parser("report", help="markdown report of the split")
    r.add_argument("--labels", default="data/raw/swrd/_labels.csv")
    r.add_argument("--config", default="configs/experiment/swrd_baseline.yaml")
    r.add_argument("--seeds", type=int, default=20)
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--out", default=None, help="also write the report here")
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        df = collect(a.url, a.zip, a.include, Path(a.out), os.environ.get(a.token_env), a.workers)
        print(f"{len(df)} rows -> {a.out}")
    else:
        text = report(Path(a.labels), a.config, a.seeds, a.k)
        print(text)
        if a.out:
            Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
