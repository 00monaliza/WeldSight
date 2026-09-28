"""Evaluate a trained run on real radiographs with MC Dropout.

    uv run weldsight-eval --run runs/<run_dir> [--split test] [--mc-samples 20]

Writes metrics_<split>.json, results_<split>.md, predictions_<split>.csv,
referral_<split>.csv and referral_<split>.png into the run directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from weldsight.config import load_config
from weldsight.data.dataset import WeldPatchDataset, load_manifest
from weldsight.eval.metrics import cluster_bootstrap_recall, detection_metrics
from weldsight.eval.referral import plot_referral, referral_curve, summarize_curve
from weldsight.eval.uncertainty import mc_predict
from weldsight.models.unet import build_model
from weldsight.utils import seed_everything, select_device, write_json


def evaluate_run(run_dir: str | Path, split: str = "test", mc_samples: int | None = None) -> dict:
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    ec = cfg["eval"]
    T = mc_samples or ec.get("mc_samples", 20)
    seed_everything(cfg["seed"])
    device = select_device(cfg["train"].get("device", "auto"))
    df, card = load_manifest(cfg["data"]["processed_dir"])
    classes = card["class_names"]
    with_seg = card["supervision"] == "mask"

    ds = WeldPatchDataset(cfg["data"]["processed_dir"], split, classes, None,
                          cfg["train"].get("normalize", "per_patch"), df=df)
    dl = DataLoader(ds, ec.get("batch_size", 64), shuffle=False, num_workers=cfg["train"]["num_workers"])
    model = build_model({**cfg, "model": {**cfg["model"], "encoder_weights": None}}, len(classes))
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    res = mc_predict(model, dl, device, T=T, with_seg=with_seg, num_classes=len(classes))
    meta = ds.df.iloc[res["idx"]].reset_index(drop=True)
    y_true = res["labels"].astype(np.uint8)
    src = ec.get("uncertainty_source", "auto")
    src = ("seg" if with_seg else "cls") if src == "auto" else src
    prob = res["seg_prob"] if src == "seg" else res["cls_prob"]
    unc_all = res["seg_unc"] if src == "seg" else res["cls_unc"]
    unc = unc_all[ec.get("uncertainty", "mutual_information")]
    thr = ec.get("threshold", 0.5)
    y_pred = (prob >= thr).astype(np.uint8)
    rad = meta["radiograph_id"].to_numpy()

    det = detection_metrics(y_true, prob, thr, classes)
    ci = cluster_bootstrap_recall(y_true, y_pred, rad, seed=cfg["seed"])
    for k, c in enumerate(classes):
        det[c]["recall_ci95"] = ci[k].tolist()

    # radiograph-level: a radiograph is positive for class k if any of its patches is
    rt = pd.DataFrame(y_true, columns=classes).groupby(rad).max()
    rp = pd.DataFrame(y_pred, columns=classes).groupby(rad).max()
    rad_metrics = {
        c: {"recall": float(((rt[c] == 1) & (rp[c] == 1)).sum() / max((rt[c] == 1).sum(), 1)),
            "support": int(rt[c].sum()), "n_radiographs": len(rt)}
        for c in classes
    }

    # does uncertainty flag errors? AUROC of uncertainty vs. "patch has any error"
    err = (y_true != y_pred).any(1)
    unc_quality = {
        name: float(roc_auc_score(err, u)) if 0 < err.sum() < len(err) else float("nan")
        for name, u in unc_all.items()
    }

    curve = referral_curve(y_true, y_pred, unc, rad, agg=ec.get("radiograph_agg", "max"), seed=cfg["seed"])
    curve_patch = referral_curve(y_true, y_pred, unc, np.arange(len(unc)), seed=cfg["seed"], n_random=20)
    curve.to_csv(run_dir / f"referral_{split}.csv", index=False)
    plot_referral(curve, classes, run_dir / f"referral_{split}.png",
                  title=f"{cfg.get('experiment')} | {split} | MC Dropout T={T}, {ec.get('uncertainty')}")

    metrics = {
        "split": split,
        "mc_samples": T,
        "prediction_source": src,
        "threshold": thr,
        "n_patches": int(len(y_true)),
        "n_radiographs": int(len(np.unique(rad))),
        "n_groups": int(meta["group_id"].nunique()),
        "data_version": card["data_version"],
        "checkpoint_epoch": ckpt.get("epoch"),
        "patch_level": det,
        "radiograph_level": rad_metrics,
        "uncertainty_error_auroc": unc_quality,
        "referral_radiograph": summarize_curve(curve, classes),
        "referral_patch": summarize_curve(curve_patch, classes),
    }
    if res["seg_confusion"] is not None:
        metrics["segmentation"] = res["seg_confusion"].summary(classes)

    pred_df = meta[["patch_id", "radiograph_id", "group_id"]].copy()
    for k, c in enumerate(classes):
        pred_df[f"true_{c}"] = y_true[:, k]
        pred_df[f"prob_{c}"] = prob[:, k]
    for name, u in unc_all.items():
        pred_df[f"unc_{name}"] = u
    pred_df.to_csv(run_dir / f"predictions_{split}.csv", index=False)
    write_json(metrics, run_dir / f"metrics_{split}.json")
    (run_dir / f"results_{split}.md").write_text(results_markdown(metrics, classes))
    print(results_markdown(metrics, classes))
    return metrics


def _f(x, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{100 * x:.1f}" if pct else f"{x:.3f}"


def results_markdown(m: dict, classes: list[str]) -> str:
    seg = m.get("segmentation")
    lines = [
        f"### {m['split']} — {m['n_patches']} patches, {m['n_radiographs']} radiographs, "
        f"{m['n_groups']} specimens (data {m['data_version']}, MC T={m['mc_samples']})",
        "",
        "| class | support | recall % (95% CI, radiograph bootstrap) | precision % | F1 | AUROC | IoU | radiograph recall % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in classes:
        d = m["patch_level"][c]
        lo, hi = d["recall_ci95"]
        iou = seg[c]["iou"] if seg else None
        lines.append(
            f"| {c} | {d['support']} | {_f(d['recall'], 1)} ({_f(lo, 1)}–{_f(hi, 1)}) | "
            f"{_f(d['precision'], 1)} | {_f(d['f1'])} | {_f(d['auroc'])} | {_f(iou)} | "
            f"{_f(m['radiograph_level'][c]['recall'], 1)} |"
        )
    mac = m["patch_level"]["_macro"]
    lines.append(f"| **macro** | | {_f(mac['recall'], 1)} | {_f(mac['precision'], 1)} | {_f(mac['f1'])} | "
                 f"{_f(mac['auroc'])} | {_f(seg['_mean']['iou']) if seg else '–'} | |")
    r = m["referral_radiograph"]
    lines += [
        "",
        "Referral (all classes, recall %): budget → model / random / oracle",
        "",
        "| referred radiographs | model | random | oracle |",
        "|---|---|---|---|",
    ]
    for b in ("0%", "10%", "20%", "30%"):
        k = f"recall@{b}"
        lines.append(f"| {b} | {_f(r['model']['all'][k], 1)} | {_f(r['random']['all'][k], 1)} | "
                     f"{_f(r['oracle']['all'][k], 1)} |")
    lines.append(f"| area under curve | {_f(r['model']['all']['area'])} | {_f(r['random']['all']['area'])} | "
                 f"{_f(r['oracle']['all']['area'])} |")
    lines += ["", "Uncertainty → error detection AUROC: " +
              ", ".join(f"{k}={_f(v)}" for k, v in m["uncertainty_error_auroc"].items()), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--mc-samples", type=int, default=None)
    args = ap.parse_args(argv)
    evaluate_run(args.run, args.split, args.mc_samples)


if __name__ == "__main__":
    main()
