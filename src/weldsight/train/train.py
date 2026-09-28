"""Train the baseline U-Net.

    uv run weldsight-train --config configs/experiment/riawelc_baseline.yaml [key=value ...]

Everything needed to reproduce the run is stored in runs/<experiment>_<timestamp>/:
config.yaml (fully resolved), provenance.json (git commit, data version, seed,
library versions, hardware), history.csv, best.pt, and test metrics.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from weldsight.config import load_config, save_config
from weldsight.data.dataset import BasicAugment, WeldPatchDataset, load_manifest
from weldsight.eval.metrics import SegConfusion, detection_metrics
from weldsight.models.losses import build_loss
from weldsight.models.unet import build_model
from weldsight.utils import (
    environment_info,
    git_info,
    make_run_dir,
    seed_everything,
    seed_worker,
    select_device,
    write_json,
)


def make_loaders(cfg: dict, df: pd.DataFrame, classes: list[str]):
    tc = cfg["train"]
    root = cfg["data"]["processed_dir"]
    sub = tc.get("subset") or {}
    train_ds = WeldPatchDataset(
        root, "train", classes, BasicAugment(**tc.get("augment", {})), tc.get("normalize", "per_patch"),
        subset=sub.get("train"), seed=cfg["seed"], df=df,
    )
    val_ds = WeldPatchDataset(
        root, "val", classes, None, tc.get("normalize", "per_patch"),
        subset=sub.get("val"), seed=cfg["seed"], df=df,
    )
    g = torch.Generator().manual_seed(cfg["seed"])
    sampler = None
    if tc.get("balanced_sampler"):
        lab = train_ds.labels()
        key = (lab * (2 ** np.arange(lab.shape[1]))).sum(1)
        freq = pd.Series(key).map(pd.Series(key).value_counts()).to_numpy()
        sampler = WeightedRandomSampler(1.0 / freq, len(train_ds), replacement=True, generator=g)
    common = dict(num_workers=tc["num_workers"], worker_init_fn=seed_worker,
                  persistent_workers=tc["num_workers"] > 0)
    train_dl = DataLoader(train_ds, tc["batch_size"], shuffle=sampler is None, sampler=sampler,
                          drop_last=True, generator=g, **common)
    val_dl = DataLoader(val_ds, cfg["eval"].get("batch_size", 64), shuffle=False, **common)
    return train_dl, val_dl


@torch.no_grad()
def validate(model, loader, loss_fn, device, classes, with_seg, threshold) -> dict:
    model.eval()
    losses, probs, labels = [], [], []
    conf = SegConfusion(len(classes) + 1) if with_seg else None
    for b in loader:
        x = b["image"].to(device)
        seg, cls = model(x, with_seg=with_seg)
        l = loss_fn(seg, cls, b["mask"].to(device), b["labels"].to(device), b["has_mask"].to(device))
        losses.append({k: v.item() for k, v in l.items()})
        if with_seg:
            sp = torch.softmax(seg.float(), 1)
            probs.append(sp[:, 1:].flatten(2).max(-1).values.cpu().numpy())
            hm = b["has_mask"].numpy()
            if hm.any():
                conf.update(sp.argmax(1).cpu().numpy()[hm], b["mask"].numpy()[hm])
        else:
            probs.append(torch.sigmoid(cls.float()).cpu().numpy())
        labels.append(b["labels"].numpy())
    det = detection_metrics(np.concatenate(labels), np.concatenate(probs), threshold, classes)
    out = {f"val/{k}": float(np.mean([d[k] for d in losses])) for k in losses[0]}
    out["val/macro_recall"] = det["_macro"]["recall"]
    out["val/macro_f1"] = det["_macro"]["f1"]
    out["val/macro_auroc"] = det["_macro"]["auroc"]
    for c in classes:
        out[f"val/recall_{c}"] = det[c]["recall"]
    if conf is not None:
        out["val/mean_iou"] = conf.summary(classes)["_mean"]["iou"]
    return out


def train(cfg: dict) -> Path:
    seed_everything(cfg["seed"], cfg["train"].get("deterministic", True))
    device = select_device(cfg["train"].get("device", "auto"))
    df, card = load_manifest(cfg["data"]["processed_dir"])
    classes = card["class_names"]
    with_seg = card["supervision"] == "mask"
    run_dir = make_run_dir(cfg.get("output_dir", "runs"), cfg.get("experiment", "run"))
    save_config(cfg, run_dir / "config.yaml")
    write_json(
        {
            "seed": cfg["seed"],
            "data_version": card["data_version"],
            "dataset": card["name"],
            "git": git_info(),
            "env": environment_info(),
            "device": str(device),
            "with_seg": with_seg,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        run_dir / "provenance.json",
    )
    print(f"[train] run_dir={run_dir} device={device} data_version={card['data_version']} "
          f"supervision={card['supervision']}")

    train_dl, val_dl = make_loaders(cfg, df, classes)
    model = build_model(cfg, len(classes)).to(device)
    loss_fn = build_loss(cfg)
    tc = cfg["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    steps_per_epoch = len(train_dl)
    if tc.get("max_steps_per_epoch"):
        steps_per_epoch = min(steps_per_epoch, tc["max_steps_per_epoch"])
    total = max(1, tc["epochs"] * steps_per_epoch)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))
    )
    use_amp = bool(tc.get("amp")) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    monitor = tc.get("monitor", "val/macro_f1")
    best, bad_epochs, history = -np.inf, 0, []
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for step, b in enumerate(train_dl):
            if step >= steps_per_epoch:
                break
            x = b["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                seg, cls = model(x, with_seg=with_seg)
            l = loss_fn(
                None if seg is None else seg.float(), cls.float(), b["mask"].to(device),
                b["labels"].to(device), b["has_mask"].to(device),
            )
            opt.zero_grad(set_to_none=True)
            scaler.scale(l["total"]).backward()
            if tc.get("grad_clip"):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += l["total"].item()
            n += 1
        val = validate(model, val_dl, loss_fn, device, classes, with_seg, cfg["eval"]["threshold"])
        rec = {"epoch": epoch, "train/loss": run_loss / max(n, 1), "lr": sched.get_last_lr()[0],
               "sec": round(time.time() - t0, 1), **val}
        history.append(rec)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        score = val.get(monitor, float("nan"))
        improved = not np.isnan(score) and score > best
        print(f"[epoch {epoch}] loss={rec['train/loss']:.4f} val_loss={val['val/total']:.4f} "
              f"{monitor}={score:.4f} recalls=" +
              ",".join(f"{c}:{val[f'val/recall_{c}']:.3f}" for c in classes) +
              f" ({rec['sec']}s){' *' if improved else ''}")
        if improved:
            best, bad_epochs = score, 0
            torch.save({"model": model.state_dict(), "classes": classes, "epoch": epoch,
                        "monitor": monitor, "score": score}, run_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= tc.get("patience", 10):
                print(f"[train] early stop at epoch {epoch}")
                break
    if not (run_dir / "best.pt").exists():  # e.g. metric always NaN in smoke tests
        torch.save({"model": model.state_dict(), "classes": classes, "epoch": epoch,
                    "monitor": monitor, "score": float("nan")}, run_dir / "best.pt")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-eval", action="store_true", help="skip test evaluation after training")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    run_dir = train(cfg)
    if not args.no_eval:
        from weldsight.eval.evaluate import evaluate_run

        evaluate_run(run_dir, "test")


if __name__ == "__main__":
    main()
