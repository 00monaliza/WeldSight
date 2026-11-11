"""Compare uncertainty scores from saved predictions, without another MC Dropout pass.

    uv run weldsight-select-uncertainty --run runs/<run_dir> --split val

Reads predictions_<split>.csv (written by weldsight-eval) and, for every `unc_*` column,
computes the radiograph-level referral curve (same settings as weldsight-eval) and the
patch-level error-detection AUROC. The selection rule is pre-registered in
docs/UNCERTAINTY_SELECTION.md: highest referral area on val; a difference < 0.005 is a tie,
broken by error-detection AUROC. Writes uncertainty_selection_<split>.{json,md}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from weldsight.config import load_config
from weldsight.eval.referral import referral_curve, summarize_curve
from weldsight.utils import write_json

TIE = 0.005


def compare_scores(pred: pd.DataFrame, classes: list[str], thr: float, agg: str, seed: int):
    y_true = pred[[f"true_{c}" for c in classes]].to_numpy().astype(np.uint8)
    y_pred = (pred[[f"prob_{c}" for c in classes]].to_numpy() >= thr).astype(np.uint8)
    rad = pred["radiograph_id"].to_numpy()
    err = (y_true != y_pred).any(1)
    rows = []
    random_area = oracle_area = None
    for col in [c for c in pred.columns if c.startswith("unc_")]:
        u = pred[col].to_numpy()
        s = summarize_curve(referral_curve(y_true, y_pred, u, rad, agg=agg, seed=seed), classes)
        random_area, oracle_area = s["random"]["all"]["area"], s["oracle"]["all"]["area"]
        rows.append(
            {
                "score": col.removeprefix("unc_"),
                "referral_area": s["model"]["all"]["area"],
                "recall@10%": s["model"]["all"]["recall@10%"],
                "recall@20%": s["model"]["all"]["recall@20%"],
                "recall@30%": s["model"]["all"]["recall@30%"],
                "error_auroc": float(roc_auc_score(err, u)) if 0 < err.sum() < len(err) else None,
            }
        )
    df = pd.DataFrame(rows).sort_values("referral_area", ascending=False).reset_index(drop=True)
    return df, random_area, oracle_area


def choose(df: pd.DataFrame) -> str:
    best = df["referral_area"].max()
    tied = df[df["referral_area"] >= best - TIE]
    return str(tied.sort_values("error_auroc", ascending=False).iloc[0]["score"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val")
    args = ap.parse_args(argv)
    run = Path(args.run)
    cfg = load_config(run / "config.yaml")
    ec = cfg["eval"]
    pred = pd.read_csv(run / f"predictions_{args.split}.csv")
    classes = [c.removeprefix("true_") for c in pred.columns if c.startswith("true_")]
    df, rnd, orc = compare_scores(
        pred, classes, ec.get("threshold", 0.5), ec.get("radiograph_agg", "max"), cfg["seed"]
    )
    chosen = choose(df)
    out = {
        "split": args.split,
        "n_radiographs": int(pred["radiograph_id"].nunique()),
        "n_groups": int(pred["group_id"].nunique()),
        "pre_registered_default": ec.get("uncertainty"),
        "chosen_by_rule": chosen,
        "random_area": rnd,
        "oracle_area": orc,
        "scores": df.to_dict(orient="records"),
    }
    write_json(out, run / f"uncertainty_selection_{args.split}.json")
    lines = [
        f"### Uncertainty scores on {args.split} ({out['n_radiographs']} radiographs, "
        f"{out['n_groups']} specimens)",
        "",
        "| score | referral area | recall@10% | recall@20% | recall@30% | error AUROC |",
        "|---|---|---|---|---|---|",
    ]
    for r in df.to_dict(orient="records"):
        mark = " **(chosen)**" if r["score"] == chosen else ""
        lines.append(
            f"| {r['score']}{mark} | {r['referral_area']:.3f} | {100 * r['recall@10%']:.1f} | "
            f"{100 * r['recall@20%']:.1f} | {100 * r['recall@30%']:.1f} | {r['error_auroc']:.3f} |"
        )
    lines += ["", f"random: {rnd:.3f}, oracle: {orc:.3f}", ""]
    (run / f"uncertainty_selection_{args.split}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
