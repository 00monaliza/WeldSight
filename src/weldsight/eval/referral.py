"""Recall vs. fraction of radiographs referred to a human inspector.

Protocol: the model ranks radiographs by uncertainty; the top fraction f is sent
to an inspector, who is assumed to find every defect on them (an optimistic
upper bound - state it when reporting). On the remaining radiographs only the
model's detections count. The "defect unit" is a positive (patch, class) pair.

    recall(f) = [ sum_{referred} positives + sum_{kept} TP ] / all positives

Reference curves: `random` referral (expected value over permutations) and an
`oracle` that refers radiographs in order of the number of missed defects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate(u: np.ndarray, groups: np.ndarray, how: str = "max") -> pd.Series:
    s = pd.Series(u).groupby(groups)
    if how == "max":
        return s.max()
    if how == "mean":
        return s.mean()
    if how.startswith("top"):
        k = int(how[3:])
        return s.apply(lambda v: np.sort(v.to_numpy())[-k:].mean())
    raise ValueError(how)


def _curve_for_order(order, pos_by_g, tp_by_g, total_pos):
    """order: group labels, most-uncertain first. Returns recall at f=0..1 (G+1 points)."""
    P, TP = pos_by_g.loc[order].to_numpy(), tp_by_g.loc[order].to_numpy()
    zero = np.zeros((1, P.shape[1]))
    ref_pos = np.concatenate([zero, np.cumsum(P, 0)])
    kept_tp = TP.sum(0) - np.concatenate([zero, np.cumsum(TP, 0)])
    return (ref_pos + kept_tp) / np.maximum(total_pos, 1)


def referral_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    groups: np.ndarray,
    agg: str = "max",
    n_random: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a long dataframe: fraction, curve (model/random/oracle), class, recall."""
    y_true, y_pred = y_true.astype(bool), y_pred.astype(bool)
    K = y_true.shape[1]
    # extra column = any class (micro over all positive (patch, class) pairs)
    hit = y_true & y_pred
    pos = np.column_stack([y_true, y_true.sum(1)]).astype(float)
    tp = np.column_stack([hit, hit.sum(1)]).astype(float)
    cols = [f"c{k}" for k in range(K + 1)]
    pos_by_g = pd.DataFrame(pos, columns=cols).groupby(groups).sum()
    tp_by_g = pd.DataFrame(tp, columns=cols).groupby(groups).sum()
    total = pos_by_g.to_numpy().sum(0)
    G = len(pos_by_g)
    frac = np.arange(G + 1) / G

    u_g = aggregate(uncertainty, groups, agg).loc[pos_by_g.index]
    # stable tie-break by name for reproducibility
    order_model = sorted(pos_by_g.index, key=lambda g: (-u_g[g], g))
    missed = (pos_by_g - tp_by_g)[f"c{K}"]
    order_oracle = sorted(pos_by_g.index, key=lambda g: (-missed[g], g))
    rng = np.random.default_rng(seed)
    rand = np.mean(
        [_curve_for_order(list(rng.permutation(pos_by_g.index)), pos_by_g, tp_by_g, total)
         for _ in range(n_random)],
        axis=0,
    )
    curves = {
        "model": _curve_for_order(order_model, pos_by_g, tp_by_g, total),
        "random": rand,
        "oracle": _curve_for_order(order_oracle, pos_by_g, tp_by_g, total),
    }
    rows = []
    for name, c in curves.items():
        for k in range(K + 1):
            for i, f in enumerate(frac):
                rows.append({"fraction": f, "curve": name, "class": k if k < K else "all", "recall": c[i, k]})
    return pd.DataFrame(rows)


def summarize_curve(df: pd.DataFrame, class_names: list[str], at=(0.0, 0.1, 0.2, 0.3)) -> dict:
    """Recall at given referral budgets (step-interpolated) + normalized area under curve."""
    out = {}
    for curve in df["curve"].unique():
        d = df[df["curve"] == curve]
        out[curve] = {}
        for k in list(range(len(class_names))) + ["all"]:
            dk = d[d["class"] == k].sort_values("fraction")
            f, r = dk["fraction"].to_numpy(), dk["recall"].to_numpy()
            name = class_names[k] if k != "all" else "all"
            # largest fraction <= budget (you cannot refer part of a radiograph)
            vals = {f"recall@{int(b * 100)}%": float(r[np.searchsorted(f, b + 1e-9) - 1]) for b in at}
            vals["area"] = float(np.trapezoid(r, f))
            out[curve][name] = vals
    return out


def plot_referral(df: pd.DataFrame, class_names: list[str], path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    styles = {"model": ("#1f77b4", "-"), "random": ("#7f7f7f", "--"), "oracle": ("#2ca02c", ":")}
    for curve, (col, ls) in styles.items():
        d = df[(df["curve"] == curve) & (df["class"] == "all")]
        ax[0].step(d["fraction"], d["recall"], where="post", color=col, ls=ls, label=curve, lw=2)
    ax[0].set_title("All defect classes (micro)")
    palette = ["#d62728", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf"]
    for k, c in enumerate(class_names):
        d = df[(df["curve"] == "model") & (df["class"] == k)]
        ax[1].step(d["fraction"], d["recall"], where="post", color=palette[k % 6], lw=2, label=c)
    ax[1].set_title("Per class (MC Dropout ranking)")
    for a in ax:
        a.set_xlabel("fraction of radiographs referred to inspector")
        a.grid(alpha=0.3)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1.02)
        a.legend(loc="lower right", fontsize=8)
    ax[0].set_ylabel("defect recall (patch-level)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
