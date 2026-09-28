"""Group-aware, class-stratified train/val/test split.

All samples that share a `group_id` (a weld specimen or a source radiograph) end up
in the same split, so neighbouring/overlapping patches of one radiograph can never
leak between train and test. Among many random group assignments we keep the one
whose per-class shares are closest to the target fractions, with a hard penalty
if any class is missing from val or test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def group_class_matrix(
    group_ids: list[str], label_matrix: np.ndarray
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Per group: count of positive samples per class and total sample count."""
    df = pd.DataFrame(label_matrix, columns=[f"c{i}" for i in range(label_matrix.shape[1])])
    df["g"] = group_ids
    agg = df.groupby("g", sort=True)
    pos = agg.sum()
    n = agg.size()
    return list(pos.index), pos.to_numpy(float), n.to_numpy(float)


def stratified_group_split(
    group_ids: list[str],
    label_matrix: np.ndarray,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
    n_trials: int = 5000,
) -> dict[str, str]:
    """Return {group_id: split}. Deterministic for a given seed and input."""
    groups, pos, n = group_class_matrix(group_ids, label_matrix)
    G, K = pos.shape
    if G < 3:
        raise ValueError(f"Need at least 3 groups to split, got {G}")
    frac = np.asarray(fractions, float)
    frac = frac / frac.sum()
    # columns: K classes + "all samples" so background share is also balanced
    stats = np.concatenate([pos, n[:, None]], axis=1)
    totals = stats.sum(0)
    totals[totals == 0] = 1
    rng = np.random.default_rng(seed)
    best, best_score = None, np.inf
    for _ in range(n_trials):
        order = rng.permutation(G)
        assign = np.zeros(G, int)
        filled = np.zeros((3, stats.shape[1]))
        # greedy: put each group where the (sample) share is most below target
        for gi in order:
            deficit = frac - filled[:, -1] / totals[-1]
            s = int(np.argmax(deficit))
            assign[gi] = s
            filled[s] += stats[gi]
        share = filled / totals
        score = np.abs(share - frac[:, None]).sum()
        present = filled[1:, :K] > 0  # val/test must contain every class that exists
        exists = pos.sum(0) > 0
        missing = (~present[:, exists]).sum()
        score += 10.0 * missing
        if np.any(np.bincount(assign, minlength=3) == 0):
            score += 100.0
        if score < best_score:
            best, best_score = assign.copy(), score
    return {g: SPLITS[s] for g, s in zip(groups, best)}


def check_no_group_leak(df: pd.DataFrame, group_col: str = "group_id") -> None:
    per_group = df.groupby(group_col)["split"].nunique()
    leaked = per_group[per_group > 1]
    if len(leaked):
        raise AssertionError(f"Groups present in several splits: {list(leaked.index)[:10]}")
