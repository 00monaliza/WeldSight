import numpy as np
import pandas as pd

from weldsight.eval.select_uncertainty import choose, compare_scores


def _preds(n_rad=20, per=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_rad):
        for i in range(per):
            t = int(rng.random() < 0.4)
            wrong = rng.random() < 0.25
            p = (1 - t if wrong else t) * 0.8 + 0.1
            rows.append(
                {
                    "radiograph_id": f"R{r}",
                    "group_id": f"G{r % 5}",
                    "true_crack": t,
                    "prob_crack": p,
                    # informative: high where the prediction is wrong; useless: noise
                    "unc_good": float(wrong) + rng.random() * 0.1,
                    "unc_noise": rng.random(),
                }
            )
    return pd.DataFrame(rows)


def test_informative_score_wins():
    df, rnd, orc = compare_scores(_preds(), ["crack"], 0.5, "max", 0)
    by = df.set_index("score")
    assert by.loc["good", "error_auroc"] > 0.9 > 0.6 > by.loc["noise", "error_auroc"]
    assert rnd <= by.loc["good", "referral_area"] <= orc + 1e-9
    assert choose(df) == "good"


def test_tie_is_broken_by_error_auroc():
    df = pd.DataFrame(
        [
            {"score": "a", "referral_area": 0.800, "error_auroc": 0.6},
            {"score": "b", "referral_area": 0.797, "error_auroc": 0.8},
            {"score": "c", "referral_area": 0.700, "error_auroc": 0.9},
        ]
    )
    assert choose(df) == "b"
