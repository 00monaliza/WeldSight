"""End-to-end smoke test: prepare -> train (2 steps) -> MC-Dropout evaluation."""

import json
from pathlib import Path

from weldsight.config import load_config
from weldsight.data.prepare import prepare
from weldsight.eval.evaluate import evaluate_run
from weldsight.train.train import train

REPO = Path(__file__).resolve().parents[1]


def test_train_and_evaluate(fake_riawelc, tmp_path):
    cfg = load_config(REPO / "configs/experiment/debug.yaml")
    cfg["data"].update(raw_dir=str(fake_riawelc), processed_dir=str(tmp_path / "proc"))
    cfg["data"]["split"]["n_trials"] = 200
    cfg["output_dir"] = str(tmp_path / "runs")
    cfg["train"].update(max_steps_per_epoch=2, subset={"train": 16, "val": 8})
    cfg["eval"].update(mc_samples=2, batch_size=16)
    prepare(cfg)
    run = train(cfg)
    for f in ("config.yaml", "provenance.json", "history.csv", "best.pt"):
        assert (run / f).exists(), f
    prov = json.loads((run / "provenance.json").read_text())
    assert prov["seed"] == cfg["seed"] and prov["data_version"]
    m = evaluate_run(run, "test")
    assert (run / "referral_test.png").exists() and (run / "results_test.md").exists()
    assert m["referral_radiograph"]["model"]["all"]["recall@0%"] <= 1.0


def test_resume_matches_uninterrupted_run(fake_riawelc, tmp_path, monkeypatch):
    """A run cut off after epoch 1 and resumed from last.pt must reproduce the
    uninterrupted run (num_workers=0, CPU: every RNG lives in the main process)."""
    import pandas as pd
    import pytest

    import weldsight.train.train as tr

    cfg = load_config(REPO / "configs/experiment/debug.yaml")
    cfg["data"].update(raw_dir=str(fake_riawelc), processed_dir=str(tmp_path / "proc"))
    cfg["data"]["split"]["n_trials"] = 200
    cfg["train"].update(
        epochs=3, patience=99, max_steps_per_epoch=2, subset={"train": 16, "val": 8}
    )
    prepare(cfg)

    cfg["output_dir"] = str(tmp_path / "full")
    full = pd.read_csv(train(cfg) / "history.csv")

    cfg["output_dir"] = str(tmp_path / "cut")
    real_validate, calls = tr.validate, {"n": 0}

    def flaky_validate(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt  # the "disconnect", during epoch 2
        return real_validate(*a, **k)

    monkeypatch.setattr(tr, "validate", flaky_validate)
    with pytest.raises(KeyboardInterrupt):
        train(cfg)
    monkeypatch.setattr(tr, "validate", real_validate)
    run = next((tmp_path / "cut").iterdir())
    assert len(pd.read_csv(run / "history.csv")) == 1
    train(load_config(run / "config.yaml"), resume=run)
    resumed = pd.read_csv(run / "history.csv")
    cols = ["epoch", "train/loss", "val/total"]
    pd.testing.assert_frame_equal(full[cols], resumed[cols])
    prov = json.loads((run / "provenance.json").read_text())
    assert prov["resumed"][0]["after_epoch"] == 1
    # resuming a finished run trains nothing more
    train(load_config(run / "config.yaml"), resume=run)
    assert len(pd.read_csv(run / "history.csv")) == 3
