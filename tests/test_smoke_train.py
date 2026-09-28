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
