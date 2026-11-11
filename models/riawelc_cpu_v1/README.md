# riawelc_cpu_v1 — the Stage 1 baseline checkpoint

U-Net (ResNet-18 encoder, no ImageNet weights) trained on RIAWELC with the specimen-level split
(data_version `2c33eedecb1697b9`), seed 42, CPU, best epoch 5 by val macro-F1 (0.892).
Label-only data: only the patch-classification head is trained, so the server runs it in
`patch` mode. Test metrics: README, "Baseline, RIAWELC".

The weights (55 MB) are not in git. They are attached to the GitHub release
`riawelc-cpu-v1` as `riawelc_cpu_v1.pt`.

| file | sha256 |
|---|---|
| riawelc_cpu_v1.pt | `552b97c1e2c6205aa6e8ea2574f1b268072745955b2fe3922e8d2a58e2777f13` |

```bash
uv run weldsight-fetch-model          # downloads to models/riawelc_cpu_v1/best.pt, checks sha256
uv run weldsight-serve --run models/riawelc_cpu_v1 --static frontend/dist
```

The dataset tab additionally needs the prepared data (`weldsight-prepare ... --download`).
