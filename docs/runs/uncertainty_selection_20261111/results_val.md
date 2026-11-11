### val — 3331 patches, 24 radiographs, 6 specimens (data 2c33eedecb1697b9, MC T=10)

| class | support | recall % (95% CI, radiograph bootstrap) | precision % | F1 | AUROC | IoU | radiograph recall % |
|---|---|---|---|---|---|---|---|
| crack | 1076 | 86.5 (76.5–96.6) | 97.5 | 0.917 | 0.992 | – | 100.0 |
| porosity | 852 | 90.7 (81.0–97.0) | 95.1 | 0.929 | 0.979 | – | 100.0 |
| lack_of_penetration | 535 | 78.9 (47.6–89.7) | 88.1 | 0.832 | 0.956 | – | 62.5 |
| **macro** | | 85.4 | 93.6 | 0.893 | 0.976 | – | |

Referral (all classes, recall %): budget → model / random / oracle

| referred radiographs | model | random | oracle |
|---|---|---|---|
| 0% | 86.3 | 86.3 | 86.3 |
| 10% | 88.1 | 87.5 | 90.7 |
| 20% | 89.0 | 88.7 | 93.6 |
| 30% | 91.4 | 90.3 | 96.5 |
| area under curve | 0.934 | 0.932 | 0.973 |

Uncertainty → error detection AUROC: predictive_entropy=0.795, expected_entropy=0.788, mutual_information=0.731, max_prob=0.843
