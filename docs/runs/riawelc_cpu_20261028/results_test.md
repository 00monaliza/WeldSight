### test — 3073 patches, 21 radiographs, 6 specimens (data 2c33eedecb1697b9, MC T=10)

| class | support | recall % (95% CI, radiograph bootstrap) | precision % | F1 | AUROC | IoU | radiograph recall % |
|---|---|---|---|---|---|---|---|
| crack | 977 | 57.0 (52.6–66.7) | 67.0 | 0.616 | 0.858 | – | 100.0 |
| porosity | 936 | 93.1 (86.8–97.2) | 93.5 | 0.933 | 0.989 | – | 100.0 |
| lack_of_penetration | 658 | 19.0 (2.3–42.8) | 64.4 | 0.293 | 0.718 | – | 50.0 |
| **macro** | | 56.4 | 75.0 | 0.614 | 0.855 | – | |

Referral (all classes, recall %): budget → model / random / oracle

| referred radiographs | model | random | oracle |
|---|---|---|---|
| 0% | 60.4 | 60.4 | 60.4 |
| 10% | 65.4 | 64.1 | 69.0 |
| 20% | 71.2 | 67.9 | 75.9 |
| 30% | 76.8 | 71.5 | 81.5 |
| area under curve | 0.832 | 0.801 | 0.875 |

Uncertainty → error detection AUROC: predictive_entropy=0.819, expected_entropy=0.820, mutual_information=0.537, max_prob=0.804
