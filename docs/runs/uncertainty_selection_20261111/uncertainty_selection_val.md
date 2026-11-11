### Uncertainty scores on val (24 radiographs, 6 specimens)

| score | referral area | recall@10% | recall@20% | recall@30% | error AUROC |
|---|---|---|---|---|---|
| max_prob **(chosen)** | 0.961 | 88.8 | 89.4 | 94.3 | 0.843 |
| predictive_entropy | 0.953 | 90.0 | 91.4 | 92.9 | 0.795 |
| expected_entropy | 0.953 | 90.0 | 91.4 | 92.5 | 0.788 |
| mutual_information | 0.934 | 88.1 | 89.0 | 91.4 | 0.731 |

random: 0.932, oracle: 0.973
