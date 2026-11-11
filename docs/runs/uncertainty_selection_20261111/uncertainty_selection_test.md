### Uncertainty scores on test (21 radiographs, 6 specimens)

| score | referral area | recall@10% | recall@20% | recall@30% | error AUROC |
|---|---|---|---|---|---|
| predictive_entropy | 0.852 | 68.0 | 72.3 | 80.0 | 0.819 |
| expected_entropy **(chosen)** | 0.852 | 68.0 | 73.2 | 80.0 | 0.820 |
| max_prob | 0.837 | 65.6 | 70.5 | 76.1 | 0.804 |
| mutual_information | 0.832 | 65.4 | 71.2 | 76.8 | 0.537 |

random: 0.801, oracle: 0.875
