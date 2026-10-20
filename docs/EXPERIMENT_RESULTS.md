# Experiment results

Results for VISTA-PATH 0.3 using five-fold probability ensembles.
PR-AUC is computed by trapezoidal integration of the precision-recall curve.

| Experiment | Evaluation cohort | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| `ndpi_vanilla` | SVS/KFB, n=252 | 0.772 | 0.447 |
| `macenko` | SVS/KFB, n=252 | 0.854 | 0.536 |
| `ndpi_adversarial` | SVS/KFB, n=252 | 0.864 | 0.629 |
| `zero_shot` | TCGA, n=324 | 0.742 | 0.379 |
| `resnet` | TCGA, n=324 | 0.485 | 0.196 |
| `plip` | TCGA, n=324 | 0.458 | 0.181 |
| `few_shot` | Held-out TCGA, n=288 | 0.762 | 0.393 |

The `few_shot` experiment uses 36 TCGA cases for adaptation. The split is in
[`tcga_standard_b_few_shot.csv`](../manifests/tcga_standard_b_few_shot.csv).

To recompute metrics, confidence intervals, and paired tests:

```bash
python analysis/reproduce_prediction_statistics.py
```

See [`results/`](../results/README.md) for prediction tables and statistical
outputs, [`final_protocol.json`](../configs/final_protocol.json) for training
settings, and [the reproducibility guide](REPRODUCIBILITY.md) for setup and
commands. Clinical covariate analysis is in
[`analysis/clinical_confounding/`](../analysis/clinical_confounding/).
