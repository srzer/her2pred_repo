# Prediction statistics

The `predictions/` directory contains model probabilities used to calculate
ROC-AUC, PR-AUC, confidence intervals, and paired statistical tests. Experiment
names match the checkpoint folders on Hugging Face.

- `predictions/external252.csv`: predictions for 252 SVS/KFB cases from
  `ndpi_vanilla`, `ndpi_adversarial`, and `macenko`.
- `predictions/tcga324.csv`: predictions for 324 TCGA cases from
  `internal_vanilla`, `zero_shot`, `resnet`, `plip`, and `few_shot`. The
  `few_shot` column has predictions for the 288 evaluation cases and blank
  values for the 36 training cases.
- `prediction_statistics.json`: metrics, confidence intervals, and paired tests.
- `bootstrap_replicates.csv`: 10,000 paired bootstrap differences used to
  calculate confidence intervals and P values.

Each prediction table includes an anonymous case key, a binary label, and
model probabilities. The SVS/KFB table also includes the slide source.
The TCGA split is defined in
[`tcga_standard_b_few_shot.csv`](../manifests/tcga_standard_b_few_shot.csv).

Run from the repository root:

```bash
python -m pip install -r requirements-analysis.txt
python analysis/reproduce_prediction_statistics.py
python analysis/make_external_roc_pr_figure.py
```

The analysis script updates `prediction_statistics.json`. The figure script
creates `figures/vanilla_vs_adversarial_roc_pr.png`.
