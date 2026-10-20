# Clinical covariate analysis

This directory contains cohort summaries and an analysis of the `zero_shot`
model score alongside clinical variables. The cohort includes 481 patients.
The regression uses 376 patients with complete clinical data.

## Results

After adjustment for age, sex, tumor location, histologic subtype,
differentiation, and stage, the pathology score had an OR of 4.44 per SD
(95% CI 3.07–6.43, P = 3.11e-15).

In-sample ROC-AUC was 0.742 for clinical covariates, 0.814 for the pathology
score, and 0.879 for their combination. The combined model improved ROC-AUC
by 0.137 over clinical covariates alone (paired DeLong 95% CI 0.082–0.192,
P = 8.98e-7).

## Files

- `clinicopathological_features_481.csv`: cohort summary.
- `clinical_confounding_regression.csv`: adjusted logistic regression results.
- `clinical_model_comparison.csv`: ROC-AUC comparisons for clinical variables,
  the pathology score, and their combination, evaluated on the regression cohort.
- `analyze_clinical_confounding.py`: generates these tables from patient data.

## Run the analysis

Supply a CSV with one row per patient and these columns:
`her2_status`, `model_score`, `age`, `sex`, `tumor_location_raw`,
`histology_raw`, and `stage`. Use the `zero_shot` model's out-of-fold (OOF)
probability as `model_score`.

Run from the repository root:

```bash
python analysis/clinical_confounding/analyze_clinical_confounding.py \
  --input /path/to/patient_data.csv \
  --out-dir /path/to/analysis_results
```

Use `--help` for accepted values and additional options.
