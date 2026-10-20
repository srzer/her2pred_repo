#!/usr/bin/env python3
"""Generate cohort summaries, adjusted regression, and model comparisons.

The input CSV contains one row per patient. Outputs are aggregate tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import MonteCarloMethod, fisher_exact, norm, random_table, ttest_ind
from sklearn.metrics import roc_auc_score


RNG_SEED = 20260903
TABLE_SEED = 20260905
N_BOOT = 10_000
REQUIRED_COLUMNS = {
    "her2_status", "model_score", "age", "sex", "tumor_location_raw",
    "histology_raw", "stage",
}
RIGHT_SIDED = {"右半结肠", "回盲部", "升结肠", "横结肠", "结肠肝曲"}
LEFT_SIDED = {"左半结肠", "乙状结肠", "降乙交界"}


def clean_text(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    return None if not value or value in {"/", "nan", "None"} else value


def group_location(value):
    value = clean_text(value)
    if value in RIGHT_SIDED:
        return "right-sided colon"
    if value in LEFT_SIDED:
        return "left-sided colon"
    if value in {"直肠", "直乙交界"}:
        return "rectum / rectosigmoid"
    return None


def table_location(value):
    value = clean_text(value)
    if value is None:
        return "Missing"
    grouped = group_location(value)
    return {
        "right-sided colon": "Right-sided colon",
        "left-sided colon": "Left-sided colon",
        "rectum / rectosigmoid": "Rectum/rectosigmoid",
    }.get(grouped, "Other")


def group_histology(value):
    value = clean_text(value)
    if value is None:
        return None
    if any(token in value for token in ("粘液", "黏液", "印戒")):
        return "adenocarcinoma with mucinous or signet-ring component"
    if "腺癌" in value and "杯状细胞" not in value:
        return "conventional adenocarcinoma"
    return None


def table_histology(value):
    if clean_text(value) is None:
        return "Missing"
    grouped = group_histology(value)
    return {
        "conventional adenocarcinoma": "Adenocarcinoma",
        "adenocarcinoma with mucinous or signet-ring component":
            "Adenocarcinoma with mucinous/signet-ring features",
    }.get(grouped, "Other")


def group_differentiation(value):
    value = clean_text(value)
    if value is None:
        return None
    if value.startswith(("中-低分化", "中低分化", "低分化")):
        return "Poor"
    if value.startswith(("高分化", "高-中分化", "中-高分化", "中分化")):
        return "Well/Moderate"
    return None


def normalize_stage(value):
    value = clean_text(value)
    if value is None:
        return None
    value = value.upper()
    if value in {"1", "2", "I", "II", "I-II"}:
        return "I-II"
    if value in {"3", "III"}:
        return "III"
    if value in {"4", "IV"}:
        return "IV"
    return None


def prepare(data):
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    data = data.copy()
    data["her2_status"] = pd.to_numeric(data["her2_status"], errors="raise").astype(int)
    data["model_score"] = pd.to_numeric(data["model_score"], errors="coerce")
    data["age"] = pd.to_numeric(data["age"], errors="coerce")
    data["sex"] = data["sex"].map(clean_text).replace({"女": "female", "男": "male"})
    data["tumor_location"] = data["tumor_location_raw"].map(group_location)
    data["location_table"] = data["tumor_location_raw"].map(table_location)
    data["histology"] = data["histology_raw"].map(group_histology)
    data["histology_table"] = data["histology_raw"].map(table_histology)
    data["differentiation"] = data["histology_raw"].map(group_differentiation)
    data["stage"] = data["stage"].map(normalize_stage)
    if len(data) != 481 or data["her2_status"].value_counts().to_dict() != {0: 364, 1: 117}:
        raise ValueError("final cohort must contain 481 rows: 117 HER2+ and 364 HER2-")
    return data


def fisher_freeman_halton(table):
    table = np.asarray(table, dtype=int)
    try:
        method = MonteCarloMethod(
            n_resamples=1_000_000,
            batch=10_000,
            rng=np.random.default_rng(TABLE_SEED),
        )
        return float(fisher_exact(table, method=method).pvalue)
    except TypeError:
        # SciPy <1.15 has random-table sampling but not fisher_exact(method=...).
        # The two-sided statistic is the fixed-margin table probability.
        row_totals, column_totals = table.sum(axis=1), table.sum(axis=0)
        observed = random_table.pmf(table, row_totals, column_totals)
        rng = np.random.default_rng(TABLE_SEED)
        extreme = 0
        for _ in range(100):
            samples = random_table.rvs(
                row_totals, column_totals, size=10_000, random_state=rng
            )
            probabilities = random_table.pmf(samples, row_totals, column_totals)
            extreme += int((probabilities <= observed * (1 + 1e-12)).sum())
        return (extreme + 1) / 1_000_001


def format_p(value):
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def count_percent(frame, column, category):
    number = int(frame[column].eq(category).sum())
    return f"{number} ({100 * number / len(frame):.1f})"


def build_characteristics(data):
    pos = data.loc[data["her2_status"].eq(1)].copy()
    neg = data.loc[data["her2_status"].eq(0)].copy()
    age_p = float(ttest_ind(pos["age"].dropna(), neg["age"].dropna(), equal_var=False).pvalue)
    sex_p = float(fisher_exact([
        [pos["sex"].eq("female").sum(), pos["sex"].eq("male").sum()],
        [neg["sex"].eq("female").sum(), neg["sex"].eq("male").sum()],
    ]).pvalue)
    location_categories = ["Right-sided colon", "Left-sided colon", "Rectum/rectosigmoid"]
    location_p = fisher_freeman_halton([
        [group["location_table"].eq(category).sum() for category in location_categories]
        for group in (pos, neg)
    ])
    histology_categories = [
        "Adenocarcinoma", "Adenocarcinoma with mucinous/signet-ring features"
    ]
    histology_p = float(fisher_exact([
        [group["histology_table"].eq(category).sum() for category in histology_categories]
        for group in (pos, neg)
    ]).pvalue)
    differentiation_categories = ["Well/Moderate", "Poor"]
    differentiation_p = float(fisher_exact([
        [group["differentiation"].eq(category).sum() for category in differentiation_categories]
        for group in (pos, neg)
    ]).pvalue)
    stage_categories = ["I-II", "III", "IV"]
    stage_p = fisher_freeman_halton([
        [group["stage"].eq(category).sum() for category in stage_categories]
        for group in (pos, neg)
    ])

    def row(label, column, category):
        return (label, count_percent(pos, column, category),
                count_percent(neg, column, category), "")

    rows = [
        ("n", str(len(pos)), str(len(neg)), ""),
        ("Age, mean (SD), years",
         f"{pos['age'].mean():.2f} ({pos['age'].std(ddof=1):.2f})",
         f"{neg['age'].mean():.2f} ({neg['age'].std(ddof=1):.2f})", format_p(age_p)),
        ("Sex, n (%)", "", "", format_p(sex_p)),
        row("  Male", "sex", "male"), row("  Female", "sex", "female"),
        ("Tumor location, n (%)", "", "", format_p(location_p)),
        *[row(f"  {category}", "location_table", category) for category in
          location_categories + ["Other", "Missing"]],
        ("Histology, n (%)", "", "", format_p(histology_p)),
        *[row(f"  {category}", "histology_table", category) for category in
          histology_categories + ["Other", "Missing"]],
        ("Differentiation, n (%)", "", "", format_p(differentiation_p)),
        *[row(f"  {category}", "differentiation", category) for category in
          differentiation_categories],
        ("  Missing", f"{pos['differentiation'].isna().sum()} ({100 * pos['differentiation'].isna().mean():.1f})",
         f"{neg['differentiation'].isna().sum()} ({100 * neg['differentiation'].isna().mean():.1f})", ""),
        ("Stage, n (%)", "", "", format_p(stage_p)),
        *[row(f"  {category}", "stage", category) for category in stage_categories],
        ("  Missing", f"{pos['stage'].isna().sum()} ({100 * pos['stage'].isna().mean():.1f})",
         f"{neg['stage'].isna().sum()} ({100 * neg['stage'].isna().mean():.1f})", ""),
    ]
    return pd.DataFrame(rows, columns=["Characteristic", "HER2+", "HER2-", "P-value"])


def design_matrix(data, include_clinical=True, include_score=False):
    columns = {"Intercept": np.ones(len(data), dtype=float)}
    if include_score:
        columns["model_score_z"] = data["model_score_z"].to_numpy(float)
    if include_clinical:
        columns.update({
            "age (per year)": data["age"].to_numpy(float),
            "sex: male vs female": data["sex"].eq("male").to_numpy(float),
            "tumor_location: left-sided colon vs right-sided colon":
                data["tumor_location"].eq("left-sided colon").to_numpy(float),
            "tumor_location: rectum / rectosigmoid vs right-sided colon":
                data["tumor_location"].eq("rectum / rectosigmoid").to_numpy(float),
            "histology: mucinous/signet-ring component vs conventional adenocarcinoma":
                data["histology"].eq(
                    "adenocarcinoma with mucinous or signet-ring component").to_numpy(float),
            "differentiation: Poor vs Well/Moderate":
                data["differentiation"].eq("Poor").to_numpy(float),
            "stage: III vs I-II": data["stage"].eq("III").to_numpy(float),
            "stage: IV vs I-II": data["stage"].eq("IV").to_numpy(float),
        })
    return pd.DataFrame(columns, index=data.index)


def fit(data, include_clinical=True, include_score=False):
    design = design_matrix(data, include_clinical, include_score)
    result = sm.Logit(data["her2_status"].astype(float), design).fit(disp=False, maxiter=300)
    return result, result.predict(design).to_numpy()


def bootstrap_auc(y, predictions):
    y = np.asarray(y, dtype=int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(RNG_SEED)
    values = np.empty((N_BOOT, len(predictions)))
    for index in range(N_BOOT):
        sample = np.concatenate((rng.choice(pos, len(pos), replace=True),
                                 rng.choice(neg, len(neg), replace=True)))
        for column, prediction in enumerate(predictions):
            values[index, column] = roc_auc_score(y[sample], prediction[sample])
    return values


def delong_components(y, score):
    """AUC and structural components for DeLong inference."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    positive, negative = score[y == 1], score[y == 0]
    comparisons = np.where(
        positive[:, None] > negative[None, :],
        1.0,
        np.where(positive[:, None] == negative[None, :], 0.5, 0.0),
    )
    return float(comparisons.mean()), comparisons.mean(axis=1), comparisons.mean(axis=0)


def delong_ci(y, score):
    auc_value, v10, v01 = delong_components(y, score)
    variance = np.var(v10, ddof=1) / len(v10) + np.var(v01, ddof=1) / len(v01)
    se = np.sqrt(max(float(variance), 0.0))
    z = norm.ppf(0.975)
    return auc_value, max(0.0, auc_value - z * se), min(1.0, auc_value + z * se)


def paired_delong(y, on, off):
    """Compare two correlated ROC-AUCs measured on the same patients."""
    auc_on, on10, on01 = delong_components(y, on)
    auc_off, off10, off01 = delong_components(y, off)
    delta = auc_on - auc_off
    d10, d01 = on10 - off10, on01 - off01
    variance = np.var(d10, ddof=1) / len(d10) + np.var(d01, ddof=1) / len(d01)
    se = np.sqrt(max(float(variance), 0.0))
    p_value = 1.0 if se == 0 and delta == 0 else (
        0.0 if se == 0 else float(2 * norm.sf(abs(delta / se)))
    )
    z = norm.ppf(0.975)
    return {
        "delta": delta,
        "ci_lower": delta - z * se,
        "ci_upper": delta + z * se,
        "p_value_two_sided": p_value,
    }


def analyze_models(data):
    required = ["her2_status", "model_score", "age", "sex", "tumor_location",
                "histology", "differentiation", "stage"]
    complete = data.dropna(subset=required).copy()
    complete["model_score_z"] = (
        (complete["model_score"] - complete["model_score"].mean())
        / complete["model_score"].std(ddof=1)
    )
    clinical_fit, clinical_pred = fit(complete, include_clinical=True)
    score_fit, score_pred = fit(complete, include_clinical=False, include_score=True)
    combined_fit, combined_pred = fit(complete, include_clinical=True, include_score=True)
    predictions = [clinical_pred, score_pred, combined_pred]
    y = complete["her2_status"].to_numpy(int)
    boot = bootstrap_auc(y, predictions)
    delong = [delong_ci(y, prediction) for prediction in predictions]
    aucs = [result[0] for result in delong]
    paired = paired_delong(y, combined_pred, clinical_pred)
    names = ["Model 1: clinical covariates only", "Model 2: pathology score only",
             "Model 3: clinical + pathology score"]
    comparison = []
    difference = boot[:, 2] - boot[:, 0]
    difference_p = min(1.0, 2 * (min(np.sum(difference <= 0), np.sum(difference >= 0)) + 1)
                       / (N_BOOT + 1))
    for index, name in enumerate(names):
        comparison.append({
            "model": name, "n": len(complete), "n_positive": int(y.sum()),
            "n_negative": int((1 - y).sum()), "roc_auc": aucs[index],
            "auc_ci_method": "DeLong",
            "auc_ci_95_lower": delong[index][1],
            "auc_ci_95_upper": delong[index][2],
            "auc_difference_vs_model1": aucs[index] - aucs[0],
            "paired_delong_difference_ci_95_lower":
                paired["ci_lower"] if index == 2 else np.nan,
            "paired_delong_difference_ci_95_upper":
                paired["ci_upper"] if index == 2 else np.nan,
            "paired_delong_p_value_vs_model1":
                paired["p_value_two_sided"] if index == 2 else np.nan,
            "paired_bootstrap_difference_ci_95_lower":
                np.quantile(difference, 0.025) if index == 2 else np.nan,
            "paired_bootstrap_difference_ci_95_upper":
                np.quantile(difference, 0.975) if index == 2 else np.nan,
            "paired_bootstrap_p_value_vs_model1": difference_p if index == 2 else np.nan,
        })
    ci = combined_fit.conf_int()
    references = {
        "sex: male vs female": "female",
        "tumor_location: left-sided colon vs right-sided colon": "right-sided colon",
        "tumor_location: rectum / rectosigmoid vs right-sided colon": "right-sided colon",
        "histology: mucinous/signet-ring component vs conventional adenocarcinoma":
            "conventional adenocarcinoma",
        "differentiation: Poor vs Well/Moderate": "Well/Moderate",
        "stage: III vs I-II": "I-II", "stage: IV vs I-II": "I-II",
    }
    regression = []
    for term in combined_fit.params.index:
        regression.append({
            "model": names[2], "term": term, "reference_category": references.get(term, ""),
            "coefficient": combined_fit.params[term],
            "adjusted_odds_ratio": np.exp(combined_fit.params[term]),
            "ci_95_lower": np.exp(ci.loc[term, 0]), "ci_95_upper": np.exp(ci.loc[term, 1]),
            "p_value": combined_fit.pvalues[term], "n": len(complete),
        })
    return pd.DataFrame(regression), pd.DataFrame(comparison)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with one row per patient")
    parser.add_argument("--out-dir", required=True, help="directory for aggregate CSV outputs")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = prepare(pd.read_csv(args.input))
    characteristics = build_characteristics(data)
    regression, comparison = analyze_models(data)
    characteristics.to_csv(out_dir / "clinicopathological_features_481.csv", index=False)
    regression.to_csv(out_dir / "clinical_confounding_regression.csv", index=False)
    comparison.to_csv(out_dir / "clinical_model_comparison.csv", index=False)
    print(f"wrote aggregate outputs to {out_dir}")


if __name__ == "__main__":
    main()
