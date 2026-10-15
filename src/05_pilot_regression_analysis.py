"""Exploratory pilot regression analysis for 95 actigraphy participants.

This is a pilot pipeline only. It does not perform confirmatory inference,
causal interpretation, train/test splitting, machine-learning prediction, raw
data modification, or harmonized dataset merging.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "datasets"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
BACKUP_DIR = PROJECT_ROOT / "outputs" / "backups" / datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

FEATURES_PATH = INTERIM_DIR / "participant_sleep_features_by_window.csv"
PHENOTYPE_PATH = DATASET_DIR / "mesa-sleep-dataset-0.8.0.csv"
DICTIONARY_PATH = DATASET_DIR / "mesa-data-dictionary-0.8.0-variables.csv"
DOMAINS_PATH = DATASET_DIR / "mesa-data-dictionary-0.8.0-domains.csv"

REGRESSION_DICTIONARY_PATH = OUTPUT_TABLE_DIR / "regression_variable_dictionary.csv"
PILOT_DATASET_PATH = PROCESSED_DIR / "pilot_regression_dataset.csv"
SAMPLE_FLOW_PATH = OUTPUT_TABLE_DIR / "pilot_sample_flow.csv"
MISSINGNESS_PATH = OUTPUT_TABLE_DIR / "pilot_missingness_summary.csv"
DESCRIPTIVE_PATH = OUTPUT_TABLE_DIR / "pilot_descriptive_statistics.csv"
PEARSON_PATH = OUTPUT_TABLE_DIR / "pilot_pearson_correlations.csv"
SPEARMAN_PATH = OUTPUT_TABLE_DIR / "pilot_spearman_correlations.csv"
COEFFICIENTS_PATH = OUTPUT_TABLE_DIR / "pilot_regression_coefficients.csv"
MODEL_FIT_PATH = OUTPUT_TABLE_DIR / "pilot_regression_model_fit.csv"
INCREMENTAL_PATH = OUTPUT_TABLE_DIR / "pilot_incremental_value_comparison.csv"
VIF_PATH = OUTPUT_TABLE_DIR / "pilot_vif.csv"
INFLUENTIAL_PATH = OUTPUT_TABLE_DIR / "pilot_influential_observations.csv"
DIAGNOSTICS_PATH = OUTPUT_TABLE_DIR / "pilot_model_diagnostics.csv"
SECONDARY_PATH = OUTPUT_TABLE_DIR / "pilot_secondary_variability_models.csv"
MIDPOINT_OUTLIER_PATH = OUTPUT_TABLE_DIR / "sleep_midpoint_outlier_review.csv"
SENSITIVITY_PATH = OUTPUT_TABLE_DIR / "pilot_sensitivity_analysis.csv"
LOG_PATH = OUTPUT_LOG_DIR / "05_pilot_regression_analysis.log"

REQUESTED_OUTPUTS = [
    REGRESSION_DICTIONARY_PATH,
    PILOT_DATASET_PATH,
    SAMPLE_FLOW_PATH,
    MISSINGNESS_PATH,
    DESCRIPTIVE_PATH,
    PEARSON_PATH,
    SPEARMAN_PATH,
    COEFFICIENTS_PATH,
    MODEL_FIT_PATH,
    INCREMENTAL_PATH,
    VIF_PATH,
    INFLUENTIAL_PATH,
    DIAGNOSTICS_PATH,
    SECONDARY_PATH,
    MIDPOINT_OUTLIER_PATH,
    SENSITIVITY_PATH,
    OUTPUT_FIGURE_DIR / "pilot_model_1_residuals.png",
    OUTPUT_FIGURE_DIR / "pilot_model_2_residuals.png",
    OUTPUT_FIGURE_DIR / "pilot_model_3_residuals.png",
    OUTPUT_FIGURE_DIR / "pilot_model_2_qq.png",
    OUTPUT_FIGURE_DIR / "pilot_model_2_influence.png",
]

CSV_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1")
PRIMARY_FEATURE_SET = "consecutive_7_daybynoon"
SENSITIVITY_FEATURE_SETS = ["first_7_valid_nights", "all_valid_nights"]
PHENOTYPE_COLUMNS = [
    "mesaid",
    "epslpscl5c",
    "sleepage5c",
    "gender1",
    "bmi5c",
    "ahi_a0h3a",
    "avgmainsleep5",
    "avgefficiency5",
    "avgwaso5",
    "avgsleepmidpoint5",
]
REGRESSION_VARIABLES = [
    "ess",
    "age",
    "sex",
    "bmi",
    "ahi",
    "sleep_duration_mean",
    "sleep_duration_sd",
    "sleep_duration_cv",
    "sleep_efficiency_sd",
    "waso_sd",
    "sleep_midpoint_circular_sd",
]
PRIMARY_COMMON_VARIABLES = [
    "ess",
    "age",
    "sex_category",
    "bmi",
    "ahi",
    "sleep_duration_mean",
    "sleep_duration_sd",
    "sleep_duration_cv",
]
MODEL_TERMS = {
    "Model 0": ["age", "C(sex_category)", "bmi", "ahi"],
    "Model 1": ["sleep_duration_mean", "age", "C(sex_category)", "bmi", "ahi"],
    "Model 2": ["sleep_duration_mean", "sleep_duration_sd", "age", "C(sex_category)", "bmi", "ahi"],
    "Model 3": ["sleep_duration_mean", "sleep_duration_cv", "age", "C(sex_category)", "bmi", "ahi"],
}


@dataclass
class OLSResult:
    model: str
    scale: str
    data: pd.DataFrame
    y: np.ndarray
    x: np.ndarray
    term_names: list[str]
    beta: np.ndarray
    residuals: np.ndarray
    fitted: np.ndarray
    hat_diag: np.ndarray
    cov_classic: np.ndarray
    cov_hc3: np.ndarray
    n: int
    p: int
    rank: int
    df_resid: float
    sse: float
    tss: float
    r_squared: float
    adjusted_r_squared: float
    aic: float
    bic: float
    rmse: float


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_csv_with_fallback(path: Path, **kwargs: Any) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logging.info("Wrote %s rows to %s", len(df), path)


def backup_existing_outputs() -> None:
    existing = [path for path in REQUESTED_OUTPUTS if path.exists()]
    if not existing:
        logging.info("No requested pilot outputs existed before this run.")
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for path in existing:
        destination = BACKUP_DIR / path.name
        shutil.copy2(path, destination)
        logging.info("Backed up %s to %s", path, destination)


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = read_csv_with_fallback(FEATURES_PATH)
    phenotype = read_csv_with_fallback(PHENOTYPE_PATH, usecols=PHENOTYPE_COLUMNS)
    dictionary = read_csv_with_fallback(DICTIONARY_PATH)
    domains = read_csv_with_fallback(DOMAINS_PATH)
    dictionary["_source_file"] = relative_path(DICTIONARY_PATH)
    return features, phenotype, dictionary, domains


def domain_coding(domains: pd.DataFrame, domain_id: str) -> list[dict[str, Any]]:
    if not domain_id or pd.isna(domain_id):
        return []
    subset = domains.loc[domains["domain_id"].astype(str).eq(str(domain_id))]
    return [
        {
            "value": str(row.get("value", "")),
            "display_name": str(row.get("display_name", "")),
            "description": str(row.get("description", "")),
        }
        for _, row in subset.iterrows()
    ]


def build_variable_dictionary(
    dictionary: pd.DataFrame,
    domains: pd.DataFrame,
    phenotype: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str], bool]:
    rows: list[dict[str, Any]] = []
    variable_map = {
        "epslpscl5c": "ESS total score",
        "sleepage5c": "age",
        "gender1": "sex/gender",
        "bmi5c": "BMI",
        "ahi_a0h3a": "AHI",
    }
    sex_labels: dict[str, str] = {}
    sex_coding_confirmed = False
    for variable, concept in variable_map.items():
        match = dictionary.loc[dictionary["id"].astype(str).eq(variable)]
        item = match.iloc[0].to_dict() if not match.empty else {}
        domain_id = str(item.get("domain", "") or "")
        coding = domain_coding(domains, domain_id)
        if variable == "gender1" and coding:
            sex_labels = {str(entry["value"]): entry["display_name"] for entry in coding}
            sex_coding_confirmed = True
        values = phenotype[variable] if variable in phenotype else pd.Series(dtype=float)
        numeric = numeric_series(values)
        rows.append(
            {
                "variable": variable,
                "analysis_concept": concept,
                "official_label": item.get("display_name", ""),
                "official_definition": item.get("description", ""),
                "type": item.get("type", ""),
                "units": item.get("units", ""),
                "domain": domain_id,
                "categorical_coding": json.dumps(coding, ensure_ascii=False),
                "categorical_coding_confirmed": bool(coding),
                "observed_non_missing_count": int(values.notna().sum()),
                "observed_unique_value_count": int(values.nunique(dropna=True)),
                "observed_min": float(numeric.min()) if numeric.notna().any() else np.nan,
                "observed_max": float(numeric.max()) if numeric.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows), sex_labels, sex_coding_confirmed


def parse_time_to_minutes(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    parts = str(value).strip().split(":")
    try:
        if len(parts) == 3:
            hour, minute, second = int(parts[0]), int(parts[1]), float(parts[2])
        elif len(parts) == 2:
            hour, minute, second = int(parts[0]), int(parts[1]), 0.0
        else:
            return np.nan
    except ValueError:
        return np.nan
    return (hour % 24) * 60 + minute + second / 60.0


def circular_diff_minutes(a: pd.Series, b: pd.Series) -> pd.Series:
    diff = numeric_series(a) - numeric_series(b)
    return ((diff + 720) % 1440) - 720


def flag_iqr_outlier(series: pd.Series) -> pd.Series:
    numeric = numeric_series(series)
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=series.index)
    return numeric.gt(q3 + 3 * iqr) | numeric.lt(q1 - 3 * iqr)


def prepare_analysis_data(
    features: pd.DataFrame,
    phenotype: pd.DataFrame,
    sex_labels: dict[str, str],
    sex_coding_confirmed: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = features.copy()
    phenotype = phenotype.copy()
    features["mesaid"] = features["mesaid"].astype(str)
    phenotype["mesaid"] = phenotype["mesaid"].astype(str)
    primary = features.loc[features["feature_set"].eq(PRIMARY_FEATURE_SET)].copy()
    phenotype_dups = phenotype["mesaid"].duplicated(keep=False)
    feature_dups = primary["mesaid"].duplicated(keep=False)
    merged = primary.merge(phenotype, on="mesaid", how="left", validate="one_to_one")
    merged["unmatched_phenotype"] = merged["epslpscl5c"].isna()
    merged = merged.rename(
        columns={
            "epslpscl5c": "ess",
            "sleepage5c": "age",
            "gender1": "sex",
            "bmi5c": "bmi",
            "ahi_a0h3a": "ahi",
        }
    )
    for column in [
        "ess",
        "age",
        "sex",
        "bmi",
        "ahi",
        "avgmainsleep5",
        "avgefficiency5",
        "avgwaso5",
    ]:
        merged[column] = numeric_series(merged[column])
    merged["ahi_log1p"] = np.log1p(merged["ahi"])
    merged["official_sleep_midpoint_minutes"] = merged["avgsleepmidpoint5"].apply(parse_time_to_minutes)
    merged["midpoint_diff_minutes"] = circular_diff_minutes(
        merged["sleep_midpoint_circular_mean"],
        merged["official_sleep_midpoint_minutes"],
    )
    if sex_coding_confirmed:
        sex_codes = merged["sex"].dropna().astype(int).astype(str)
        merged["sex_category"] = sex_codes.map(sex_labels)
        merged["sex_category"] = merged["sex_category"].where(
            merged["sex_category"].notna(),
            merged["sex"].astype("string"),
        )
    else:
        merged["sex_category"] = merged["sex"].astype("string")
    merged["sleep_duration_cv_outlier"] = flag_iqr_outlier(merged["sleep_duration_cv"])
    merged["sleep_midpoint_circular_sd_outlier"] = flag_iqr_outlier(
        merged["sleep_midpoint_circular_sd"]
    )
    complete_mask = merged[PRIMARY_COMMON_VARIABLES].notna().all(axis=1)
    merged["primary_common_complete_case"] = complete_mask
    sample_rows = [
        {"step": "primary_feature_rows", "n": int(len(primary)), "notes": "feature_set = consecutive_7_daybynoon"},
        {"step": "duplicate_mesaid_in_primary_features", "n": int(feature_dups.sum()), "notes": "Rows with duplicate mesaid before merge."},
        {"step": "main_phenotype_rows", "n": int(len(phenotype)), "notes": "mesa-sleep-dataset-0.8.0.csv"},
        {"step": "duplicate_mesaid_in_main_phenotype", "n": int(phenotype_dups.sum()), "notes": "Rows with duplicate mesaid before merge."},
        {"step": "merged_rows", "n": int(len(merged)), "notes": "Primary features joined to main phenotype by mesaid."},
        {"step": "unmatched_mesaid", "n": int(merged["unmatched_phenotype"].sum()), "notes": "Feature participants not found in main phenotype."},
        {"step": "primary_model_common_complete_case", "n": int(complete_mask.sum()), "notes": "Common sample for Models 0-3."},
        {"step": "excluded_from_primary_common_complete_case", "n": int((~complete_mask).sum()), "notes": "Any missing variable required by Models 0-3."},
    ]
    for variable in PRIMARY_COMMON_VARIABLES:
        sample_rows.append(
            {
                "step": f"excluded_reason_missing_{variable}",
                "n": int(merged[variable].isna().sum()),
                "notes": "Counts are not mutually exclusive.",
            }
        )
    return merged, pd.DataFrame(sample_rows), build_missingness_summary(merged)


def build_missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable in REGRESSION_VARIABLES + ["ahi_log1p"]:
        values = df[variable] if variable in df else pd.Series(dtype=float)
        numeric = numeric_series(values)
        notes: list[str] = []
        if variable == "ess":
            notes.append(f"invalid_ess_range_count={int(numeric.lt(0).sum() + numeric.gt(24).sum())}")
        if variable == "sleep_duration_cv":
            notes.append(f"iqr_outlier_count={int(df['sleep_duration_cv_outlier'].sum())}")
        if variable == "sleep_midpoint_circular_sd":
            notes.append(f"iqr_outlier_count={int(df['sleep_midpoint_circular_sd_outlier'].sum())}")
        rows.append(
            {
                "variable": variable,
                "missing_count": int(values.isna().sum()),
                "missing_rate": float(values.isna().mean()) if len(values) else np.nan,
                "infinite_count": int(np.isinf(numeric).sum()),
                "min": float(numeric.min()) if numeric.notna().any() else np.nan,
                "max": float(numeric.max()) if numeric.notna().any() else np.nan,
                "notes": "; ".join(notes),
            }
        )
    return pd.DataFrame(rows)


def continuous_summary(sample: str, variable: str, label: str, series: pd.Series) -> dict[str, Any]:
    values = numeric_series(series).dropna()
    return {
        "sample": sample,
        "variable": variable,
        "label": label,
        "variable_type": "continuous",
        "category": "",
        "n": int(values.count()),
        "percent": np.nan,
        "mean": float(values.mean()) if not values.empty else np.nan,
        "sd": float(values.std(ddof=1)) if len(values) >= 2 else np.nan,
        "median": float(values.median()) if not values.empty else np.nan,
        "q1": float(values.quantile(0.25)) if not values.empty else np.nan,
        "q3": float(values.quantile(0.75)) if not values.empty else np.nan,
        "min": float(values.min()) if not values.empty else np.nan,
        "max": float(values.max()) if not values.empty else np.nan,
    }


def categorical_summaries(sample: str, variable: str, label: str, series: pd.Series) -> list[dict[str, Any]]:
    counts = series.fillna("<missing>").astype(str).value_counts(dropna=False).sort_index()
    total = int(counts.sum())
    rows: list[dict[str, Any]] = []
    for category, count in counts.items():
        rows.append(
            {
                "sample": sample,
                "variable": variable,
                "label": label,
                "variable_type": "categorical",
                "category": category,
                "n": int(count),
                "percent": float(count / total * 100) if total else np.nan,
                "mean": np.nan,
                "sd": np.nan,
                "median": np.nan,
                "q1": np.nan,
                "q3": np.nan,
                "min": np.nan,
                "max": np.nan,
            }
        )
    return rows


def build_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "ess": "ESS",
        "age": "age",
        "sex_category": "sex",
        "bmi": "BMI",
        "ahi": "AHI",
        "ahi_log1p": "log1p(AHI)",
        "sleep_duration_mean": "sleep duration mean",
        "sleep_duration_sd": "sleep duration SD",
        "sleep_duration_cv": "sleep duration CV",
        "sleep_efficiency_sd": "sleep efficiency SD",
        "waso_sd": "WASO SD",
        "sleep_midpoint_circular_sd": "sleep midpoint circular SD",
    }
    continuous = [
        "ess",
        "age",
        "bmi",
        "ahi",
        "ahi_log1p",
        "sleep_duration_mean",
        "sleep_duration_sd",
        "sleep_duration_cv",
        "sleep_efficiency_sd",
        "waso_sd",
        "sleep_midpoint_circular_sd",
    ]
    rows: list[dict[str, Any]] = []
    for sample, sample_df in {
        "merged": df,
        "primary_common_complete_case": df.loc[df["primary_common_complete_case"]],
    }.items():
        for variable in continuous:
            rows.append(continuous_summary(sample, variable, labels[variable], sample_df[variable]))
        rows.extend(categorical_summaries(sample, "sex_category", labels["sex_category"], sample_df["sex_category"]))
    return pd.DataFrame(rows)


def build_correlations(df: pd.DataFrame, method: str) -> pd.DataFrame:
    corr_vars = [
        "ess",
        "age",
        "sex",
        "bmi",
        "ahi",
        "sleep_duration_mean",
        "sleep_duration_sd",
        "sleep_duration_cv",
        "sleep_efficiency_sd",
        "waso_sd",
        "sleep_midpoint_circular_sd",
    ]
    sample = df.loc[df["primary_common_complete_case"], corr_vars].apply(pd.to_numeric, errors="coerce")
    corr = sample.corr(method=method)
    corr.insert(0, "variable", corr.index)
    return corr.reset_index(drop=True)


def add_standardized_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        values = numeric_series(df[column])
        sd = values.std(ddof=1)
        df[f"{column}_z"] = (values - values.mean()) / sd if pd.notna(sd) and sd > 0 else np.nan
    return df


def resolve_terms(terms: list[str], standardized: bool = False, ahi_variable: str = "ahi") -> list[str]:
    resolved: list[str] = []
    for term in terms:
        if term == "C(sex_category)":
            resolved.append(term)
        elif term == "ahi":
            resolved.append("ahi_log1p" if ahi_variable == "ahi_log1p" else ("ahi_z" if standardized else "ahi"))
        else:
            resolved.append(f"{term}_z" if standardized else term)
    return resolved


def design_matrix(data: pd.DataFrame, terms: list[str]) -> tuple[np.ndarray, list[str]]:
    columns = [pd.Series(1.0, index=data.index, name="Intercept")]
    names = ["Intercept"]
    for term in terms:
        if term == "C(sex_category)":
            dummies = pd.get_dummies(data["sex_category"], prefix="sex", drop_first=True, dtype=float)
            for dummy in dummies.columns:
                columns.append(dummies[dummy].rename(dummy))
                names.append(dummy)
        else:
            columns.append(numeric_series(data[term]).astype(float).rename(term))
            names.append(term)
    xdf = pd.concat(columns, axis=1)
    return xdf.to_numpy(dtype=float), names


def ols_fit(model: str, scale: str, data: pd.DataFrame, terms: list[str]) -> OLSResult:
    y = numeric_series(data["ess"]).to_numpy(dtype=float)
    x, term_names = design_matrix(data, terms)
    beta = np.linalg.pinv(x) @ y
    fitted = x @ beta
    residuals = y - fitted
    n = int(len(y))
    rank = int(np.linalg.matrix_rank(x))
    p = int(x.shape[1])
    df_resid = float(n - rank)
    sse = float(np.sum(residuals**2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - sse / tss if tss > 0 else np.nan
    adjusted = 1 - (1 - r_squared) * (n - 1) / df_resid if df_resid > 0 else np.nan
    xtx_inv = np.linalg.pinv(x.T @ x)
    hat_diag = np.sum((x @ xtx_inv) * x, axis=1)
    mse = sse / df_resid if df_resid > 0 else np.nan
    cov_classic = mse * xtx_inv
    hc3_scale = (residuals / np.clip(1 - hat_diag, 1e-12, None)) ** 2
    cov_hc3 = xtx_inv @ (x.T @ (hc3_scale[:, None] * x)) @ xtx_inv
    rmse = float(np.sqrt(np.mean(residuals**2)))
    log_likelihood = -n / 2 * (math.log(2 * math.pi) + 1 + math.log(sse / n)) if sse > 0 else np.nan
    aic = float(-2 * log_likelihood + 2 * p) if pd.notna(log_likelihood) else np.nan
    bic = float(-2 * log_likelihood + math.log(n) * p) if pd.notna(log_likelihood) else np.nan
    return OLSResult(
        model=model,
        scale=scale,
        data=data.copy(),
        y=y,
        x=x,
        term_names=term_names,
        beta=beta,
        residuals=residuals,
        fitted=fitted,
        hat_diag=hat_diag,
        cov_classic=cov_classic,
        cov_hc3=cov_hc3,
        n=n,
        p=p,
        rank=rank,
        df_resid=df_resid,
        sse=sse,
        tss=tss,
        r_squared=float(r_squared),
        adjusted_r_squared=float(adjusted),
        aic=aic,
        bic=bic,
        rmse=rmse,
    )


def coefficient_table(result: OLSResult, covariance: str = "HC3") -> pd.DataFrame:
    cov = result.cov_hc3 if covariance == "HC3" else result.cov_classic
    se = np.sqrt(np.diag(cov))
    t_values = result.beta / se
    p_values = 2 * stats.t.sf(np.abs(t_values), result.df_resid)
    critical = stats.t.ppf(0.975, result.df_resid)
    rows: list[dict[str, Any]] = []
    for index, term in enumerate(result.term_names):
        rows.append(
            {
                "model": result.model,
                "scale": result.scale,
                "term": term,
                "n": result.n,
                "coefficient": float(result.beta[index]),
                "robust_se": float(se[index]) if covariance == "HC3" else np.nan,
                "classical_se": float(se[index]) if covariance == "classical" else np.nan,
                "ci_lower": float(result.beta[index] - critical * se[index]),
                "ci_upper": float(result.beta[index] + critical * se[index]),
                "p_value": float(p_values[index]),
                "r_squared": result.r_squared,
                "adjusted_r_squared": result.adjusted_r_squared,
                "aic": result.aic,
                "bic": result.bic,
                "rmse": result.rmse,
                "residual_df": result.df_resid,
            }
        )
    return pd.DataFrame(rows)


def model_fit_row(result: OLSResult) -> dict[str, Any]:
    return {
        "model": result.model,
        "scale": result.scale,
        "n": result.n,
        "r_squared": result.r_squared,
        "adjusted_r_squared": result.adjusted_r_squared,
        "aic": result.aic,
        "bic": result.bic,
        "rmse": result.rmse,
        "residual_df": result.df_resid,
        "parameter_count": result.p,
        "observations_per_parameter": result.n / result.p if result.p else np.nan,
        "pilot_overfit_warning": "Small pilot sample; interpret effect sizes, CIs, and incremental value conservatively.",
    }


def fit_primary_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, OLSResult], pd.DataFrame]:
    common = df.loc[df["primary_common_complete_case"]].copy()
    common = add_standardized_columns(
        common,
        ["sleep_duration_mean", "sleep_duration_sd", "sleep_duration_cv", "age", "bmi", "ahi"],
    )
    coefficient_frames: list[pd.DataFrame] = []
    fit_rows: list[dict[str, Any]] = []
    fitted: dict[str, OLSResult] = {}
    for model_label, terms in MODEL_TERMS.items():
        raw = ols_fit(model_label, "raw_units", common, resolve_terms(terms, standardized=False))
        coefficient_frames.append(coefficient_table(raw, "HC3"))
        fit_rows.append(model_fit_row(raw))
        fitted[model_label] = raw
        standardized = ols_fit(
            model_label,
            "standardized_predictors",
            common,
            resolve_terms(terms, standardized=True),
        )
        coefficient_frames.append(coefficient_table(standardized, "HC3"))
        fit_rows.append(model_fit_row(standardized))
    return pd.concat(coefficient_frames, ignore_index=True), pd.DataFrame(fit_rows), fitted, common


def partial_f_test(reduced: OLSResult, full: OLSResult) -> tuple[float, float, float]:
    df_diff = reduced.df_resid - full.df_resid
    if df_diff <= 0 or full.df_resid <= 0:
        return np.nan, np.nan, df_diff
    f_stat = ((reduced.sse - full.sse) / df_diff) / (full.sse / full.df_resid)
    p_value = stats.f.sf(f_stat, df_diff, full.df_resid)
    return float(f_stat), float(p_value), float(df_diff)


def build_incremental_comparison(fitted: dict[str, OLSResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = fitted["Model 1"]
    for label in ["Model 2", "Model 3"]:
        full = fitted[label]
        f_stat, p_value, df_diff = partial_f_test(base, full)
        rows.append(
            {
                "comparison": f"Model 1 vs {label}",
                "n": full.n,
                "delta_r_squared": full.r_squared - base.r_squared,
                "delta_adjusted_r_squared": full.adjusted_r_squared - base.adjusted_r_squared,
                "delta_aic": full.aic - base.aic,
                "delta_bic": full.bic - base.bic,
                "partial_f_statistic": f_stat,
                "partial_f_p_value": p_value,
                "partial_f_df_diff": df_diff,
                "interpretation_note": "Partial F-test uses classical OLS assumptions and is not based on HC3 robust covariance.",
            }
        )
    return pd.DataFrame(rows)


def r_squared_from_arrays(y: np.ndarray, x: np.ndarray) -> float:
    beta = np.linalg.pinv(x) @ y
    resid = y - x @ beta
    sse = np.sum(resid**2)
    tss = np.sum((y - y.mean()) ** 2)
    return float(1 - sse / tss) if tss > 0 else np.nan


def build_vif(fitted: dict[str, OLSResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_label, result in fitted.items():
        for idx, term in enumerate(result.term_names):
            if term == "Intercept":
                continue
            y = result.x[:, idx]
            others = np.delete(result.x, idx, axis=1)
            r2 = r_squared_from_arrays(y, others)
            vif = 1 / (1 - r2) if pd.notna(r2) and r2 < 1 else np.inf
            rows.append({"model": model_label, "term": term, "vif": float(vif)})
    return pd.DataFrame(rows)


def breusch_pagan(result: OLSResult) -> tuple[float, float, float, float]:
    y = result.residuals**2
    r2 = r_squared_from_arrays(y, result.x)
    lm = result.n * r2
    df = result.rank - 1
    lm_p = stats.chi2.sf(lm, df) if df > 0 else np.nan
    f_stat = (r2 / df) / ((1 - r2) / (result.n - df - 1)) if df > 0 and result.n > df + 1 else np.nan
    f_p = stats.f.sf(f_stat, df, result.n - df - 1) if pd.notna(f_stat) else np.nan
    return float(lm), float(lm_p), float(f_stat), float(f_p)


def influence_frame(result: OLSResult) -> pd.DataFrame:
    mse = result.sse / result.df_resid
    denom = np.sqrt(mse * np.clip(1 - result.hat_diag, 1e-12, None))
    studentized = result.residuals / denom
    cooks = (result.residuals**2 / (result.p * mse)) * (
        result.hat_diag / np.clip((1 - result.hat_diag) ** 2, 1e-12, None)
    )
    return pd.DataFrame(
        {
            "hat_diag": result.hat_diag,
            "student_resid": studentized,
            "cooks_d": cooks,
        }
    )


def build_diagnostics_and_influence(fitted: dict[str, OLSResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostic_rows: list[dict[str, Any]] = []
    influence_rows: list[dict[str, Any]] = []
    for model_label, result in fitted.items():
        frame = influence_frame(result)
        cooks_threshold = 4 / result.n
        leverage_threshold = 2 * result.p / result.n
        studentized_threshold = 3
        flagged = (
            frame["cooks_d"].gt(cooks_threshold)
            | frame["student_resid"].abs().gt(studentized_threshold)
            | frame["hat_diag"].gt(leverage_threshold)
        )
        bp_lm, bp_lm_p, bp_f, bp_f_p = breusch_pagan(result)
        shapiro_stat, shapiro_p = stats.shapiro(result.residuals) if result.n <= 5000 else (np.nan, np.nan)
        diagnostic_rows.append(
            {
                "model": model_label,
                "n": result.n,
                "parameter_count": result.p,
                "residual_df": result.df_resid,
                "cooks_distance_threshold": cooks_threshold,
                "leverage_threshold": leverage_threshold,
                "studentized_residual_threshold": studentized_threshold,
                "max_cooks_distance": float(frame["cooks_d"].max()),
                "max_abs_studentized_residual": float(frame["student_resid"].abs().max()),
                "max_leverage": float(frame["hat_diag"].max()),
                "influential_observation_count": int(flagged.sum()),
                "breusch_pagan_lm": bp_lm,
                "breusch_pagan_lm_p_value": bp_lm_p,
                "breusch_pagan_f": bp_f,
                "breusch_pagan_f_p_value": bp_f_p,
                "shapiro_wilk_statistic": float(shapiro_stat),
                "shapiro_wilk_p_value": float(shapiro_p),
            }
        )
        data = result.data.reset_index(drop=True)
        for idx in np.flatnonzero(flagged.to_numpy()):
            influence_rows.append(
                {
                    "model": model_label,
                    "mesaid": data.iloc[idx]["mesaid"],
                    "cooks_distance": float(frame.iloc[idx]["cooks_d"]),
                    "studentized_residual": float(frame.iloc[idx]["student_resid"]),
                    "leverage": float(frame.iloc[idx]["hat_diag"]),
                    "flag_cooks_gt_4_over_n": bool(frame.iloc[idx]["cooks_d"] > cooks_threshold),
                    "flag_abs_studentized_residual_gt_3": bool(abs(frame.iloc[idx]["student_resid"]) > studentized_threshold),
                    "flag_leverage_gt_2p_over_n": bool(frame.iloc[idx]["hat_diag"] > leverage_threshold),
                    "ess": data.iloc[idx]["ess"],
                    "sleep_duration_mean": data.iloc[idx].get("sleep_duration_mean", np.nan),
                    "sleep_duration_sd": data.iloc[idx].get("sleep_duration_sd", np.nan),
                    "sleep_duration_cv": data.iloc[idx].get("sleep_duration_cv", np.nan),
                }
            )
    return pd.DataFrame(diagnostic_rows), pd.DataFrame(influence_rows)


def save_residual_plot(model_label: str, result: OLSResult, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(result.fitted, result.residuals, alpha=0.75)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title(f"{model_label}: residuals vs fitted")
    axes[1].hist(result.residuals, bins=14, edgecolor="black", alpha=0.8)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{model_label}: residual distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_qq_plot(model_label: str, result: OLSResult, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    stats.probplot(result.residuals, dist="norm", plot=ax)
    ax.set_title(f"{model_label}: Q-Q plot")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_influence_plot(model_label: str, result: OLSResult, output_path: Path) -> None:
    frame = influence_frame(result)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(frame["hat_diag"], frame["student_resid"], alpha=0.75)
    axes[0].axhline(3, color="red", linestyle="--", linewidth=1)
    axes[0].axhline(-3, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Leverage")
    axes[0].set_ylabel("Studentized residual")
    axes[0].set_title(f"{model_label}: leverage vs residual")
    axes[1].scatter(np.arange(len(frame)), frame["cooks_d"], alpha=0.75)
    axes[1].axhline(4 / len(frame), color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Observation index")
    axes[1].set_ylabel("Cook's distance")
    axes[1].set_title(f"{model_label}: Cook's distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_diagnostic_figures(fitted: dict[str, OLSResult]) -> None:
    save_residual_plot("Model 1", fitted["Model 1"], OUTPUT_FIGURE_DIR / "pilot_model_1_residuals.png")
    save_residual_plot("Model 2", fitted["Model 2"], OUTPUT_FIGURE_DIR / "pilot_model_2_residuals.png")
    save_residual_plot("Model 3", fitted["Model 3"], OUTPUT_FIGURE_DIR / "pilot_model_3_residuals.png")
    save_qq_plot("Model 2", fitted["Model 2"], OUTPUT_FIGURE_DIR / "pilot_model_2_qq.png")
    save_influence_plot("Model 2", fitted["Model 2"], OUTPUT_FIGURE_DIR / "pilot_model_2_influence.png")


def bh_fdr(pvalues: list[float]) -> list[float]:
    p = np.array(pvalues, dtype=float)
    output = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return output.tolist()
    valid_p = p[valid]
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    temp = np.empty_like(adjusted)
    temp[order] = adjusted
    output[valid] = temp
    return output.tolist()


def build_midpoint_outlier_review(df: pd.DataFrame) -> pd.DataFrame:
    threshold = numeric_series(df["sleep_midpoint_circular_sd"]).quantile(0.95)
    rows = df.copy()
    rows["flag_official_derived_midpoint_diff_ge_60min"] = rows["midpoint_diff_minutes"].abs().ge(60)
    rows["flag_midpoint_sd_top_5_percent"] = numeric_series(rows["sleep_midpoint_circular_sd"]).ge(threshold)
    rows["flag_midnight_boundary_near"] = (
        numeric_series(rows["sleep_midpoint_circular_mean"]).le(60)
        | numeric_series(rows["sleep_midpoint_circular_mean"]).ge(1380)
    )
    rows["flag_midnight_boundary_abnormal"] = (
        rows["flag_midnight_boundary_near"]
        & rows["flag_official_derived_midpoint_diff_ge_60min"]
    )
    return rows[
        [
            "mesaid",
            "feature_set",
            "sleep_midpoint_circular_mean",
            "official_sleep_midpoint_minutes",
            "midpoint_diff_minutes",
            "sleep_midpoint_circular_sd",
            "flag_official_derived_midpoint_diff_ge_60min",
            "flag_midpoint_sd_top_5_percent",
            "flag_midnight_boundary_near",
            "flag_midnight_boundary_abnormal",
        ]
    ]


def build_secondary_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    midpoint_review = build_midpoint_outlier_review(df)
    frames: list[pd.DataFrame] = []
    variability_pvalues: list[float] = []
    variability_indices: list[int] = []
    for term in ["sleep_efficiency_sd", "waso_sd", "sleep_midpoint_circular_sd"]:
        required = ["ess", "sleep_duration_mean", term, "age", "sex_category", "bmi", "ahi"]
        sample = df.dropna(subset=required).copy()
        terms = ["sleep_duration_mean", term, "age", "C(sex_category)", "bmi", "ahi"]
        result = ols_fit(f"secondary_{term}", "raw_units", sample, terms)
        table = coefficient_table(result, "HC3")
        table["is_variability_term"] = table["term"].eq(term)
        frames.append(table)
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output["variability_p_value_fdr_bh"] = np.nan
    for idx, row in output.iterrows():
        if row.get("is_variability_term", False):
            variability_indices.append(idx)
            variability_pvalues.append(float(row["p_value"]))
    for idx, adjusted in zip(variability_indices, bh_fdr(variability_pvalues)):
        output.loc[idx, "variability_p_value_fdr_bh"] = adjusted
    return output, midpoint_review


def sensitivity_dataset_for_feature_set(
    features: pd.DataFrame,
    phenotype: pd.DataFrame,
    feature_set: str,
    sex_labels: dict[str, str],
) -> pd.DataFrame:
    subset = features.loc[features["feature_set"].eq(feature_set)].copy()
    phenotype = phenotype.copy()
    subset["mesaid"] = subset["mesaid"].astype(str)
    phenotype["mesaid"] = phenotype["mesaid"].astype(str)
    merged = subset.merge(phenotype, on="mesaid", how="left", validate="one_to_one")
    merged = merged.rename(
        columns={
            "epslpscl5c": "ess",
            "sleepage5c": "age",
            "gender1": "sex",
            "bmi5c": "bmi",
            "ahi_a0h3a": "ahi",
        }
    )
    for column in ["ess", "age", "sex", "bmi", "ahi"]:
        merged[column] = numeric_series(merged[column])
    merged["sex_category"] = merged["sex"].dropna().astype(int).astype(str).map(sex_labels)
    merged["sex_category"] = merged["sex_category"].where(
        merged["sex_category"].notna(),
        merged["sex"].astype("string"),
    )
    merged["ahi_log1p"] = np.log1p(merged["ahi"])
    return merged


def sensitivity_model1_model2(
    label: str,
    df: pd.DataFrame,
    ahi_variable: str = "ahi",
    exclude_mesaids: set[str] | None = None,
    covariance: str = "HC3",
) -> dict[str, Any]:
    data = df.copy()
    if exclude_mesaids:
        data = data.loc[~data["mesaid"].astype(str).isin(exclude_mesaids)].copy()
    required = ["ess", "sleep_duration_mean", "sleep_duration_sd", "age", "sex_category", "bmi", ahi_variable]
    data = data.dropna(subset=required).copy()
    terms1 = ["sleep_duration_mean", "age", "C(sex_category)", "bmi", ahi_variable]
    terms2 = ["sleep_duration_mean", "sleep_duration_sd", "age", "C(sex_category)", "bmi", ahi_variable]
    result1 = ols_fit("sensitivity_model_1", covariance, data, terms1)
    result2 = ols_fit("sensitivity_model_2", covariance, data, terms2)
    table2 = coefficient_table(result2, "HC3" if covariance == "HC3" else "classical")
    row = table2.loc[table2["term"].eq("sleep_duration_sd")].iloc[0]
    se_column = "robust_se" if covariance == "HC3" else "classical_se"
    return {
        "analysis": label,
        "feature_set": str(data["feature_set"].iloc[0]) if len(data) and "feature_set" in data else "",
        "covariance": covariance,
        "ahi_variable": ahi_variable,
        "excluded_mesaids": json.dumps(sorted(exclude_mesaids or [])),
        "n": result2.n,
        "sleep_duration_sd_coefficient": float(row["coefficient"]),
        "sleep_duration_sd_standard_error": float(row[se_column]),
        "sleep_duration_sd_ci_lower": float(row["ci_lower"]),
        "sleep_duration_sd_ci_upper": float(row["ci_upper"]),
        "sleep_duration_sd_p_value": float(row["p_value"]),
        "delta_r_squared_model2_minus_model1": result2.r_squared - result1.r_squared,
        "model1_r_squared": result1.r_squared,
        "model2_r_squared": result2.r_squared,
        "model1_adjusted_r_squared": result1.adjusted_r_squared,
        "model2_adjusted_r_squared": result2.adjusted_r_squared,
    }


def build_sensitivity_analysis(
    features: pd.DataFrame,
    phenotype: pd.DataFrame,
    primary_df: pd.DataFrame,
    sex_labels: dict[str, str],
    influential: pd.DataFrame,
    ahi_skewness: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_set in SENSITIVITY_FEATURE_SETS:
        data = sensitivity_dataset_for_feature_set(features, phenotype, feature_set, sex_labels)
        rows.append(sensitivity_model1_model2(f"feature_set_{feature_set}", data))
    if ahi_skewness > 1:
        rows.append(sensitivity_model1_model2("ahi_log1p_sensitivity", primary_df, ahi_variable="ahi_log1p"))
    else:
        row = sensitivity_model1_model2("ahi_log1p_sensitivity_not_indicated", primary_df)
        row["notes"] = "AHI skewness threshold > 1 was not met; row repeats raw AHI specification."
        rows.append(row)
    model2_ids = set(influential.loc[influential["model"].eq("Model 2"), "mesaid"].astype(str).tolist()) if not influential.empty else set()
    rows.append(sensitivity_model1_model2("exclude_influential_model2_observations", primary_df, exclude_mesaids=model2_ids))
    rows.append(sensitivity_model1_model2("classical_se_comparison", primary_df, covariance="classical"))
    rows.append(sensitivity_model1_model2("hc3_se_comparison", primary_df, covariance="HC3"))
    output = pd.DataFrame(rows)
    output["ahi_skewness_primary_complete_case"] = ahi_skewness
    output["log1p_ahi_sensitivity_indicated"] = ahi_skewness > 1
    return output


def main() -> None:
    setup_logging()
    logging.info("Project root: %s", PROJECT_ROOT)
    backup_existing_outputs()

    features, phenotype, dictionary, domains = load_inputs()
    variable_dictionary, sex_labels, sex_coding_confirmed = build_variable_dictionary(
        dictionary, domains, phenotype
    )
    write_csv(variable_dictionary, REGRESSION_DICTIONARY_PATH)
    if not sex_coding_confirmed:
        raise RuntimeError("Official gender1 coding was not confirmed; stopping pilot regression.")

    merged, sample_flow, missingness = prepare_analysis_data(
        features, phenotype, sex_labels, sex_coding_confirmed
    )
    write_csv(merged, PILOT_DATASET_PATH)
    write_csv(sample_flow, SAMPLE_FLOW_PATH)
    write_csv(missingness, MISSINGNESS_PATH)
    write_csv(build_descriptive_statistics(merged), DESCRIPTIVE_PATH)
    write_csv(build_correlations(merged, "pearson"), PEARSON_PATH)
    write_csv(build_correlations(merged, "spearman"), SPEARMAN_PATH)

    coefficients, model_fit, fitted, primary_common = fit_primary_models(merged)
    write_csv(coefficients, COEFFICIENTS_PATH)
    write_csv(model_fit, MODEL_FIT_PATH)
    write_csv(build_incremental_comparison(fitted), INCREMENTAL_PATH)
    write_csv(build_vif(fitted), VIF_PATH)
    diagnostics, influential = build_diagnostics_and_influence(fitted)
    write_csv(influential, INFLUENTIAL_PATH)
    write_csv(diagnostics, DIAGNOSTICS_PATH)
    save_diagnostic_figures(fitted)

    secondary, midpoint_review = build_secondary_models(merged)
    write_csv(secondary, SECONDARY_PATH)
    write_csv(midpoint_review, MIDPOINT_OUTLIER_PATH)

    ahi_skewness = float(numeric_series(merged.loc[merged["primary_common_complete_case"], "ahi"]).skew())
    sensitivity = build_sensitivity_analysis(
        features=features,
        phenotype=phenotype,
        primary_df=merged,
        sex_labels=sex_labels,
        influential=influential,
        ahi_skewness=ahi_skewness,
    )
    write_csv(sensitivity, SENSITIVITY_PATH)

    print("Pilot regression analysis complete")
    print(f"Merged participants: {len(merged)}")
    print(f"Primary common complete-case participants: {int(merged['primary_common_complete_case'].sum())}")
    print(f"AHI skewness: {ahi_skewness:.3f}")
    print(f"Influential Model 2 observations: {int(influential['model'].eq('Model 2').sum()) if not influential.empty else 0}")


if __name__ == "__main__":
    main()
