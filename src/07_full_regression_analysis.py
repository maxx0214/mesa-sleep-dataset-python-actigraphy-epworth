"""Full MESA actigraphy regression analysis with statsmodels.

This script extends the previously validated pilot analysis to the full
available actigraphy sample. It keeps the fixed QC rule, feature window,
variables, and model specifications. It does not use harmonized data and does
not modify raw data.
"""

from __future__ import annotations

import importlib
import json
import logging
import platform
import shutil
import struct
import sys
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIGRAPHY_DIR = PROJECT_ROOT / "data" / "raw" / "actigraphy"
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
REST_BLOCKS_PATH = INTERIM_DIR / "rest_s_blocks.csv"
DAILY_SLEEP_PATH = INTERIM_DIR / "provisional_daily_sleep_metrics.csv"
ACTIGRAPHY_FEASIBILITY_PATH = OUTPUT_TABLE_DIR / "actigraphy_feasibility_summary.csv"
BOUNDARY_COMPARISON_PATH = OUTPUT_TABLE_DIR / "boundary_validity_comparison.csv"
PHENOTYPE_PATH = DATASET_DIR / "mesa-sleep-dataset-0.8.0.csv"
DICTIONARY_PATH = DATASET_DIR / "mesa-data-dictionary-0.8.0-variables.csv"
DOMAINS_PATH = DATASET_DIR / "mesa-data-dictionary-0.8.0-domains.csv"

FULL_DATASET_PATH = PROCESSED_DIR / "full_regression_dataset.csv"
SAMPLE_FLOW_PATH = OUTPUT_TABLE_DIR / "full_sample_flow.csv"
MISSINGNESS_PATH = OUTPUT_TABLE_DIR / "full_missingness_summary.csv"
DESCRIPTIVE_PATH = OUTPUT_TABLE_DIR / "full_descriptive_statistics.csv"
PEARSON_PATH = OUTPUT_TABLE_DIR / "full_pearson_correlations.csv"
SPEARMAN_PATH = OUTPUT_TABLE_DIR / "full_spearman_correlations.csv"
COEFFICIENTS_PATH = OUTPUT_TABLE_DIR / "full_regression_coefficients.csv"
MODEL_FIT_PATH = OUTPUT_TABLE_DIR / "full_regression_model_fit.csv"
INCREMENTAL_PATH = OUTPUT_TABLE_DIR / "full_incremental_value_comparison.csv"
SECONDARY_PATH = OUTPUT_TABLE_DIR / "full_secondary_variability_models.csv"
SENSITIVITY_PATH = OUTPUT_TABLE_DIR / "full_sensitivity_analysis.csv"
VIF_PATH = OUTPUT_TABLE_DIR / "full_vif.csv"
INFLUENTIAL_PATH = OUTPUT_TABLE_DIR / "full_influential_observations.csv"
DIAGNOSTICS_PATH = OUTPUT_TABLE_DIR / "full_model_diagnostics.csv"
LOG_PATH = OUTPUT_LOG_DIR / "07_full_regression_analysis.log"

FIGURE_PATHS = {
    "Model 1 residuals": OUTPUT_FIGURE_DIR / "full_model_1_residuals.png",
    "Model 2 residuals": OUTPUT_FIGURE_DIR / "full_model_2_residuals.png",
    "Model 3 residuals": OUTPUT_FIGURE_DIR / "full_model_3_residuals.png",
    "Model 2 qq": OUTPUT_FIGURE_DIR / "full_model_2_qq.png",
    "Model 2 influence": OUTPUT_FIGURE_DIR / "full_model_2_influence.png",
}

REQUESTED_OUTPUTS = [
    FULL_DATASET_PATH,
    SAMPLE_FLOW_PATH,
    MISSINGNESS_PATH,
    DESCRIPTIVE_PATH,
    PEARSON_PATH,
    SPEARMAN_PATH,
    COEFFICIENTS_PATH,
    MODEL_FIT_PATH,
    INCREMENTAL_PATH,
    SECONDARY_PATH,
    SENSITIVITY_PATH,
    VIF_PATH,
    INFLUENTIAL_PATH,
    DIAGNOSTICS_PATH,
    *FIGURE_PATHS.values(),
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
    "sex_Male",
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
PRIMARY_COMMON_VARIABLES = [
    "ess",
    "age",
    "sex_Male",
    "bmi",
    "ahi",
    "sleep_duration_mean",
    "sleep_duration_sd",
    "sleep_duration_cv",
]
MODEL_TERMS = {
    "Model 0": ["age", "sex_Male", "bmi", "ahi"],
    "Model 1": ["sleep_duration_mean", "age", "sex_Male", "bmi", "ahi"],
    "Model 2": ["sleep_duration_mean", "sleep_duration_sd", "age", "sex_Male", "bmi", "ahi"],
    "Model 3": ["sleep_duration_mean", "sleep_duration_cv", "age", "sex_Male", "bmi", "ahi"],
}
SECONDARY_TERMS = [
    "sleep_efficiency_sd",
    "waso_sd",
    "sleep_midpoint_circular_sd",
]

sm = None
statsmodels_module = None
variance_inflation_factor = None
OLSInfluence = None
het_breuschpagan = None


@dataclass
class ModelResult:
    model: str
    robust_fit: Any
    classical_fit: Any
    data: pd.DataFrame
    design: pd.DataFrame
    inference_label: str


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
        logging.info("No requested full-analysis outputs existed before this run.")
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for path in existing:
        destination = BACKUP_DIR / path.name
        shutil.copy2(path, destination)
        logging.info("Backed up %s to %s", path, destination)


def check_environment() -> None:
    global sm, statsmodels_module, variance_inflation_factor, OLSInfluence, het_breuschpagan
    try:
        statsmodels_module = importlib.import_module("statsmodels")
        sm = importlib.import_module("statsmodels.api")
        outliers = importlib.import_module("statsmodels.stats.outliers_influence")
        diagnostics = importlib.import_module("statsmodels.stats.diagnostic")
        variance_inflation_factor = outliers.variance_inflation_factor
        OLSInfluence = outliers.OLSInfluence
        het_breuschpagan = diagnostics.het_breuschpagan
    except Exception as exc:
        message = (
            "statsmodels import failed. "
            f"sys.executable={sys.executable}; error={type(exc).__name__}: {exc}"
        )
        print(message)
        logging.error(message)
        raise SystemExit(1) from exc

    logging.info("sys.executable: %s", sys.executable)
    logging.info("python_version: %s", platform.python_version())
    logging.info("numpy_version: %s", np.__version__)
    logging.info("scipy_version: %s", scipy.__version__)
    logging.info("pandas_version: %s", pd.__version__)
    logging.info("statsmodels_version: %s", statsmodels_module.__version__)


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def metric_value(table: pd.DataFrame, metric: str) -> int | float:
    if table.empty or "metric" not in table:
        return np.nan
    match = table.loc[table["metric"].astype(str).eq(metric), "value"]
    if match.empty:
        return np.nan
    value = pd.to_numeric(match.iloc[0], errors="coerce")
    return float(value) if pd.notna(value) else np.nan


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


def sex_labels_from_dictionary(dictionary: pd.DataFrame, domains: pd.DataFrame) -> dict[str, str]:
    match = dictionary.loc[dictionary["id"].astype(str).eq("gender1")]
    if match.empty:
        return {}
    domain_id = str(match.iloc[0].get("domain", "") or "")
    coding = domain_coding(domains, domain_id)
    return {str(entry["value"]): str(entry["display_name"]) for entry in coding}


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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = read_csv_with_fallback(FEATURES_PATH)
    phenotype = read_csv_with_fallback(PHENOTYPE_PATH, usecols=PHENOTYPE_COLUMNS)
    dictionary = read_csv_with_fallback(DICTIONARY_PATH)
    domains = read_csv_with_fallback(DOMAINS_PATH)
    rest_blocks = read_csv_with_fallback(REST_BLOCKS_PATH)
    daily = read_csv_with_fallback(DAILY_SLEEP_PATH)
    feasibility = read_csv_with_fallback(ACTIGRAPHY_FEASIBILITY_PATH)
    boundary = read_csv_with_fallback(BOUNDARY_COMPARISON_PATH)
    return features, phenotype, dictionary, domains, rest_blocks, daily, feasibility, boundary


def add_analysis_columns(
    merged: pd.DataFrame,
    sex_labels: dict[str, str],
) -> pd.DataFrame:
    merged = merged.copy()
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
        "sleep_duration_mean",
        "sleep_duration_sd",
        "sleep_duration_cv",
        "sleep_efficiency_sd",
        "waso_sd",
        "sleep_midpoint_circular_sd",
    ]:
        if column in merged:
            merged[column] = numeric_series(merged[column])

    sex_codes = numeric_series(merged["sex"]).dropna().astype(int).astype(str)
    merged["sex_category"] = pd.Series(pd.NA, index=merged.index, dtype="string")
    merged.loc[sex_codes.index, "sex_category"] = sex_codes.map(sex_labels).astype("string")
    merged["sex_category"] = merged["sex_category"].where(
        merged["sex_category"].notna(),
        merged["sex"].astype("string"),
    )
    merged["sex_Male"] = np.nan
    merged.loc[merged["sex_category"].astype(str).eq("Female"), "sex_Male"] = 0.0
    merged.loc[merged["sex_category"].astype(str).eq("Male"), "sex_Male"] = 1.0
    merged["ahi_log1p"] = np.where(merged["ahi"].ge(0), np.log1p(merged["ahi"]), np.nan)
    merged["official_sleep_midpoint_minutes"] = merged["avgsleepmidpoint5"].apply(parse_time_to_minutes)
    merged["midpoint_diff_minutes"] = circular_diff_minutes(
        merged["sleep_midpoint_circular_mean"],
        merged["official_sleep_midpoint_minutes"],
    )
    merged["sleep_duration_cv_outlier"] = flag_iqr_outlier(merged["sleep_duration_cv"])
    merged["sleep_midpoint_circular_sd_outlier"] = flag_iqr_outlier(
        merged["sleep_midpoint_circular_sd"]
    )
    complete_mask = merged[PRIMARY_COMMON_VARIABLES].notna().all(axis=1)
    merged["primary_common_complete_case"] = complete_mask
    return merged


def prepare_analysis_data(
    features: pd.DataFrame,
    phenotype: pd.DataFrame,
    sex_labels: dict[str, str],
    rest_blocks: pd.DataFrame,
    feasibility: pd.DataFrame,
    boundary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = features.copy()
    phenotype = phenotype.copy()
    features["mesaid"] = features["mesaid"].astype(str)
    phenotype["mesaid"] = phenotype["mesaid"].astype(str)
    primary = features.loc[features["feature_set"].eq(PRIMARY_FEATURE_SET)].copy()
    primary_dups = int(primary["mesaid"].duplicated(keep=False).sum())
    phenotype_dups = int(phenotype["mesaid"].duplicated(keep=False).sum())
    merged = primary.merge(
        phenotype,
        on="mesaid",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    merged["phenotype_merge_status"] = merged["_merge"].astype(str)
    merged["unmatched_phenotype"] = merged["_merge"].ne("both")
    merged = merged.drop(columns=["_merge"])
    merged = add_analysis_columns(merged, sex_labels)

    b2 = boundary.loc[boundary["criterion"].astype(str).eq("B2_boundary_aware")]
    b2_row = b2.iloc[0] if not b2.empty else pd.Series(dtype=object)
    rest_s_participants = int(rest_blocks["mesaid"].nunique()) if "mesaid" in rest_blocks else np.nan
    all_covariates = ["age", "sex_Male", "bmi", "ahi"]

    sample_rows = [
        {
            "step": "actigraphy_filesystem_mesa_sleep_csv",
            "n": len(list(ACTIGRAPHY_DIR.glob("mesa-sleep-*.csv"))),
            "notes": "Pattern: data/raw/actigraphy/mesa-sleep-*.csv.",
        },
        {
            "step": "actigraphy_successfully_parsed_files",
            "n": metric_value(feasibility, "successfully_parsed_file_count"),
            "notes": "From actigraphy_feasibility_summary.csv.",
        },
        {
            "step": "actigraphy_parse_failed_files",
            "n": metric_value(feasibility, "parse_failed_file_count"),
            "notes": "From actigraphy_feasibility_summary.csv.",
        },
        {
            "step": "participants_with_rest_s",
            "n": rest_s_participants,
            "notes": "Unique mesaid with at least one REST-S candidate block.",
        },
        {
            "step": "b2_valid_night_1plus_participants",
            "n": int(b2_row.get("valid_night_1plus_participants", np.nan)),
            "notes": "B2 boundary-aware valid-night rule.",
        },
        {
            "step": "b2_valid_night_5plus_participants",
            "n": int(b2_row.get("valid_night_5plus_participants", np.nan)),
            "notes": "B2 boundary-aware valid-night rule.",
        },
        {
            "step": "b2_valid_night_7plus_participants",
            "n": int(b2_row.get("valid_night_7plus_participants", np.nan)),
            "notes": "B2 boundary-aware valid-night rule.",
        },
        {
            "step": "primary_feature_rows",
            "n": int(len(primary)),
            "notes": f"feature_set = {PRIMARY_FEATURE_SET}.",
        },
        {
            "step": "duplicate_mesaid_in_primary_features",
            "n": primary_dups,
            "notes": "Rows with duplicate mesaid before merge.",
        },
        {
            "step": "main_phenotype_rows",
            "n": int(len(phenotype)),
            "notes": "mesa-sleep-dataset-0.8.0.csv; native mesaid merge only.",
        },
        {
            "step": "duplicate_mesaid_in_main_phenotype",
            "n": phenotype_dups,
            "notes": "Rows with duplicate mesaid before merge.",
        },
        {
            "step": "phenotype_merge_success_participants",
            "n": int(merged["phenotype_merge_status"].eq("both").sum()),
            "notes": "Primary feature rows matched to phenotype by mesaid.",
        },
        {
            "step": "unmatched_mesaid",
            "n": int(merged["unmatched_phenotype"].sum()),
            "notes": "Feature participants not found in the native phenotype file.",
        },
        {
            "step": "ess_nonmissing_participants",
            "n": int(merged["ess"].notna().sum()),
            "notes": "ESS = epslpscl5c among merged primary-feature rows.",
        },
        {
            "step": "age_sex_bmi_ahi_nonmissing_participants",
            "n": int(merged[all_covariates].notna().all(axis=1).sum()),
            "notes": "Complete native covariate set: sleepage5c, gender1, bmi5c, ahi_a0h3a.",
        },
        {
            "step": "primary_model_common_complete_case",
            "n": int(merged["primary_common_complete_case"].sum()),
            "notes": "Common sample for Models 0-3.",
        },
        {
            "step": "excluded_from_primary_common_complete_case",
            "n": int((~merged["primary_common_complete_case"]).sum()),
            "notes": "Any missing variable required by Models 0-3.",
        },
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
    for variable in REGRESSION_VARIABLES:
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
    samples = {
        "merged_primary_feature_rows": df,
        "primary_common_complete_case": df.loc[df["primary_common_complete_case"]],
    }
    for sample, sample_df in samples.items():
        for variable in continuous:
            rows.append(continuous_summary(sample, variable, labels[variable], sample_df[variable]))
        rows.extend(
            categorical_summaries(
                sample,
                "sex_category",
                labels["sex_category"],
                sample_df["sex_category"],
            )
        )
    return pd.DataFrame(rows)


def build_correlations(df: pd.DataFrame, method: str) -> pd.DataFrame:
    corr_vars = [
        "ess",
        "age",
        "sex_Male",
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


def design_matrix(sample: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    columns = [pd.Series(1.0, index=sample.index, name="Intercept")]
    for term in terms:
        columns.append(numeric_series(sample[term]).astype(float).rename(term))
    return pd.concat(columns, axis=1)


def fit_model(
    model_label: str,
    sample: pd.DataFrame,
    terms: list[str],
    use_t: bool | None = None,
    inference_label: str = "HC3_default_z",
) -> ModelResult:
    y = numeric_series(sample["ess"]).astype(float)
    x = design_matrix(sample, terms)
    classical = sm.OLS(y, x).fit()
    if use_t is None:
        robust = sm.OLS(y, x).fit(cov_type="HC3")
    else:
        robust = sm.OLS(y, x).fit(cov_type="HC3", use_t=use_t)
    return ModelResult(
        model=model_label,
        robust_fit=robust,
        classical_fit=classical,
        data=sample.copy(),
        design=x.copy(),
        inference_label=inference_label,
    )


def coefficient_table(result: ModelResult) -> pd.DataFrame:
    fit = result.robust_fit
    conf = fit.conf_int(alpha=0.05)
    conf.columns = ["ci_lower", "ci_upper"]
    rows: list[dict[str, Any]] = []
    for term in fit.params.index:
        rows.append(
            {
                "model": result.model,
                "scale": "raw_units",
                "term": term,
                "n": int(fit.nobs),
                "coefficient": float(fit.params[term]),
                "hc3_standard_error": float(fit.bse[term]),
                "ci_lower": float(conf.loc[term, "ci_lower"]),
                "ci_upper": float(conf.loc[term, "ci_upper"]),
                "p_value": float(fit.pvalues[term]),
                "reference_distribution": "t" if bool(getattr(fit, "use_t", False)) else "normal_z",
                "statsmodels_use_t": bool(getattr(fit, "use_t", False)),
                "r_squared": float(result.classical_fit.rsquared),
                "adjusted_r_squared": float(result.classical_fit.rsquared_adj),
                "aic": float(result.classical_fit.aic),
                "bic": float(result.classical_fit.bic),
                "rmse": float(np.sqrt(np.mean(np.square(result.classical_fit.resid)))),
                "residual_df": float(result.classical_fit.df_resid),
            }
        )
    return pd.DataFrame(rows)


def model_fit_row(result: ModelResult) -> dict[str, Any]:
    fit = result.classical_fit
    parameter_count = int(fit.df_model + 1)
    return {
        "model": result.model,
        "scale": "raw_units",
        "n": int(fit.nobs),
        "r_squared": float(fit.rsquared),
        "adjusted_r_squared": float(fit.rsquared_adj),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "rmse": float(np.sqrt(np.mean(np.square(fit.resid)))),
        "residual_df": float(fit.df_resid),
        "parameter_count": parameter_count,
        "observations_per_parameter": float(fit.nobs / parameter_count) if parameter_count else np.nan,
        "analysis_warning": "Observational cross-sectional analysis; do not interpret as causal.",
    }


def fit_primary_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, ModelResult], pd.DataFrame]:
    common = df.loc[df["primary_common_complete_case"]].copy()
    coefficient_frames: list[pd.DataFrame] = []
    fit_rows: list[dict[str, Any]] = []
    fitted: dict[str, ModelResult] = {}
    for model_label, terms in MODEL_TERMS.items():
        result = fit_model(model_label, common, terms)
        fitted[model_label] = result
        coefficient_frames.append(coefficient_table(result))
        fit_rows.append(model_fit_row(result))
    return pd.concat(coefficient_frames, ignore_index=True), pd.DataFrame(fit_rows), fitted, common


def build_incremental_comparison(fitted: dict[str, ModelResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = fitted["Model 1"]
    for label, added_term in [("Model 2", "sleep_duration_sd"), ("Model 3", "sleep_duration_cv")]:
        full = fitted[label]
        f_stat, f_p, df_diff = full.classical_fit.compare_f_test(base.classical_fit)
        coef = coefficient_table(full)
        added = coef.loc[coef["term"].eq(added_term)].iloc[0]
        rows.append(
            {
                "comparison": f"Model 1 vs {label}",
                "n": int(full.classical_fit.nobs),
                "delta_r_squared": float(full.classical_fit.rsquared - base.classical_fit.rsquared),
                "delta_adjusted_r_squared": float(
                    full.classical_fit.rsquared_adj - base.classical_fit.rsquared_adj
                ),
                "delta_aic": float(full.classical_fit.aic - base.classical_fit.aic),
                "delta_bic": float(full.classical_fit.bic - base.classical_fit.bic),
                "classical_partial_f_statistic": float(f_stat),
                "classical_partial_f_p_value": float(f_p),
                "classical_partial_f_df_diff": float(df_diff),
                "added_variable": added_term,
                "added_variable_coefficient": float(added["coefficient"]),
                "added_variable_hc3_standard_error": float(added["hc3_standard_error"]),
                "added_variable_hc3_ci_lower": float(added["ci_lower"]),
                "added_variable_hc3_ci_upper": float(added["ci_upper"]),
                "added_variable_hc3_p_value": float(added["p_value"]),
                "notes": "Classical partial F-test and HC3 robust inference use different assumptions.",
            }
        )
    return pd.DataFrame(rows)


def build_vif(fitted: dict[str, ModelResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_label, result in fitted.items():
        exog = np.asarray(result.classical_fit.model.exog, dtype=float)
        names = result.classical_fit.model.exog_names
        for idx, term in enumerate(names):
            if term == "Intercept":
                continue
            rows.append(
                {
                    "model": model_label,
                    "term": term,
                    "vif": float(variance_inflation_factor(exog, idx)),
                }
            )
    return pd.DataFrame(rows)


def build_diagnostics_and_influence(fitted: dict[str, ModelResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostic_rows: list[dict[str, Any]] = []
    influence_rows: list[dict[str, Any]] = []
    for model_label, result in fitted.items():
        fit = result.classical_fit
        influence = OLSInfluence(fit)
        cooks_distance = np.asarray(influence.cooks_distance[0], dtype=float)
        leverage = np.asarray(influence.hat_matrix_diag, dtype=float)
        studentized_internal = np.asarray(influence.resid_studentized_internal, dtype=float)
        studentized_external = np.asarray(influence.resid_studentized_external, dtype=float)
        n = int(fit.nobs)
        p = int(fit.df_model + 1)
        cooks_threshold = 4 / n
        leverage_threshold = 2 * p / n
        studentized_threshold = 3
        bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(fit.resid, fit.model.exog)
        shapiro_stat, shapiro_p = stats.shapiro(fit.resid) if n <= 5000 else (np.nan, np.nan)
        flagged = np.asarray(
            (cooks_distance > cooks_threshold)
            | (np.abs(studentized_internal) > studentized_threshold)
            | (leverage > leverage_threshold),
            dtype=bool,
        )
        diagnostic_rows.append(
            {
                "model": model_label,
                "n": n,
                "parameter_count": p,
                "residual_df": float(fit.df_resid),
                "cooks_distance_threshold": cooks_threshold,
                "leverage_threshold": leverage_threshold,
                "studentized_residual_threshold": studentized_threshold,
                "max_cooks_distance": float(np.max(cooks_distance)),
                "max_abs_studentized_internal_residual": float(np.max(np.abs(studentized_internal))),
                "max_abs_studentized_external_residual": float(np.max(np.abs(studentized_external))),
                "max_leverage": float(np.max(leverage)),
                "influential_observation_count": int(np.sum(flagged)),
                "breusch_pagan_lm": float(bp_lm),
                "breusch_pagan_lm_p_value": float(bp_lm_p),
                "breusch_pagan_f": float(bp_f),
                "breusch_pagan_f_p_value": float(bp_f_p),
                "shapiro_wilk_statistic": float(shapiro_stat),
                "shapiro_wilk_p_value": float(shapiro_p),
            }
        )
        data = result.data.reset_index(drop=True)
        for idx in np.flatnonzero(flagged):
            influence_rows.append(
                {
                    "model": model_label,
                    "mesaid": data.loc[idx, "mesaid"],
                    "cooks_distance": float(cooks_distance[idx]),
                    "leverage": float(leverage[idx]),
                    "studentized_internal_residual": float(studentized_internal[idx]),
                    "studentized_external_residual": float(studentized_external[idx]),
                    "flag_cooks_gt_4_over_n": bool(cooks_distance[idx] > cooks_threshold),
                    "flag_abs_studentized_residual_gt_3": bool(
                        abs(studentized_internal[idx]) > studentized_threshold
                    ),
                    "flag_leverage_gt_2p_over_n": bool(leverage[idx] > leverage_threshold),
                    "ess": data.loc[idx, "ess"],
                    "sleep_duration_mean": data.loc[idx].get("sleep_duration_mean", np.nan),
                    "sleep_duration_sd": data.loc[idx].get("sleep_duration_sd", np.nan),
                    "sleep_duration_cv": data.loc[idx].get("sleep_duration_cv", np.nan),
                }
            )
    return pd.DataFrame(diagnostic_rows), pd.DataFrame(influence_rows)


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError("PNG writer expects an RGB image.")
    raw_rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw_rows, level=6))
        + png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def new_canvas(width: int, height: int) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def draw_rect(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    height, width, _ = image.shape
    x0, x1 = sorted((max(0, x0), min(width - 1, x1)))
    y0, y1 = sorted((max(0, y0), min(height - 1, y1)))
    image[y0 : y1 + 1, x0 : x1 + 1] = color


def draw_line(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, steps + 1).round().astype(int)
    ys = np.linspace(y0, y1, steps + 1).round().astype(int)
    radius = max(0, thickness // 2)
    for x, y in zip(xs, ys):
        draw_rect(image, x - radius, y - radius, x + radius, y + radius, color)


def draw_point(image: np.ndarray, x: int, y: int, color: tuple[int, int, int], radius: int = 2) -> None:
    height, width, _ = image.shape
    for yy in range(max(0, y - radius), min(height, y + radius + 1)):
        for xx in range(max(0, x - radius), min(width, x + radius + 1)):
            if (xx - x) ** 2 + (yy - y) ** 2 <= radius**2:
                image[yy, xx] = color


def panel_bounds(index: int, width: int, height: int) -> tuple[int, int, int, int]:
    margin = 44
    gap = 42
    panel_width = (width - 2 * margin - gap) // 2
    x0 = margin + index * (panel_width + gap)
    x1 = x0 + panel_width
    y0 = margin
    y1 = height - margin
    return x0, y0, x1, y1


def scale_to_panel(
    x: np.ndarray,
    y: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = bounds
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if xmin == xmax:
        xmin -= 0.5
        xmax += 0.5
    if ymin == ymax:
        ymin -= 0.5
        ymax += 0.5
    px = x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)
    py = y1 - (y - ymin) / (ymax - ymin) * (y1 - y0)
    return px.round().astype(int), py.round().astype(int)


def draw_axes(image: np.ndarray, bounds: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = bounds
    draw_line(image, x0, y1, x1, y1, (45, 45, 45), 2)
    draw_line(image, x0, y0, x0, y1, (45, 45, 45), 2)
    draw_rect(image, x0, y0, x1, y0, (220, 220, 220))
    draw_rect(image, x1, y0, x1, y1, (220, 220, 220))


def draw_scatter_panel(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    bounds: tuple[int, int, int, int],
    color: tuple[int, int, int] = (44, 103, 242),
    horizontal_lines: list[float] | None = None,
) -> None:
    draw_axes(image, bounds)
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    px, py = scale_to_panel(x_values, y_values, bounds)
    for x_point, y_point in zip(px, py):
        draw_point(image, int(x_point), int(y_point), color, radius=2)
    if horizontal_lines:
        x0, y0, x1, y1 = bounds
        valid_y = y_values[np.isfinite(y_values)]
        if len(valid_y):
            ymin, ymax = float(np.min(valid_y)), float(np.max(valid_y))
            if ymin == ymax:
                ymin -= 0.5
                ymax += 0.5
            for value in horizontal_lines:
                if ymin <= value <= ymax:
                    y_line = int(round(y1 - (value - ymin) / (ymax - ymin) * (y1 - y0)))
                    draw_line(image, x0, y_line, x1, y_line, (210, 45, 45), 1)


def draw_histogram_panel(
    image: np.ndarray,
    values: np.ndarray,
    bounds: tuple[int, int, int, int],
    bins: int = 30,
) -> None:
    draw_axes(image, bounds)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return
    counts, _ = np.histogram(values, bins=bins)
    max_count = max(int(np.max(counts)), 1)
    x0, y0, x1, y1 = bounds
    panel_width = x1 - x0
    for idx, count in enumerate(counts):
        bar_x0 = x0 + int(idx / bins * panel_width)
        bar_x1 = x0 + int((idx + 1) / bins * panel_width) - 1
        bar_y1 = y1 - 1
        bar_y0 = y1 - int(count / max_count * (y1 - y0))
        draw_rect(image, bar_x0, bar_y0, bar_x1, bar_y1, (70, 130, 180))


def save_residual_plot(model_label: str, result: ModelResult, output_path: Path) -> None:
    del model_label
    fit = result.classical_fit
    residuals = np.asarray(fit.resid, dtype=float)
    fitted_values = np.asarray(fit.fittedvalues, dtype=float)
    image = new_canvas(980, 440)
    draw_scatter_panel(image, fitted_values, residuals, panel_bounds(0, 980, 440), horizontal_lines=[0])
    draw_histogram_panel(image, residuals, panel_bounds(1, 980, 440), bins=30)
    write_png(output_path, image)


def save_qq_plot(model_label: str, result: ModelResult, output_path: Path) -> None:
    del model_label
    residuals = np.asarray(result.classical_fit.resid, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    image = new_canvas(620, 520)
    bounds = (64, 48, 570, 468)
    draw_axes(image, bounds)
    if len(residuals):
        (theoretical, ordered), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
        draw_scatter_panel(image, theoretical, ordered, bounds, color=(44, 103, 242))
        x_line = np.array([np.min(theoretical), np.max(theoretical)])
        y_line = intercept + slope * x_line
        px, py = scale_to_panel(x_line, y_line, bounds)
        if len(px) == 2:
            draw_line(image, int(px[0]), int(py[0]), int(px[1]), int(py[1]), (210, 45, 45), 2)
    write_png(output_path, image)


def save_influence_plot(model_label: str, result: ModelResult, output_path: Path) -> None:
    del model_label
    influence = OLSInfluence(result.classical_fit)
    cooks = np.asarray(influence.cooks_distance[0], dtype=float)
    leverage = np.asarray(influence.hat_matrix_diag, dtype=float)
    studentized = np.asarray(influence.resid_studentized_internal, dtype=float)
    image = new_canvas(980, 440)
    draw_scatter_panel(
        image,
        leverage,
        studentized,
        panel_bounds(0, 980, 440),
        horizontal_lines=[-3, 3],
    )
    draw_scatter_panel(
        image,
        np.arange(len(cooks), dtype=float),
        cooks,
        panel_bounds(1, 980, 440),
        color=(70, 130, 180),
        horizontal_lines=[4 / len(cooks)],
    )
    write_png(output_path, image)


def save_diagnostic_figures(fitted: dict[str, ModelResult]) -> None:
    save_residual_plot("Model 1", fitted["Model 1"], FIGURE_PATHS["Model 1 residuals"])
    save_residual_plot("Model 2", fitted["Model 2"], FIGURE_PATHS["Model 2 residuals"])
    save_residual_plot("Model 3", fitted["Model 3"], FIGURE_PATHS["Model 3 residuals"])
    save_qq_plot("Model 2", fitted["Model 2"], FIGURE_PATHS["Model 2 qq"])
    save_influence_plot("Model 2", fitted["Model 2"], FIGURE_PATHS["Model 2 influence"])


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


def build_secondary_models(df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    variability_pvalues: list[float] = []
    variability_indices: list[int] = []
    for term in SECONDARY_TERMS:
        required = ["ess", "sleep_duration_mean", term, "age", "sex_Male", "bmi", "ahi"]
        sample = df.dropna(subset=required).copy()
        terms = ["sleep_duration_mean", term, "age", "sex_Male", "bmi", "ahi"]
        result = fit_model(f"secondary_{term}", sample, terms)
        table = coefficient_table(result)
        table["is_variability_term"] = table["term"].eq(term)
        frames.append(table)
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output["variability_p_value_fdr_bh"] = np.nan
    for idx, row in output.iterrows():
        if bool(row.get("is_variability_term", False)):
            variability_indices.append(idx)
            variability_pvalues.append(float(row["p_value"]))
    for idx, adjusted in zip(variability_indices, bh_fdr(variability_pvalues)):
        output.loc[idx, "variability_p_value_fdr_bh"] = adjusted
    return output


def dataset_for_feature_set(
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
    merged["unmatched_phenotype"] = merged["epslpscl5c"].isna()
    return add_analysis_columns(merged, sex_labels)


def sensitivity_model1_model2(
    label: str,
    df: pd.DataFrame,
    ahi_variable: str = "ahi",
    exclude_mesaids: set[str] | None = None,
    use_t: bool | None = None,
    notes: str = "",
) -> dict[str, Any]:
    data = df.copy()
    if exclude_mesaids:
        data = data.loc[~data["mesaid"].astype(str).isin(exclude_mesaids)].copy()
    required = ["ess", "sleep_duration_mean", "sleep_duration_sd", "age", "sex_Male", "bmi", ahi_variable]
    data = data.dropna(subset=required).copy()
    terms1 = ["sleep_duration_mean", "age", "sex_Male", "bmi", ahi_variable]
    terms2 = ["sleep_duration_mean", "sleep_duration_sd", "age", "sex_Male", "bmi", ahi_variable]
    inference_label = "HC3_use_t_true" if use_t else "HC3_default_z"
    result1 = fit_model("sensitivity_model_1", data, terms1, use_t=use_t, inference_label=inference_label)
    result2 = fit_model("sensitivity_model_2", data, terms2, use_t=use_t, inference_label=inference_label)
    table2 = coefficient_table(result2)
    row = table2.loc[table2["term"].eq("sleep_duration_sd")].iloc[0]
    return {
        "analysis": label,
        "feature_set": str(data["feature_set"].iloc[0]) if len(data) and "feature_set" in data else "",
        "inference": inference_label,
        "ahi_variable": ahi_variable,
        "excluded_mesaids": json.dumps(sorted(exclude_mesaids or [])),
        "excluded_mesaid_count": len(exclude_mesaids or []),
        "n": int(result2.classical_fit.nobs),
        "sleep_duration_sd_coefficient": float(row["coefficient"]),
        "sleep_duration_sd_hc3_standard_error": float(row["hc3_standard_error"]),
        "sleep_duration_sd_ci_lower": float(row["ci_lower"]),
        "sleep_duration_sd_ci_upper": float(row["ci_upper"]),
        "sleep_duration_sd_p_value": float(row["p_value"]),
        "delta_r_squared_model2_minus_model1": float(
            result2.classical_fit.rsquared - result1.classical_fit.rsquared
        ),
        "model1_r_squared": float(result1.classical_fit.rsquared),
        "model2_r_squared": float(result2.classical_fit.rsquared),
        "model1_adjusted_r_squared": float(result1.classical_fit.rsquared_adj),
        "model2_adjusted_r_squared": float(result2.classical_fit.rsquared_adj),
        "notes": notes,
    }


def build_sensitivity_analysis(
    features: pd.DataFrame,
    phenotype: pd.DataFrame,
    primary_df: pd.DataFrame,
    sex_labels: dict[str, str],
    influential: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(
        sensitivity_model1_model2(
            "primary_consecutive_7_daybynoon_hc3_default_z",
            primary_df,
            notes="Reference primary analysis; statsmodels default HC3 uses normal z inference.",
        )
    )
    for feature_set in SENSITIVITY_FEATURE_SETS:
        data = dataset_for_feature_set(features, phenotype, feature_set, sex_labels)
        rows.append(sensitivity_model1_model2(f"feature_set_{feature_set}", data))
    rows.append(
        sensitivity_model1_model2(
            "ahi_log1p_sensitivity",
            primary_df,
            ahi_variable="ahi_log1p",
            notes="AHI replaced by log1p(AHI); sleep duration SD and CV are not modeled together.",
        )
    )
    model2_ids = (
        set(influential.loc[influential["model"].eq("Model 2"), "mesaid"].astype(str).tolist())
        if not influential.empty
        else set()
    )
    rows.append(
        sensitivity_model1_model2(
            "exclude_influential_model2_observations",
            primary_df,
            exclude_mesaids=model2_ids,
            notes="Influential observations are excluded only in this sensitivity analysis.",
        )
    )
    rows.append(
        sensitivity_model1_model2(
            "hc3_use_t_true_comparison",
            primary_df,
            use_t=True,
            notes="Same HC3 covariance as primary but p-values and CIs use residual df t distribution.",
        )
    )
    output = pd.DataFrame(rows)
    output["ahi_skewness_primary_complete_case"] = float(
        numeric_series(primary_df.loc[primary_df["primary_common_complete_case"], "ahi"]).skew()
    )
    return output


def main() -> None:
    setup_logging()
    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Current working directory: %s", Path.cwd())
    backup_existing_outputs()
    check_environment()

    (
        features,
        phenotype,
        dictionary,
        domains,
        rest_blocks,
        daily,
        feasibility,
        boundary,
    ) = load_inputs()
    del daily
    sex_labels = sex_labels_from_dictionary(dictionary, domains)
    if sex_labels.get("0") != "Female" or sex_labels.get("1") != "Male":
        raise RuntimeError(f"Official gender1 coding was not confirmed as 0=Female, 1=Male: {sex_labels}")

    merged, sample_flow, missingness = prepare_analysis_data(
        features=features,
        phenotype=phenotype,
        sex_labels=sex_labels,
        rest_blocks=rest_blocks,
        feasibility=feasibility,
        boundary=boundary,
    )
    write_csv(merged, FULL_DATASET_PATH)
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

    secondary = build_secondary_models(merged)
    write_csv(secondary, SECONDARY_PATH)

    sensitivity = build_sensitivity_analysis(
        features=features,
        phenotype=phenotype,
        primary_df=merged,
        sex_labels=sex_labels,
        influential=influential,
    )
    write_csv(sensitivity, SENSITIVITY_PATH)

    print("Full regression analysis complete")
    print(f"Primary feature participants: {len(merged)}")
    print(f"Primary common complete-case participants: {len(primary_common)}")
    print(
        "Influential Model 2 observations: "
        f"{int(influential['model'].eq('Model 2').sum()) if not influential.empty else 0}"
    )


if __name__ == "__main__":
    main()
