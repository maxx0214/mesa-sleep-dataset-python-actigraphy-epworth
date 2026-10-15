"""Mean-adjusted and joint sleep variability analyses for the full sample.

This script creates new final add-on analyses without changing the validated
QC, feature window, variables, raw data, or prior outputs. Any existing outputs
from this script are copied to outputs/backups/<timestamp>/ before rewriting.
"""

from __future__ import annotations

import importlib
import logging
import math
import platform
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
BACKUP_DIR = PROJECT_ROOT / "outputs" / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")

FULL_DATASET_PATH = PROCESSED_DIR / "full_regression_dataset.csv"
PREVIOUS_COMPARISON_PATH = OUTPUT_TABLE_DIR / "full_variability_metric_comparison.csv"

VARIABLE_DEFINITIONS_PATH = OUTPUT_TABLE_DIR / "mean_variability_variable_definitions.csv"
SELF_MODELS_PATH = OUTPUT_TABLE_DIR / "self_mean_adjusted_variability_models.csv"
SELF_INCREMENTAL_PATH = OUTPUT_TABLE_DIR / "self_mean_adjusted_incremental_value.csv"
JOINT_SAMPLE_FLOW_PATH = OUTPUT_TABLE_DIR / "joint_variability_sample_flow.csv"
JOINT_COEFFICIENTS_PATH = OUTPUT_TABLE_DIR / "joint_variability_model_coefficients.csv"
JOINT_FIT_PATH = OUTPUT_TABLE_DIR / "joint_variability_model_fit.csv"
PEARSON_CORRELATIONS_PATH = OUTPUT_TABLE_DIR / "variability_pearson_correlations.csv"
SPEARMAN_CORRELATIONS_PATH = OUTPUT_TABLE_DIR / "variability_spearman_correlations.csv"
VARIABILITY_PAIRWISE_CORRELATIONS_PATH = OUTPUT_TABLE_DIR / "variability_metric_pairwise_correlations.csv"
JOINT_VIF_PATH = OUTPUT_TABLE_DIR / "joint_variability_vif.csv"
HIGH_CORRELATION_PAIRS_PATH = OUTPUT_TABLE_DIR / "high_correlation_pairs.csv"
ADJUSTED_QUARTILE_RESULTS_PATH = OUTPUT_TABLE_DIR / "adjusted_quartile_ess_results.csv"
PREVIOUS_VS_SELF_PATH = OUTPUT_TABLE_DIR / "previous_vs_self_mean_adjusted_results.csv"

FIGURE_PATHS = {
    "sleep_duration_sd": OUTPUT_FIGURE_DIR / "adjusted_ess_by_sleep_duration_sd_quartile.png",
    "sleep_efficiency_sd": OUTPUT_FIGURE_DIR / "adjusted_ess_by_sleep_efficiency_sd_quartile.png",
    "waso_sd": OUTPUT_FIGURE_DIR / "adjusted_ess_by_waso_sd_quartile.png",
    "sleep_midpoint_circular_sd": OUTPUT_FIGURE_DIR / "adjusted_ess_by_midpoint_sd_quartile.png",
}
PANEL_FIGURE_PATH = OUTPUT_FIGURE_DIR / "adjusted_ess_by_variability_quartiles_panel.png"
LOG_PATH = OUTPUT_LOG_DIR / "09_mean_adjusted_and_joint_variability_analysis.log"

REQUESTED_OUTPUTS = [
    VARIABLE_DEFINITIONS_PATH,
    SELF_MODELS_PATH,
    SELF_INCREMENTAL_PATH,
    JOINT_SAMPLE_FLOW_PATH,
    JOINT_COEFFICIENTS_PATH,
    JOINT_FIT_PATH,
    PEARSON_CORRELATIONS_PATH,
    SPEARMAN_CORRELATIONS_PATH,
    VARIABILITY_PAIRWISE_CORRELATIONS_PATH,
    JOINT_VIF_PATH,
    HIGH_CORRELATION_PAIRS_PATH,
    ADJUSTED_QUARTILE_RESULTS_PATH,
    PREVIOUS_VS_SELF_PATH,
    *FIGURE_PATHS.values(),
    PANEL_FIGURE_PATH,
]

PRIMARY_FEATURE_SET = "consecutive_7_daybynoon"
OUTCOME = "ess"
COVARIATES = ["age", "sex_Male", "bmi", "ahi"]
QUARTILE_LABELS = ["Q1", "Q2", "Q3", "Q4"]
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1")

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None
    PIL_AVAILABLE = False

sm = None
statsmodels_module = None
variance_inflation_factor = None


@dataclass(frozen=True)
class MetricConfig:
    family: str
    label: str
    mean_variable: str
    variability_variable: str
    quartile_variable: str
    mean_label: str
    variability_label: str
    mean_unit: str
    variability_unit: str
    unit_definition: str
    source_file: str
    source_detail: str
    figure_title: str


METRICS = [
    MetricConfig(
        family="duration",
        label="Sleep duration",
        mean_variable="sleep_duration_mean",
        variability_variable="sleep_duration_sd",
        quartile_variable="duration_sd_quartile",
        mean_label="Mean sleep duration",
        variability_label="Sleep duration SD",
        mean_unit="minutes",
        variability_unit="minutes",
        unit_definition=(
            "sleep_duration_minutes equals sleep epochs multiplied by the 30-second "
            "epoch length in minutes; participant mean and sample SD use these nightly minutes."
        ),
        source_file=(
            "src/02_validate_actigraphy.py:40,418; "
            "src/04_resolve_ids_and_validate_sleep_features.py:679-680"
        ),
        source_detail="EPOCH_SECONDS=30, EPOCH_MINUTES=0.5; SD uses ddof=1.",
        figure_title="Sleep duration SD",
    ),
    MetricConfig(
        family="efficiency",
        label="Sleep efficiency",
        mean_variable="sleep_efficiency_mean",
        variability_variable="sleep_efficiency_sd",
        quartile_variable="efficiency_sd_quartile",
        mean_label="Mean sleep efficiency",
        variability_label="Sleep efficiency SD",
        mean_unit="percent",
        variability_unit="percentage points",
        unit_definition=(
            "sleep_efficiency is sleep epochs divided by valid scored epochs times 100; "
            "participant mean is percent and SD is percentage points."
        ),
        source_file=(
            "src/02_validate_actigraphy.py:420-421; "
            "src/04_resolve_ids_and_validate_sleep_features.py:693-694"
        ),
        source_detail="The nightly efficiency variable is stored on a 0-100 scale.",
        figure_title="Sleep efficiency SD",
    ),
    MetricConfig(
        family="waso",
        label="WASO",
        mean_variable="waso_mean",
        variability_variable="waso_sd",
        quartile_variable="waso_sd_quartile",
        mean_label="Mean WASO",
        variability_label="WASO SD",
        mean_unit="minutes",
        variability_unit="minutes",
        unit_definition=(
            "waso_minutes is wake epochs between the first and last sleep epoch multiplied "
            "by the 30-second epoch length in minutes; mean and sample SD use these nightly minutes."
        ),
        source_file=(
            "src/02_validate_actigraphy.py:386-395; "
            "src/04_resolve_ids_and_validate_sleep_features.py:695-696"
        ),
        source_detail="WASO is calculated in minutes before participant-level summaries.",
        figure_title="WASO SD",
    ),
    MetricConfig(
        family="midpoint",
        label="Sleep midpoint",
        mean_variable="sleep_midpoint_circular_mean",
        variability_variable="sleep_midpoint_circular_sd",
        quartile_variable="midpoint_sd_quartile",
        mean_label="Mean sleep midpoint",
        variability_label="Sleep midpoint circular SD",
        mean_unit="clock minutes after midnight, circular mean",
        variability_unit="circular minutes on a 24-hour clock",
        unit_definition=(
            "sleep midpoint mean and SD are circular summaries of nightly "
            "sleep_midpoint_clock_minutes on the 24-hour clock; circular SD radians are "
            "converted to minutes by sd_radians / (2*pi) * 1440."
        ),
        source_file="src/04_resolve_ids_and_validate_sleep_features.py:657-674,697-698",
        source_detail=(
            "Existing circular mean and circular SD are reused; ordinary clock-time mean or SD is not recalculated."
        ),
        figure_title="Sleep midpoint circular SD",
    ),
]

MEAN_VARIABLES = [metric.mean_variable for metric in METRICS]
VARIABILITY_VARIABLES = [metric.variability_variable for metric in METRICS]
CORRELATION_VARIABLES = [*MEAN_VARIABLES, *VARIABILITY_VARIABLES]
JOINT_TERMS = [*MEAN_VARIABLES, *VARIABILITY_VARIABLES, *COVARIATES]
JOINT_REQUIRED = [OUTCOME, *JOINT_TERMS]


def ensure_dirs() -> None:
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


def backup_existing_outputs() -> None:
    existing = [path for path in REQUESTED_OUTPUTS if path.exists()]
    if not existing:
        logging.info("No requested outputs from script 09 existed before this run.")
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for path in existing:
        destination = BACKUP_DIR / path.name
        shutil.copy2(path, destination)
        logging.info("Backed up %s to %s", path, destination)


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


def check_environment() -> None:
    global sm, statsmodels_module, variance_inflation_factor
    try:
        statsmodels_module = importlib.import_module("statsmodels")
        sm = importlib.import_module("statsmodels.api")
        outliers = importlib.import_module("statsmodels.stats.outliers_influence")
        variance_inflation_factor = outliers.variance_inflation_factor
    except Exception as exc:
        message = (
            "statsmodels import failed. "
            f"sys.executable={sys.executable}; error={type(exc).__name__}: {exc}"
        )
        print(message)
        logging.error(message)
        raise SystemExit(1) from exc

    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Current working directory: %s", Path.cwd())
    logging.info("sys.executable: %s", sys.executable)
    logging.info("python_version: %s", platform.python_version())
    logging.info("numpy_version: %s", np.__version__)
    logging.info("scipy_version: %s", scipy.__version__)
    logging.info("pandas_version: %s", pd.__version__)
    logging.info("statsmodels_version: %s", statsmodels_module.__version__)
    logging.info("PIL_available_for_figures: %s", PIL_AVAILABLE)


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


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


def load_primary_dataset() -> pd.DataFrame:
    if not FULL_DATASET_PATH.exists():
        raise FileNotFoundError(f"Required input does not exist: {FULL_DATASET_PATH}")
    df = read_csv_with_fallback(FULL_DATASET_PATH)
    if "feature_set" not in df.columns:
        raise RuntimeError("full_regression_dataset.csv does not include feature_set.")
    primary = df.loc[df["feature_set"].eq(PRIMARY_FEATURE_SET)].copy()
    required = sorted(set(JOINT_REQUIRED + ["mesaid", "participant_key", "feature_set"]))
    missing = [column for column in required if column not in primary.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in full_regression_dataset.csv: {missing}")
    for column in JOINT_REQUIRED:
        primary[column] = numeric_series(primary[column])
    logging.info("Loaded %s primary-window rows.", len(primary))
    return primary


def design_matrix(sample: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    x = pd.DataFrame(index=sample.index)
    for term in terms:
        x[term] = numeric_series(sample[term]).astype(float)
    return sm.add_constant(x, has_constant="add")


def fit_ols_hc3(sample: pd.DataFrame, terms: list[str]) -> tuple[Any, Any, pd.DataFrame]:
    y = numeric_series(sample[OUTCOME]).astype(float)
    x = design_matrix(sample, terms)
    classical = sm.OLS(y, x).fit()
    robust = sm.OLS(y, x).fit(cov_type="HC3")
    return classical, robust, x


def conf_int_values(fit: Any, term: str) -> tuple[float, float]:
    conf = fit.conf_int(alpha=0.05)
    if isinstance(conf, pd.DataFrame):
        return float(conf.loc[term, 0]), float(conf.loc[term, 1])
    term_index = list(fit.params.index).index(term)
    return float(conf[term_index, 0]), float(conf[term_index, 1])


def standardized_beta(sample: pd.DataFrame, coefficient: float, variable: str) -> float:
    x_sd = numeric_series(sample[variable]).std(ddof=1)
    y_sd = numeric_series(sample[OUTCOME]).std(ddof=1)
    if pd.isna(x_sd) or pd.isna(y_sd) or y_sd == 0:
        return float("nan")
    return float(coefficient * x_sd / y_sd)


def rmse_from_fit(fit: Any) -> float:
    resid = np.asarray(fit.resid, dtype=float)
    return float(np.sqrt(np.mean(np.square(resid))))


def build_variable_definitions() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        rows.extend(
            [
                {
                    "family": metric.family,
                    "role": "mean",
                    "variable": metric.mean_variable,
                    "label": metric.mean_label,
                    "paired_variability_variable": metric.variability_variable,
                    "unit": metric.mean_unit,
                    "official_or_code_definition": metric.unit_definition,
                    "source_file": metric.source_file,
                    "source_detail": metric.source_detail,
                    "analysis_implication": (
                        "Use the existing feature column; do not recalculate from raw epochs in script 09."
                    ),
                },
                {
                    "family": metric.family,
                    "role": "variability",
                    "variable": metric.variability_variable,
                    "label": metric.variability_label,
                    "paired_mean_variable": metric.mean_variable,
                    "unit": metric.variability_unit,
                    "official_or_code_definition": metric.unit_definition,
                    "source_file": metric.source_file,
                    "source_detail": metric.source_detail,
                    "analysis_implication": (
                        "Use as the variability term in self-mean-adjusted and joint models."
                    ),
                },
            ]
        )
    return pd.DataFrame(rows)


def build_self_mean_adjusted_models(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, Any]] = []
    incremental_rows: list[dict[str, Any]] = []
    pvalues: list[float] = []

    for metric in METRICS:
        baseline_terms = [metric.mean_variable, *COVARIATES]
        extended_terms = [metric.mean_variable, metric.variability_variable, *COVARIATES]
        required = [OUTCOME, *extended_terms]
        sample = primary.dropna(subset=required).copy()
        baseline_classical, _, _ = fit_ols_hc3(sample, baseline_terms)
        extended_classical, extended_robust, _ = fit_ols_hc3(sample, extended_terms)
        term = metric.variability_variable
        ci_lower, ci_upper = conf_int_values(extended_robust, term)
        beta = float(extended_robust.params[term])
        p_value = float(extended_robust.pvalues[term])
        pvalues.append(p_value)

        model_rows.append(
            {
                "model_family": metric.family,
                "model_label": metric.label,
                "primary_window": PRIMARY_FEATURE_SET,
                "outcome": OUTCOME,
                "mean_variable": metric.mean_variable,
                "variability_metric": metric.variability_variable,
                "variability_metric_label": metric.variability_label,
                "unit": metric.variability_unit,
                "n": int(extended_classical.nobs),
                "coefficient": beta,
                "hc3_standard_error": float(extended_robust.bse[term]),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_value": p_value,
                "standardized_beta": standardized_beta(sample, beta, term),
                "r_squared": float(extended_classical.rsquared),
                "adjusted_r_squared": float(extended_classical.rsquared_adj),
                "baseline_r_squared_same_sample": float(baseline_classical.rsquared),
                "baseline_adjusted_r_squared_same_sample": float(baseline_classical.rsquared_adj),
                "delta_r_squared": float(extended_classical.rsquared - baseline_classical.rsquared),
                "delta_adjusted_r_squared": float(
                    extended_classical.rsquared_adj - baseline_classical.rsquared_adj
                ),
                "aic": float(extended_classical.aic),
                "bic": float(extended_classical.bic),
                "reference_distribution": (
                    "t" if bool(getattr(extended_robust, "use_t", False)) else "normal_z"
                ),
                "statsmodels_use_t": bool(getattr(extended_robust, "use_t", False)),
                "formula": (
                    f"ESS ~ {metric.mean_variable} + {term} + age + sex + BMI + AHI"
                ),
                "standardized_beta_method": (
                    "raw HC3 coefficient multiplied by predictor SD / ESS SD on the same sample"
                ),
            }
        )
        incremental_rows.append(
            {
                "model_family": metric.family,
                "model_label": metric.label,
                "primary_window": PRIMARY_FEATURE_SET,
                "baseline_formula": f"ESS ~ {metric.mean_variable} + age + sex + BMI + AHI",
                "extended_formula": (
                    f"ESS ~ {metric.mean_variable} + {term} + age + sex + BMI + AHI"
                ),
                "n": int(extended_classical.nobs),
                "baseline_r_squared": float(baseline_classical.rsquared),
                "extended_r_squared": float(extended_classical.rsquared),
                "delta_r_squared": float(extended_classical.rsquared - baseline_classical.rsquared),
                "baseline_adjusted_r_squared": float(baseline_classical.rsquared_adj),
                "extended_adjusted_r_squared": float(extended_classical.rsquared_adj),
                "delta_adjusted_r_squared": float(
                    extended_classical.rsquared_adj - baseline_classical.rsquared_adj
                ),
                "baseline_aic": float(baseline_classical.aic),
                "extended_aic": float(extended_classical.aic),
                "baseline_bic": float(baseline_classical.bic),
                "extended_bic": float(extended_classical.bic),
                "variability_metric": term,
                "variability_p_value": p_value,
            }
        )

    fdr = bh_fdr(pvalues)
    for row, adjusted in zip(model_rows, fdr):
        row["p_value_fdr_bh"] = adjusted
    for row, adjusted in zip(incremental_rows, fdr):
        row["variability_p_value_fdr_bh"] = adjusted
    return pd.DataFrame(model_rows), pd.DataFrame(incremental_rows)


def build_joint_sample_flow(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    joint = primary.dropna(subset=JOINT_REQUIRED).copy()
    original_n = int(len(primary))
    joint_n = int(len(joint))
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "row_type": "sample_summary",
            "variable": "original_primary_window_n",
            "original_n": original_n,
            "joint_complete_case_n": joint_n,
            "missing_count": np.nan,
            "nonmissing_count": original_n,
            "excluded_participant_count": original_n - joint_n,
            "notes": f"Primary window is {PRIMARY_FEATURE_SET}.",
        }
    )
    rows.append(
        {
            "row_type": "sample_summary",
            "variable": "joint_model_complete_case_n",
            "original_n": original_n,
            "joint_complete_case_n": joint_n,
            "missing_count": np.nan,
            "nonmissing_count": joint_n,
            "excluded_participant_count": original_n - joint_n,
            "notes": "Complete case for ESS, four means, four variability metrics, age, sex, BMI, and AHI.",
        }
    )
    for variable in JOINT_REQUIRED:
        missing_count = int(primary[variable].isna().sum())
        rows.append(
            {
                "row_type": "variable_missingness",
                "variable": variable,
                "original_n": original_n,
                "joint_complete_case_n": joint_n,
                "missing_count": missing_count,
                "nonmissing_count": int(primary[variable].notna().sum()),
                "excluded_participant_count": original_n - joint_n,
                "notes": "Variable-level missingness is not mutually exclusive.",
            }
        )
    return pd.DataFrame(rows), joint


def build_joint_model(joint: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Any, pd.DataFrame]:
    classical, robust, design = fit_ols_hc3(joint, JOINT_TERMS)
    conf = robust.conf_int(alpha=0.05)
    rows: list[dict[str, Any]] = []
    for term in ["const", *JOINT_TERMS]:
        ci_lower = float(conf.loc[term, 0])
        ci_upper = float(conf.loc[term, 1])
        beta = float(robust.params[term])
        variable_for_sd = term if term != "const" else ""
        rows.append(
            {
                "term": term,
                "term_role": (
                    "intercept"
                    if term == "const"
                    else "variability"
                    if term in VARIABILITY_VARIABLES
                    else "mean"
                    if term in MEAN_VARIABLES
                    else "covariate"
                ),
                "n": int(classical.nobs),
                "coefficient": beta,
                "standardized_beta": (
                    standardized_beta(joint, beta, variable_for_sd) if term != "const" else np.nan
                ),
                "hc3_standard_error": float(robust.bse[term]),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_value": float(robust.pvalues[term]),
                "reference_distribution": (
                    "t" if bool(getattr(robust, "use_t", False)) else "normal_z"
                ),
                "statsmodels_use_t": bool(getattr(robust, "use_t", False)),
            }
        )
    fit_rows = [
        {
            "model": "joint_variability_model",
            "primary_window": PRIMARY_FEATURE_SET,
            "n": int(classical.nobs),
            "r_squared": float(classical.rsquared),
            "adjusted_r_squared": float(classical.rsquared_adj),
            "aic": float(classical.aic),
            "bic": float(classical.bic),
            "rmse": rmse_from_fit(classical),
            "predictor_count_excluding_intercept": len(JOINT_TERMS),
            "formula": (
                "ESS ~ four sleep means + four sleep variability metrics + age + sex + BMI + AHI"
            ),
        }
    ]
    return pd.DataFrame(rows), pd.DataFrame(fit_rows), robust, design


def wide_correlation_matrix(df: pd.DataFrame, method: str) -> pd.DataFrame:
    matrix = df[CORRELATION_VARIABLES].corr(method=method)
    output = matrix.reset_index().rename(columns={"index": "variable"})
    return output


def pairwise_correlations(df: pd.DataFrame, method: str) -> list[dict[str, Any]]:
    matrix = df[CORRELATION_VARIABLES].corr(method=method)
    rows: list[dict[str, Any]] = []
    for i, var1 in enumerate(CORRELATION_VARIABLES):
        for var2 in CORRELATION_VARIABLES[i + 1 :]:
            r = float(matrix.loc[var1, var2])
            rows.append(
                {
                    "correlation_type": method,
                    "variable_1": var1,
                    "variable_2": var2,
                    "correlation": r,
                    "abs_correlation": abs(r),
                    "is_variability_pair": var1 in VARIABILITY_VARIABLES and var2 in VARIABILITY_VARIABLES,
                    "is_high_correlation_abs_ge_0_70": bool(abs(r) >= 0.7),
                }
            )
    return rows


def build_correlation_outputs(joint: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pearson = wide_correlation_matrix(joint, "pearson")
    spearman = wide_correlation_matrix(joint, "spearman")
    pair_rows = pairwise_correlations(joint, "pearson") + pairwise_correlations(joint, "spearman")
    pairwise = pd.DataFrame(pair_rows)
    variability_pairwise = pairwise.loc[pairwise["is_variability_pair"]].copy()
    high_pairs = pairwise.loc[pairwise["is_high_correlation_abs_ge_0_70"]].copy()
    high_pairs = high_pairs.sort_values(["correlation_type", "abs_correlation"], ascending=[True, False])
    return pearson, spearman, variability_pairwise, high_pairs


def build_vif(design: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    values = design.astype(float).to_numpy()
    for idx, term in enumerate(design.columns):
        if term == "const":
            continue
        vif = float(variance_inflation_factor(values, idx))
        rows.append(
            {
                "term": term,
                "term_role": (
                    "variability"
                    if term in VARIABILITY_VARIABLES
                    else "mean"
                    if term in MEAN_VARIABLES
                    else "covariate"
                ),
                "vif": vif,
                "vif_gt_5": bool(vif > 5),
                "vif_gt_10": bool(vif > 10),
                "notes": "Intercept excluded from reporting.",
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def assign_quartiles(sample: pd.DataFrame, metric: MetricConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = sample.copy()
    values = numeric_series(work[metric.variability_variable]).astype(float)
    try:
        quartiles = pd.qcut(values, q=4, labels=QUARTILE_LABELS, duplicates="raise")
        method = "pd.qcut(metric, q=4, duplicates='raise')"
        failure_reason = ""
    except ValueError as exc:
        ranks = values.rank(method="first")
        quartiles = pd.qcut(ranks, q=4, labels=QUARTILE_LABELS, duplicates="raise")
        method = "pd.qcut(rank(method='first'), q=4) after tied cutpoint failure"
        failure_reason = str(exc)
    work[metric.quartile_variable] = quartiles.astype(str)
    work[metric.quartile_variable + "_ordinal"] = work[metric.quartile_variable].map(
        {label: idx + 1 for idx, label in enumerate(QUARTILE_LABELS)}
    )
    quantiles = values.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    metadata = {
        "cutpoint_method": method,
        "qcut_failure_reason": failure_reason,
        "minimum": float(quantiles.loc[0.0]),
        "q1_q2_boundary_25th_percentile": float(quantiles.loc[0.25]),
        "q2_q3_boundary_50th_percentile": float(quantiles.loc[0.5]),
        "q3_q4_boundary_75th_percentile": float(quantiles.loc[0.75]),
        "maximum": float(quantiles.loc[1.0]),
        "n_unique_values": int(values.nunique(dropna=True)),
    }
    return work, metadata


def quartile_design(sample: pd.DataFrame, metric: MetricConfig, force_quartile: str | None = None) -> pd.DataFrame:
    x = pd.DataFrame(index=sample.index)
    source = force_quartile if force_quartile is not None else None
    for label in QUARTILE_LABELS[1:]:
        column = f"{label}_vs_Q1"
        if source is None:
            x[column] = sample[metric.quartile_variable].eq(label).astype(float)
        else:
            x[column] = 1.0 if source == label else 0.0
    for term in [metric.mean_variable, *COVARIATES]:
        x[term] = numeric_series(sample[term]).astype(float)
    return sm.add_constant(x, has_constant="add")


def trend_design(sample: pd.DataFrame, metric: MetricConfig) -> pd.DataFrame:
    x = pd.DataFrame(index=sample.index)
    x["quartile_ordinal"] = numeric_series(sample[metric.quartile_variable + "_ordinal"]).astype(float)
    for term in [metric.mean_variable, *COVARIATES]:
        x[term] = numeric_series(sample[term]).astype(float)
    return sm.add_constant(x, has_constant="add")


def marginal_adjusted_estimate(
    robust_fit: Any,
    robust_cov: pd.DataFrame,
    base_sample: pd.DataFrame,
    metric: MetricConfig,
    quartile: str,
    q1_design_mean: pd.Series,
) -> dict[str, float]:
    forced_design = quartile_design(base_sample, metric, force_quartile=quartile)
    l_vec = forced_design.mean(axis=0).reindex(robust_fit.params.index)
    params = robust_fit.params.reindex(l_vec.index)
    adjusted = float(np.dot(l_vec, params))
    cov = robust_cov.reindex(index=l_vec.index, columns=l_vec.index)
    se = float(np.sqrt(max(np.dot(l_vec, np.dot(cov.to_numpy(dtype=float), l_vec)), 0.0)))
    ci_lower = adjusted - 1.96 * se
    ci_upper = adjusted + 1.96 * se

    diff_vec = l_vec - q1_design_mean.reindex(l_vec.index)
    diff = float(np.dot(diff_vec, params))
    diff_se = float(np.sqrt(max(np.dot(diff_vec, np.dot(cov.to_numpy(dtype=float), diff_vec)), 0.0)))
    diff_ci_lower = diff - 1.96 * diff_se
    diff_ci_upper = diff + 1.96 * diff_se
    return {
        "adjusted_ess": adjusted,
        "adjusted_ess_standard_error": se,
        "adjusted_ess_ci_lower": ci_lower,
        "adjusted_ess_ci_upper": ci_upper,
        "adjusted_difference_vs_q1": diff,
        "adjusted_difference_vs_q1_standard_error": diff_se,
        "adjusted_difference_vs_q1_ci_lower": diff_ci_lower,
        "adjusted_difference_vs_q1_ci_upper": diff_ci_upper,
    }


def build_adjusted_quartile_results(primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        required = [OUTCOME, metric.mean_variable, metric.variability_variable, *COVARIATES]
        sample = primary.dropna(subset=required).copy()
        sample, metadata = assign_quartiles(sample, metric)
        y = numeric_series(sample[OUTCOME]).astype(float)
        x = quartile_design(sample, metric)
        classical = sm.OLS(y, x).fit()
        robust = sm.OLS(y, x).fit(cov_type="HC3")
        robust_cov = robust.cov_params()
        q1_design_mean = quartile_design(sample, metric, force_quartile="Q1").mean(axis=0)

        trend_x = trend_design(sample, metric)
        trend_classical = sm.OLS(y, trend_x).fit()
        trend_robust = sm.OLS(y, trend_x).fit(cov_type="HC3")
        p_for_trend = float(trend_robust.pvalues["quartile_ordinal"])
        trend_beta = float(trend_robust.params["quartile_ordinal"])

        for label in QUARTILE_LABELS:
            group = sample.loc[sample[metric.quartile_variable].eq(label)]
            estimates = marginal_adjusted_estimate(
                robust,
                robust_cov,
                sample,
                metric,
                label,
                q1_design_mean,
            )
            if label == "Q1":
                p_value = np.nan
            else:
                p_value = float(robust.pvalues[f"{label}_vs_Q1"])
            rows.append(
                {
                    "model_family": metric.family,
                    "variability_metric": metric.variability_variable,
                    "variability_metric_label": metric.variability_label,
                    "mean_variable_adjusted": metric.mean_variable,
                    "quartile_variable": metric.quartile_variable,
                    "quartile": label,
                    "quartile_ordinal": QUARTILE_LABELS.index(label) + 1,
                    "n": int(len(group)),
                    "model_n": int(classical.nobs),
                    "raw_ess_mean": float(numeric_series(group[OUTCOME]).mean()),
                    "raw_ess_sd": float(numeric_series(group[OUTCOME]).std(ddof=1)),
                    "raw_ess_median": float(numeric_series(group[OUTCOME]).median()),
                    "metric_min": float(numeric_series(group[metric.variability_variable]).min()),
                    "metric_max": float(numeric_series(group[metric.variability_variable]).max()),
                    **estimates,
                    "p_value_vs_q1": p_value,
                    "p_for_trend": p_for_trend,
                    "trend_beta_per_quartile": trend_beta,
                    "r_squared": float(classical.rsquared),
                    "adjusted_r_squared": float(classical.rsquared_adj),
                    "trend_r_squared": float(trend_classical.rsquared),
                    **metadata,
                    "adjustment_method": (
                        "Marginal standardization: actual covariates retained and only quartile set to Q1-Q4."
                    ),
                    "ci_method": "Delta method with statsmodels HC3 covariance matrix and normal 1.96 multiplier.",
                }
            )
    return pd.DataFrame(rows)


def build_previous_vs_self(self_models: pd.DataFrame) -> pd.DataFrame:
    if not PREVIOUS_COMPARISON_PATH.exists():
        rows = []
        for _, row in self_models.iterrows():
            rows.append(
                {
                    "variability_metric": row["variability_metric"],
                    "previous_result_found": False,
                    "previous_source_file": relative_path(PREVIOUS_COMPARISON_PATH),
                    "new_standardized_beta": row["standardized_beta"],
                    "new_p_value": row["p_value"],
                    "new_fdr_p_value": row["p_value_fdr_bh"],
                    "new_delta_r_squared": row["delta_r_squared"],
                }
            )
        return pd.DataFrame(rows)

    previous = read_csv_with_fallback(PREVIOUS_COMPARISON_PATH)
    if "analysis_sample_type" in previous.columns:
        previous = previous.loc[previous["analysis_sample_type"].eq("common_all_metrics")].copy()
    rows = []
    for _, row in self_models.iterrows():
        metric = row["variability_metric"]
        match = previous.loc[previous["variability_metric"].astype(str).eq(str(metric))]
        if match.empty:
            rows.append(
                {
                    "variability_metric": metric,
                    "previous_result_found": False,
                    "previous_source_file": relative_path(PREVIOUS_COMPARISON_PATH),
                    "new_standardized_beta": row["standardized_beta"],
                    "new_p_value": row["p_value"],
                    "new_fdr_p_value": row["p_value_fdr_bh"],
                    "new_delta_r_squared": row["delta_r_squared"],
                }
            )
            continue
        prev = match.iloc[0]
        prev_fdr = (
            float(prev["p_value_fdr_bh_all_four"])
            if "p_value_fdr_bh_all_four" in prev and pd.notna(prev["p_value_fdr_bh_all_four"])
            else np.nan
        )
        rows.append(
            {
                "variability_metric": metric,
                "previous_result_found": True,
                "previous_source_file": relative_path(PREVIOUS_COMPARISON_PATH),
                "previous_baseline": "sleep_duration_mean for every variability model",
                "new_baseline": "the paired mean for each variability model",
                "previous_standardized_beta": float(prev["standardized_beta"]),
                "new_standardized_beta": float(row["standardized_beta"]),
                "standardized_beta_change": float(row["standardized_beta"] - prev["standardized_beta"]),
                "previous_p_value": float(prev["p_value"]),
                "new_p_value": float(row["p_value"]),
                "previous_fdr_p_value": prev_fdr,
                "new_fdr_p_value": float(row["p_value_fdr_bh"]),
                "previous_delta_r_squared": float(prev["delta_r_squared"]),
                "new_delta_r_squared": float(row["delta_r_squared"]),
                "delta_r_squared_change": float(row["delta_r_squared"] - prev["delta_r_squared"]),
                "significance_changed_p_lt_0_05": bool(
                    (float(prev["p_value"]) < 0.05) != (float(row["p_value"]) < 0.05)
                ),
                "significance_changed_fdr_lt_0_05": (
                    bool((prev_fdr < 0.05) != (float(row["p_value_fdr_bh"]) < 0.05))
                    if pd.notna(prev_fdr)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def get_font(size: int, bold: bool = False) -> Any:
    if not PIL_AVAILABLE:
        return None
    candidates = [
        Path("C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_centered_text(draw: Any, xy: tuple[int, int], text: str, font: Any, fill: str = "#263238") -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    x = xy[0] - (bbox[2] - bbox[0]) / 2
    y = xy[1] - (bbox[3] - bbox[1]) / 2
    draw.text((x, y), text, font=font, fill=fill)


def format_p(value: float) -> str:
    if pd.isna(value):
        return "p=NA"
    if value < 0.001:
        return "p<0.001"
    return f"p={value:.3f}"


def y_bounds_for_rows(rows: pd.DataFrame) -> tuple[float, float]:
    lower = float(rows["adjusted_ess_ci_lower"].min())
    upper = float(rows["adjusted_ess_ci_upper"].max())
    pad = max((upper - lower) * 0.18, 0.4)
    return math.floor((lower - pad) * 2) / 2, math.ceil((upper + pad) * 2) / 2


def draw_quartile_panel(
    image: Any,
    rows: pd.DataFrame,
    bounds: tuple[int, int, int, int],
    metric: MetricConfig,
    title_font: Any,
    label_font: Any,
    small_font: Any,
    color: str,
    y_min: float,
    y_max: float,
) -> None:
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = bounds
    draw.rectangle((left, top, right, bottom), fill="#ffffff", outline="#d8e0e8")
    draw.text((left + 22, top + 18), metric.figure_title, font=title_font, fill="#1f2933")
    trend_p = float(rows["p_for_trend"].iloc[0])
    model_n = int(rows["model_n"].iloc[0])
    draw.text(
        (left + 22, top + 50),
        f"Adjusted ESS, 95% CI; n={model_n}; trend {format_p(trend_p)}",
        font=small_font,
        fill="#52616b",
    )

    px0, py0, px1, py1 = left + 72, top + 95, right - 40, bottom - 70
    draw.line((px0, py1, px1, py1), fill="#90a4ae", width=2)
    draw.line((px0, py0, px0, py1), fill="#90a4ae", width=2)
    draw.text((px0 - 54, py0 - 8), f"{y_max:g}", font=small_font, fill="#52616b")
    draw.text((px0 - 54, py1 - 10), f"{y_min:g}", font=small_font, fill="#52616b")

    points: list[tuple[int, int]] = []
    for _, row in rows.sort_values("quartile_ordinal").iterrows():
        ordinal = int(row["quartile_ordinal"])
        x = int(px0 + (ordinal - 1) / 3 * (px1 - px0))
        y = int(py1 - (float(row["adjusted_ess"]) - y_min) / (y_max - y_min) * (py1 - py0))
        y_lo = int(py1 - (float(row["adjusted_ess_ci_lower"]) - y_min) / (y_max - y_min) * (py1 - py0))
        y_hi = int(py1 - (float(row["adjusted_ess_ci_upper"]) - y_min) / (y_max - y_min) * (py1 - py0))
        points.append((x, y))
        draw.line((x, y_hi, x, y_lo), fill=color, width=3)
        draw.line((x - 9, y_hi, x + 9, y_hi), fill=color, width=3)
        draw.line((x - 9, y_lo, x + 9, y_lo), fill=color, width=3)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
        draw_centered_text(draw, (x, y - 24), f"{float(row['adjusted_ess']):.2f}", small_font, fill="#1f2933")
        draw_centered_text(draw, (x, py1 + 26), str(row["quartile"]), label_font, fill="#52616b")
    if len(points) > 1:
        draw.line(points, fill=color, width=3)


def save_individual_quartile_figures(quartiles: pd.DataFrame) -> None:
    if not PIL_AVAILABLE:
        logging.warning("PIL unavailable; adjusted quartile figures were not created.")
        return
    colors = {
        "sleep_duration_sd": "#2f6f73",
        "sleep_efficiency_sd": "#7b4f9d",
        "waso_sd": "#c17c1c",
        "sleep_midpoint_circular_sd": "#4d74b8",
    }
    title_font = get_font(28, bold=True)
    panel_title_font = get_font(22, bold=True)
    label_font = get_font(16)
    small_font = get_font(14)
    for metric in METRICS:
        rows = quartiles.loc[quartiles["variability_metric"].eq(metric.variability_variable)].copy()
        y_min, y_max = y_bounds_for_rows(rows)
        image = Image.new("RGB", (980, 700), "#f7f9fb")
        draw = ImageDraw.Draw(image)
        draw.text((46, 32), metric.figure_title, font=title_font, fill="#1f2933")
        draw.text(
            (46, 72),
            "Marginal-standardized adjusted ESS; raw ESS means are not plotted.",
            font=label_font,
            fill="#52616b",
        )
        draw_quartile_panel(
            image,
            rows,
            (56, 125, 925, 635),
            metric,
            panel_title_font,
            label_font,
            small_font,
            colors[metric.variability_variable],
            y_min,
            y_max,
        )
        image.save(FIGURE_PATHS[metric.variability_variable])
        logging.info("Wrote figure %s", FIGURE_PATHS[metric.variability_variable])


def save_panel_figure(quartiles: pd.DataFrame) -> None:
    if not PIL_AVAILABLE:
        logging.warning("PIL unavailable; panel figure was not created.")
        return
    colors = ["#2f6f73", "#7b4f9d", "#c17c1c", "#4d74b8"]
    title_font = get_font(32, bold=True)
    subtitle_font = get_font(18)
    panel_title_font = get_font(17, bold=True)
    label_font = get_font(13)
    small_font = get_font(12)
    image = Image.new("RGB", (1360, 980), "#f7f9fb")
    draw = ImageDraw.Draw(image)
    draw.text((54, 34), "Adjusted ESS by Sleep Variability Quartile", font=title_font, fill="#1f2933")
    draw.text(
        (54, 82),
        "Marginal-standardized predictions from self-mean-adjusted OLS + HC3 models; raw ESS means are not plotted.",
        font=subtitle_font,
        fill="#52616b",
    )
    panel_bounds = [(70, 145, 635, 500), (725, 145, 1290, 500), (70, 570, 635, 925), (725, 570, 1290, 925)]
    y_min, y_max = y_bounds_for_rows(quartiles)
    for idx, metric in enumerate(METRICS):
        rows = quartiles.loc[quartiles["variability_metric"].eq(metric.variability_variable)].copy()
        draw_quartile_panel(
            image,
            rows,
            panel_bounds[idx],
            metric,
            panel_title_font,
            label_font,
            small_font,
            colors[idx],
            y_min,
            y_max,
        )
    image.save(PANEL_FIGURE_PATH)
    logging.info("Wrote figure %s", PANEL_FIGURE_PATH)


def save_quartile_figures(quartiles: pd.DataFrame) -> None:
    save_individual_quartile_figures(quartiles)
    save_panel_figure(quartiles)


def main() -> None:
    setup_logging()
    backup_existing_outputs()
    check_environment()
    primary = load_primary_dataset()

    definitions = build_variable_definitions()
    self_models, self_incremental = build_self_mean_adjusted_models(primary)
    joint_sample_flow, joint_sample = build_joint_sample_flow(primary)
    joint_coefficients, joint_fit, _, joint_design = build_joint_model(joint_sample)
    pearson, spearman, variability_pairwise, high_pairs = build_correlation_outputs(joint_sample)
    vif = build_vif(joint_design)
    quartiles = build_adjusted_quartile_results(primary)
    previous_vs_self = build_previous_vs_self(self_models)

    write_csv(definitions, VARIABLE_DEFINITIONS_PATH)
    write_csv(self_models, SELF_MODELS_PATH)
    write_csv(self_incremental, SELF_INCREMENTAL_PATH)
    write_csv(joint_sample_flow, JOINT_SAMPLE_FLOW_PATH)
    write_csv(joint_coefficients, JOINT_COEFFICIENTS_PATH)
    write_csv(joint_fit, JOINT_FIT_PATH)
    write_csv(pearson, PEARSON_CORRELATIONS_PATH)
    write_csv(spearman, SPEARMAN_CORRELATIONS_PATH)
    write_csv(variability_pairwise, VARIABILITY_PAIRWISE_CORRELATIONS_PATH)
    write_csv(vif, JOINT_VIF_PATH)
    write_csv(high_pairs, HIGH_CORRELATION_PAIRS_PATH)
    write_csv(quartiles, ADJUSTED_QUARTILE_RESULTS_PATH)
    write_csv(previous_vs_self, PREVIOUS_VS_SELF_PATH)
    save_quartile_figures(quartiles)

    print("Mean-adjusted and joint variability analysis complete")
    print(f"Primary window: {PRIMARY_FEATURE_SET}")
    print(f"Primary rows: {len(primary)}")
    print(f"Joint complete-case n: {len(joint_sample)}")
    for _, row in self_models.iterrows():
        print(
            f"{row['variability_metric']}: n={int(row['n'])}, "
            f"beta={float(row['coefficient']):.6f}, "
            f"std_beta={float(row['standardized_beta']):.6f}, "
            f"p={float(row['p_value']):.6g}, "
            f"FDR={float(row['p_value_fdr_bh']):.6g}, "
            f"delta_R2={float(row['delta_r_squared']):.6f}"
        )


if __name__ == "__main__":
    main()
