"""Compare full-sample sleep variability metrics and quartile effects.

This script extends the validated full-sample analysis without changing the
primary feature window, outcome, covariates, sample construction, QC criteria,
or raw data. Existing output files from this script are backed up before being
rewritten.
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
BACKUP_DIR = PROJECT_ROOT / "outputs" / "backups" / datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

FULL_DATASET_PATH = PROCESSED_DIR / "full_regression_dataset.csv"
EXISTING_COEFFICIENTS_PATH = OUTPUT_TABLE_DIR / "full_regression_coefficients.csv"
EXISTING_SECONDARY_PATH = OUTPUT_TABLE_DIR / "full_secondary_variability_models.csv"

METRIC_COMPARISON_PATH = OUTPUT_TABLE_DIR / "full_variability_metric_comparison.csv"
EFFECT_INTERPRETATION_PATH = OUTPUT_TABLE_DIR / "full_variability_effect_interpretation.csv"
QUARTILE_CUTPOINTS_PATH = OUTPUT_TABLE_DIR / "full_variability_quartile_cutpoints.csv"
QUARTILE_DESCRIPTIVES_PATH = OUTPUT_TABLE_DIR / "full_variability_quartile_descriptives.csv"
QUARTILE_REGRESSION_PATH = OUTPUT_TABLE_DIR / "full_variability_quartile_regression.csv"
TREND_TESTS_PATH = OUTPUT_TABLE_DIR / "full_variability_trend_tests.csv"

STANDARDIZED_BETA_FIGURE_PATH = OUTPUT_FIGURE_DIR / "full_variability_standardized_beta.png"
DELTA_R2_FIGURE_PATH = OUTPUT_FIGURE_DIR / "full_variability_delta_r2.png"
QUARTILE_ESS_FIGURE_PATH = OUTPUT_FIGURE_DIR / "full_variability_quartile_ess.png"
LOG_PATH = OUTPUT_LOG_DIR / "08_compare_variability_metrics_and_quartiles.log"

REQUESTED_OUTPUTS = [
    METRIC_COMPARISON_PATH,
    EFFECT_INTERPRETATION_PATH,
    QUARTILE_CUTPOINTS_PATH,
    QUARTILE_DESCRIPTIVES_PATH,
    QUARTILE_REGRESSION_PATH,
    TREND_TESTS_PATH,
    STANDARDIZED_BETA_FIGURE_PATH,
    DELTA_R2_FIGURE_PATH,
    QUARTILE_ESS_FIGURE_PATH,
]

PRIMARY_FEATURE_SET = "consecutive_7_daybynoon"
OUTCOME = "ess"
BASELINE_TERMS = ["sleep_duration_mean", "age", "sex_Male", "bmi", "ahi"]
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


@dataclass(frozen=True)
class MetricConfig:
    variable: str
    label: str
    model_label: str
    unit: str
    unit_definition: str
    unit_source_file: str
    unit_source_detail: str
    increments: tuple[float, ...]
    increment_unit_label: str
    is_secondary_metric: bool


@dataclass
class ModelFit:
    model_label: str
    sample: pd.DataFrame
    terms: list[str]
    robust_fit: Any
    classical_fit: Any
    design: pd.DataFrame


METRICS = [
    MetricConfig(
        variable="sleep_duration_sd",
        label="Sleep duration SD",
        model_label="A",
        unit="minutes",
        unit_definition=(
            "Sample SD of nightly sleep_duration_minutes; sleep_duration_minutes "
            "equals sleep epochs multiplied by 30-second epoch length in minutes."
        ),
        unit_source_file=(
            "src/02_validate_actigraphy.py:40,418; "
            "src/04_resolve_ids_and_validate_sleep_features.py:679-680"
        ),
        unit_source_detail="EPOCH_SECONDS=30, EPOCH_MINUTES=0.5; SD uses ddof=1.",
        increments=(10.0, 30.0, 60.0),
        increment_unit_label="minutes",
        is_secondary_metric=False,
    ),
    MetricConfig(
        variable="sleep_efficiency_sd",
        label="Sleep efficiency SD",
        model_label="B",
        unit="percentage points",
        unit_definition=(
            "Sample SD of nightly sleep_efficiency, where sleep_efficiency is "
            "sleep epochs divided by valid scored epochs times 100."
        ),
        unit_source_file=(
            "src/02_validate_actigraphy.py:420-421; "
            "src/04_resolve_ids_and_validate_sleep_features.py:693-694"
        ),
        unit_source_detail="The nightly efficiency variable is stored on a 0-100 percent scale.",
        increments=(1.0, 5.0),
        increment_unit_label="percentage points",
        is_secondary_metric=True,
    ),
    MetricConfig(
        variable="waso_sd",
        label="WASO SD",
        model_label="C",
        unit="minutes",
        unit_definition=(
            "Sample SD of nightly waso_minutes; WASO minutes are wake epochs "
            "inside the first-to-last sleep window multiplied by 30-second epoch "
            "length in minutes."
        ),
        unit_source_file=(
            "src/02_validate_actigraphy.py:386-395; "
            "src/04_resolve_ids_and_validate_sleep_features.py:695-696"
        ),
        unit_source_detail="WASO is calculated in minutes before participant-level SD.",
        increments=(10.0, 30.0),
        increment_unit_label="minutes",
        is_secondary_metric=True,
    ),
    MetricConfig(
        variable="sleep_midpoint_circular_sd",
        label="Sleep midpoint circular SD",
        model_label="D",
        unit="circular minutes on a 24-hour clock",
        unit_definition=(
            "Circular SD of nightly sleep_midpoint_clock_minutes on the 24-hour "
            "clock; radians are converted back to minutes by sd_radians / "
            "(2*pi) * 1440."
        ),
        unit_source_file=(
            "src/04_resolve_ids_and_validate_sleep_features.py:657-674,697-698"
        ),
        unit_source_detail=(
            "Existing circular dispersion is retained; ordinary clock-time SD is not recalculated."
        ),
        increments=(10.0, 30.0, 60.0),
        increment_unit_label="circular minutes",
        is_secondary_metric=True,
    ),
]


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
        logging.info("No requested outputs from script 08 existed before this run.")
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
    global sm, statsmodels_module
    try:
        statsmodels_module = importlib.import_module("statsmodels")
        sm = importlib.import_module("statsmodels.api")
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
        raise FileNotFoundError(
            f"Required full regression dataset was not found: {FULL_DATASET_PATH}"
        )
    df = read_csv_with_fallback(FULL_DATASET_PATH)
    if "feature_set" not in df.columns:
        raise RuntimeError("full_regression_dataset.csv does not contain feature_set.")
    primary = df.loc[df["feature_set"].eq(PRIMARY_FEATURE_SET)].copy()
    required_columns = [OUTCOME, *BASELINE_TERMS, *[metric.variable for metric in METRICS]]
    missing = [column for column in required_columns if column not in primary.columns]
    if missing:
        raise RuntimeError(f"Missing required analysis columns: {missing}")
    for column in required_columns:
        primary[column] = numeric_series(primary[column])
    logging.info("Loaded %s primary-window rows.", len(primary))
    return primary


def design_matrix(sample: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    x = pd.DataFrame(index=sample.index)
    for term in terms:
        x[term] = numeric_series(sample[term]).astype(float)
    return sm.add_constant(x, has_constant="add")


def fit_model(model_label: str, sample: pd.DataFrame, terms: list[str]) -> ModelFit:
    y = numeric_series(sample[OUTCOME]).astype(float)
    x = design_matrix(sample, terms)
    classical_fit = sm.OLS(y, x).fit()
    robust_fit = sm.OLS(y, x).fit(cov_type="HC3")
    return ModelFit(
        model_label=model_label,
        sample=sample,
        terms=terms,
        robust_fit=robust_fit,
        classical_fit=classical_fit,
        design=x,
    )


def conf_int_row(fit: Any, term: str) -> tuple[float, float]:
    conf = fit.conf_int(alpha=0.05)
    if isinstance(conf, pd.DataFrame):
        return float(conf.loc[term, 0]), float(conf.loc[term, 1])
    term_index = list(fit.params.index).index(term)
    return float(conf[term_index, 0]), float(conf[term_index, 1])


def std_factor(sample: pd.DataFrame, variable: str) -> float:
    x_sd = numeric_series(sample[variable]).std(ddof=1)
    y_sd = numeric_series(sample[OUTCOME]).std(ddof=1)
    if pd.isna(x_sd) or pd.isna(y_sd) or y_sd == 0:
        return float("nan")
    return float(x_sd / y_sd)


def metric_terms(metric: MetricConfig) -> list[str]:
    return ["sleep_duration_mean", metric.variable, "age", "sex_Male", "bmi", "ahi"]


def load_existing_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficients = (
        read_csv_with_fallback(EXISTING_COEFFICIENTS_PATH)
        if EXISTING_COEFFICIENTS_PATH.exists()
        else pd.DataFrame()
    )
    secondary = (
        read_csv_with_fallback(EXISTING_SECONDARY_PATH)
        if EXISTING_SECONDARY_PATH.exists()
        else pd.DataFrame()
    )
    return coefficients, secondary


def existing_metric_result(
    metric: MetricConfig,
    coefficients: pd.DataFrame,
    secondary: pd.DataFrame,
) -> dict[str, Any]:
    if metric.variable == "sleep_duration_sd":
        table = coefficients
        mask = (
            table.get("model", pd.Series(dtype=str)).astype(str).eq("Model 2")
            & table.get("term", pd.Series(dtype=str)).astype(str).eq(metric.variable)
        )
        source = relative_path(EXISTING_COEFFICIENTS_PATH)
    else:
        table = secondary
        mask = (
            table.get("model", pd.Series(dtype=str)).astype(str).eq(f"secondary_{metric.variable}")
            & table.get("term", pd.Series(dtype=str)).astype(str).eq(metric.variable)
        )
        source = relative_path(EXISTING_SECONDARY_PATH)

    if table.empty or not mask.any():
        return {
            "existing_result_found": False,
            "existing_source_file": source,
            "existing_n": np.nan,
            "existing_coefficient": np.nan,
            "existing_hc3_standard_error": np.nan,
            "existing_ci_lower": np.nan,
            "existing_ci_upper": np.nan,
            "existing_p_value": np.nan,
        }

    row = table.loc[mask].iloc[0]
    return {
        "existing_result_found": True,
        "existing_source_file": source,
        "existing_n": int(row["n"]),
        "existing_coefficient": float(row["coefficient"]),
        "existing_hc3_standard_error": float(row["hc3_standard_error"]),
        "existing_ci_lower": float(row["ci_lower"]),
        "existing_ci_upper": float(row["ci_upper"]),
        "existing_p_value": float(row["p_value"]),
    }


def is_close_to_existing(row: dict[str, Any]) -> bool | float:
    if not row["existing_result_found"]:
        return np.nan
    checks = [
        int(row["n"]) == int(row["existing_n"]),
        math.isclose(float(row["coefficient"]), float(row["existing_coefficient"]), rel_tol=1e-9, abs_tol=1e-9),
        math.isclose(
            float(row["hc3_standard_error"]),
            float(row["existing_hc3_standard_error"]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ),
        math.isclose(float(row["p_value"]), float(row["existing_p_value"]), rel_tol=1e-9, abs_tol=1e-9),
    ]
    return bool(all(checks))


def build_metric_comparison(primary: pd.DataFrame) -> pd.DataFrame:
    coefficients, secondary = load_existing_results()
    common_required = [OUTCOME, *BASELINE_TERMS, *[metric.variable for metric in METRICS]]
    common_sample = primary.dropna(subset=common_required).copy()
    logging.info("Common complete-case sample across all variability metrics: %s", len(common_sample))

    rows: list[dict[str, Any]] = []
    for analysis_sample_type in ["common_all_metrics", "metric_specific_available"]:
        for metric in METRICS:
            if analysis_sample_type == "common_all_metrics":
                sample = common_sample.copy()
                sample_definition = (
                    "Complete case for ESS, sleep duration mean, age, sex, BMI, AHI, "
                    "and all four variability metrics."
                )
            else:
                required = [OUTCOME, *BASELINE_TERMS, metric.variable]
                sample = primary.dropna(subset=required).copy()
                sample_definition = (
                    "Complete case for ESS, sleep duration mean, age, sex, BMI, AHI, "
                    f"and {metric.variable}."
                )

            baseline = fit_model("Baseline", sample, BASELINE_TERMS)
            extended = fit_model(f"Extended {metric.model_label}", sample, metric_terms(metric))
            robust = extended.robust_fit
            ci_lower, ci_upper = conf_int_row(robust, metric.variable)
            scale = std_factor(sample, metric.variable)
            existing = existing_metric_result(metric, coefficients, secondary)

            row: dict[str, Any] = {
                "analysis_sample_type": analysis_sample_type,
                "sample_definition": sample_definition,
                "primary_window": PRIMARY_FEATURE_SET,
                "outcome": OUTCOME,
                "baseline_formula": "ESS ~ sleep_duration_mean + age + sex + BMI + AHI",
                "extended_formula": (
                    "ESS ~ sleep_duration_mean + "
                    f"{metric.variable} + age + sex + BMI + AHI"
                ),
                "model_label": metric.model_label,
                "variability_metric": metric.variable,
                "variability_metric_label": metric.label,
                "unit": metric.unit,
                "unit_definition": metric.unit_definition,
                "unit_source_file": metric.unit_source_file,
                "unit_source_detail": metric.unit_source_detail,
                "is_secondary_metric": metric.is_secondary_metric,
                "n": int(extended.classical_fit.nobs),
                "coefficient": float(robust.params[metric.variable]),
                "hc3_standard_error": float(robust.bse[metric.variable]),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_value": float(robust.pvalues[metric.variable]),
                "standardized_beta": float(robust.params[metric.variable] * scale),
                "standardized_beta_ci_lower": float(ci_lower * scale),
                "standardized_beta_ci_upper": float(ci_upper * scale),
                "r_squared": float(extended.classical_fit.rsquared),
                "adjusted_r_squared": float(extended.classical_fit.rsquared_adj),
                "baseline_r_squared_same_sample": float(baseline.classical_fit.rsquared),
                "baseline_adjusted_r_squared_same_sample": float(baseline.classical_fit.rsquared_adj),
                "delta_r_squared": float(
                    extended.classical_fit.rsquared - baseline.classical_fit.rsquared
                ),
                "delta_adjusted_r_squared": float(
                    extended.classical_fit.rsquared_adj - baseline.classical_fit.rsquared_adj
                ),
                "aic": float(extended.classical_fit.aic),
                "bic": float(extended.classical_fit.bic),
                "baseline_aic_same_sample": float(baseline.classical_fit.aic),
                "baseline_bic_same_sample": float(baseline.classical_fit.bic),
                "reference_distribution": (
                    "t" if bool(getattr(robust, "use_t", False)) else "normal_z"
                ),
                "statsmodels_use_t": bool(getattr(robust, "use_t", False)),
                "standardized_beta_method": (
                    "raw HC3 coefficient multiplied by predictor SD / ESS SD on the same sample"
                ),
            }
            row.update(existing)
            row["matches_existing_result"] = is_close_to_existing(row)
            rows.append(row)

    output = pd.DataFrame(rows)
    output["p_value_fdr_bh_all_four"] = np.nan
    output["p_value_fdr_bh_secondary_only"] = np.nan
    for sample_type, group in output.groupby("analysis_sample_type", sort=False):
        output.loc[group.index, "p_value_fdr_bh_all_four"] = bh_fdr(group["p_value"].tolist())
        secondary_group = group.loc[group["is_secondary_metric"].astype(bool)]
        output.loc[secondary_group.index, "p_value_fdr_bh_secondary_only"] = bh_fdr(
            secondary_group["p_value"].tolist()
        )
        logging.info(
            "Analysis sample %s p-values: %s",
            sample_type,
            dict(zip(group["variability_metric"], group["p_value"])),
        )
    return output


def build_effect_interpretation(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    config_by_var = {metric.variable: metric for metric in METRICS}
    for _, model_row in comparison.iterrows():
        metric = config_by_var[str(model_row["variability_metric"])]
        for increment in metric.increments:
            rows.append(
                {
                    "analysis_sample_type": model_row["analysis_sample_type"],
                    "primary_window": PRIMARY_FEATURE_SET,
                    "variability_metric": metric.variable,
                    "variability_metric_label": metric.label,
                    "unit": metric.unit,
                    "unit_definition": metric.unit_definition,
                    "unit_source_file": metric.unit_source_file,
                    "unit_source_detail": metric.unit_source_detail,
                    "increment": increment,
                    "increment_unit_label": metric.increment_unit_label,
                    "increment_description": f"{increment:g} {metric.increment_unit_label}",
                    "ess_change_for_increment": float(model_row["coefficient"] * increment),
                    "ess_change_ci_lower_for_increment": float(model_row["ci_lower"] * increment),
                    "ess_change_ci_upper_for_increment": float(model_row["ci_upper"] * increment),
                    "p_value": float(model_row["p_value"]),
                    "p_value_fdr_bh_all_four": float(model_row["p_value_fdr_bh_all_four"]),
                    "p_value_fdr_bh_secondary_only": (
                        float(model_row["p_value_fdr_bh_secondary_only"])
                        if pd.notna(model_row["p_value_fdr_bh_secondary_only"])
                        else np.nan
                    ),
                    "interpretation_note": (
                        "Association from observational OLS + HC3 model; not causal."
                    ),
                }
            )
    return pd.DataFrame(rows)


def quartile_assignments(sample: pd.DataFrame, metric: MetricConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = sample.copy()
    values = numeric_series(work[metric.variable]).astype(float)
    tied_at_requested_cutpoints = values.quantile([0.25, 0.5, 0.75]).duplicated().any()
    try:
        quartiles = pd.qcut(values, q=4, labels=QUARTILE_LABELS, duplicates="raise")
        method = "pd.qcut(metric, q=4, duplicates='raise')"
        failure_reason = ""
    except ValueError as exc:
        ranks = values.rank(method="first")
        quartiles = pd.qcut(ranks, q=4, labels=QUARTILE_LABELS, duplicates="raise")
        method = "pd.qcut(rank(method='first'), q=4) after tied cutpoint failure"
        failure_reason = str(exc)

    work["variability_quartile"] = quartiles.astype(str)
    work["variability_quartile_ordinal"] = work["variability_quartile"].map(
        {label: idx + 1 for idx, label in enumerate(QUARTILE_LABELS)}
    )
    quantiles = values.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    metadata = {
        "cutpoint_method": method,
        "qcut_failure_reason": failure_reason,
        "n_unique_values": int(values.nunique(dropna=True)),
        "tied_at_requested_cutpoints": bool(tied_at_requested_cutpoints),
        "minimum": float(quantiles.loc[0.0]),
        "q1_q2_boundary_25th_percentile": float(quantiles.loc[0.25]),
        "q2_q3_boundary_50th_percentile": float(quantiles.loc[0.5]),
        "q3_q4_boundary_75th_percentile": float(quantiles.loc[0.75]),
        "maximum": float(quantiles.loc[1.0]),
    }
    return work, metadata


def build_quartile_analyses(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common_required = [OUTCOME, *BASELINE_TERMS, *[metric.variable for metric in METRICS]]
    common_sample = primary.dropna(subset=common_required).copy()
    cutpoint_rows: list[dict[str, Any]] = []
    descriptive_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    trend_rows: list[dict[str, Any]] = []

    for metric in METRICS:
        sample, metadata = quartile_assignments(common_sample, metric)
        cutpoint_rows.append(
            {
                "analysis_sample_type": "common_all_metrics",
                "primary_window": PRIMARY_FEATURE_SET,
                "variability_metric": metric.variable,
                "variability_metric_label": metric.label,
                "unit": metric.unit,
                "n": int(len(sample)),
                **metadata,
                "notes": (
                    "Quartiles are based on the common complete-case sample used for "
                    "cross-metric comparison."
                ),
            }
        )

        for label in QUARTILE_LABELS:
            group = sample.loc[sample["variability_quartile"].eq(label)].copy()
            descriptive_rows.append(
                {
                    "analysis_sample_type": "common_all_metrics",
                    "primary_window": PRIMARY_FEATURE_SET,
                    "variability_metric": metric.variable,
                    "variability_metric_label": metric.label,
                    "unit": metric.unit,
                    "quartile": label,
                    "quartile_ordinal": QUARTILE_LABELS.index(label) + 1,
                    "n": int(len(group)),
                    "metric_min": float(numeric_series(group[metric.variable]).min()),
                    "metric_max": float(numeric_series(group[metric.variable]).max()),
                    "metric_mean": float(numeric_series(group[metric.variable]).mean()),
                    "metric_sd": float(numeric_series(group[metric.variable]).std(ddof=1)),
                    "metric_median": float(numeric_series(group[metric.variable]).median()),
                    "ess_mean": float(numeric_series(group[OUTCOME]).mean()),
                    "ess_sd": float(numeric_series(group[OUTCOME]).std(ddof=1)),
                    "ess_median": float(numeric_series(group[OUTCOME]).median()),
                    "cutpoint_method": metadata["cutpoint_method"],
                }
            )

        dummies = pd.get_dummies(sample["variability_quartile"], prefix="quartile")
        for label in QUARTILE_LABELS[1:]:
            column = f"quartile_{label}"
            if column not in dummies.columns:
                dummies[column] = 0
        x = pd.DataFrame(index=sample.index)
        for label in QUARTILE_LABELS[1:]:
            x[f"{label}_vs_Q1"] = dummies[f"quartile_{label}"].astype(float)
        for term in BASELINE_TERMS:
            x[term] = numeric_series(sample[term]).astype(float)
        x = sm.add_constant(x, has_constant="add")
        y = numeric_series(sample[OUTCOME]).astype(float)
        classical = sm.OLS(y, x).fit()
        robust = sm.OLS(y, x).fit(cov_type="HC3")
        conf = robust.conf_int(alpha=0.05)
        for label in QUARTILE_LABELS[1:]:
            term = f"{label}_vs_Q1"
            regression_rows.append(
                {
                    "analysis_sample_type": "common_all_metrics",
                    "primary_window": PRIMARY_FEATURE_SET,
                    "variability_metric": metric.variable,
                    "variability_metric_label": metric.label,
                    "unit": metric.unit,
                    "comparison": term,
                    "reference_quartile": "Q1",
                    "n": int(classical.nobs),
                    "coefficient": float(robust.params[term]),
                    "hc3_standard_error": float(robust.bse[term]),
                    "ci_lower": float(conf.loc[term, 0]),
                    "ci_upper": float(conf.loc[term, 1]),
                    "p_value": float(robust.pvalues[term]),
                    "r_squared": float(classical.rsquared),
                    "adjusted_r_squared": float(classical.rsquared_adj),
                    "aic": float(classical.aic),
                    "bic": float(classical.bic),
                    "cutpoint_method": metadata["cutpoint_method"],
                }
            )

        trend_x = pd.DataFrame(index=sample.index)
        trend_x["quartile_ordinal"] = numeric_series(sample["variability_quartile_ordinal"]).astype(float)
        for term in BASELINE_TERMS:
            trend_x[term] = numeric_series(sample[term]).astype(float)
        trend_x = sm.add_constant(trend_x, has_constant="add")
        trend_classical = sm.OLS(y, trend_x).fit()
        trend_robust = sm.OLS(y, trend_x).fit(cov_type="HC3")
        trend_conf = trend_robust.conf_int(alpha=0.05)
        trend_rows.append(
            {
                "analysis_sample_type": "common_all_metrics",
                "primary_window": PRIMARY_FEATURE_SET,
                "variability_metric": metric.variable,
                "variability_metric_label": metric.label,
                "unit": metric.unit,
                "n": int(trend_classical.nobs),
                "trend_term": "quartile_ordinal",
                "coefficient_per_quartile": float(trend_robust.params["quartile_ordinal"]),
                "hc3_standard_error": float(trend_robust.bse["quartile_ordinal"]),
                "ci_lower": float(trend_conf.loc["quartile_ordinal", 0]),
                "ci_upper": float(trend_conf.loc["quartile_ordinal", 1]),
                "p_for_linear_trend": float(trend_robust.pvalues["quartile_ordinal"]),
                "r_squared": float(trend_classical.rsquared),
                "adjusted_r_squared": float(trend_classical.rsquared_adj),
                "aic": float(trend_classical.aic),
                "bic": float(trend_classical.bic),
                "cutpoint_method": metadata["cutpoint_method"],
            }
        )

    return (
        pd.DataFrame(cutpoint_rows),
        pd.DataFrame(descriptive_rows),
        pd.DataFrame(regression_rows),
        pd.DataFrame(trend_rows),
    )


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


def save_bar_figure(
    data: pd.DataFrame,
    value_column: str,
    output_path: Path,
    title: str,
    subtitle: str,
    value_formatter: str,
) -> None:
    if not PIL_AVAILABLE:
        logging.warning("PIL is unavailable; skipping %s", output_path)
        return

    rows = data.loc[data["analysis_sample_type"].eq("common_all_metrics")].copy()
    rows["order"] = rows["variability_metric"].map(
        {metric.variable: idx for idx, metric in enumerate(METRICS)}
    )
    rows = rows.sort_values("order")

    width, height = 1240, 760
    image = Image.new("RGB", (width, height), "#f7f9fb")
    draw = ImageDraw.Draw(image)
    title_font = get_font(34, bold=True)
    subtitle_font = get_font(20)
    label_font = get_font(18)
    small_font = get_font(15)
    draw.text((60, 34), title, font=title_font, fill="#1f2933")
    draw.text((60, 82), subtitle, font=subtitle_font, fill="#52616b")

    plot_left, plot_top, plot_right, plot_bottom = 330, 150, 1140, 640
    zero = 0.0
    values = rows[value_column].astype(float).to_numpy()
    span = max(abs(np.nanmin(values)), abs(np.nanmax(values)), 0.001)
    xmin, xmax = -span * 1.15, span * 1.15
    zero_x = int(plot_left + (zero - xmin) / (xmax - xmin) * (plot_right - plot_left))

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#b0bec5", width=2)
    draw.line((zero_x, plot_top, zero_x, plot_bottom), fill="#78909c", width=2)

    colors = ["#2f6f73", "#7b4f9d", "#c17c1c", "#4d74b8"]
    bar_h = 58
    gap = 48
    for i, (_, row) in enumerate(rows.iterrows()):
        y = plot_top + i * (bar_h + gap) + 12
        value = float(row[value_column])
        end_x = int(plot_left + (value - xmin) / (xmax - xmin) * (plot_right - plot_left))
        x0, x1 = sorted([zero_x, end_x])
        draw.rounded_rectangle((x0, y, x1, y + bar_h), radius=8, fill=colors[i])
        draw.text((60, y + 6), str(row["variability_metric_label"]), font=label_font, fill="#1f2933")
        draw.text(
            (60, y + 34),
            f"p={float(row['p_value']):.3g}, FDR={float(row['p_value_fdr_bh_all_four']):.3g}",
            font=small_font,
            fill="#52616b",
        )
        label = value_formatter.format(value)
        label_x = end_x + 12 if value >= 0 else end_x - 120
        draw.text((label_x, y + 16), label, font=label_font, fill="#1f2933")

    draw.text((plot_left, plot_bottom + 34), f"{xmin:.3f}", font=small_font, fill="#52616b")
    draw_centered_text(draw, (zero_x, plot_bottom + 43), "0", small_font, fill="#52616b")
    draw.text((plot_right - 65, plot_bottom + 34), f"{xmax:.3f}", font=small_font, fill="#52616b")
    image.save(output_path)
    logging.info("Wrote figure %s", output_path)


def save_quartile_ess_figure(descriptives: pd.DataFrame, output_path: Path) -> None:
    if not PIL_AVAILABLE:
        logging.warning("PIL is unavailable; skipping %s", output_path)
        return

    rows = descriptives.loc[descriptives["analysis_sample_type"].eq("common_all_metrics")].copy()
    width, height = 1280, 900
    image = Image.new("RGB", (width, height), "#f7f9fb")
    draw = ImageDraw.Draw(image)
    title_font = get_font(34, bold=True)
    subtitle_font = get_font(20)
    label_font = get_font(17)
    small_font = get_font(14)
    draw.text((54, 34), "ESS by Variability Quartile", font=title_font, fill="#1f2933")
    draw.text(
        (54, 82),
        "Unadjusted ESS summaries by quartile on the common complete-case sample",
        font=subtitle_font,
        fill="#52616b",
    )

    panel_w, panel_h = 550, 310
    panel_positions = [(70, 150), (680, 150), (70, 520), (680, 520)]
    colors = ["#2f6f73", "#7b4f9d", "#c17c1c", "#4d74b8"]
    y_min = math.floor(float(rows["ess_mean"].min()) - 0.5)
    y_max = math.ceil(float(rows["ess_mean"].max()) + 0.5)
    y_min = min(y_min, 5)
    y_max = max(y_max, 9)

    for idx, metric in enumerate(METRICS):
        px, py = panel_positions[idx]
        panel = rows.loc[rows["variability_metric"].eq(metric.variable)].sort_values("quartile_ordinal")
        draw.rectangle((px, py, px + panel_w, py + panel_h), fill="#ffffff", outline="#d9e2ec")
        draw.text((px + 20, py + 16), metric.label, font=label_font, fill="#1f2933")
        left, top, right, bottom = px + 58, py + 62, px + panel_w - 34, py + panel_h - 54
        draw.line((left, bottom, right, bottom), fill="#90a4ae", width=2)
        draw.line((left, top, left, bottom), fill="#90a4ae", width=2)

        points: list[tuple[int, int]] = []
        for _, row in panel.iterrows():
            x = int(left + (float(row["quartile_ordinal"]) - 1) / 3 * (right - left))
            y = int(bottom - (float(row["ess_mean"]) - y_min) / (y_max - y_min) * (bottom - top))
            points.append((x, y))
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=colors[idx])
            draw_centered_text(draw, (x, bottom + 22), str(row["quartile"]), small_font, fill="#52616b")
            draw_centered_text(draw, (x, y - 22), f"{float(row['ess_mean']):.2f}", small_font, fill="#1f2933")
        if len(points) > 1:
            draw.line(points, fill=colors[idx], width=3)
        draw.text((left - 46, top - 6), f"{y_max:g}", font=small_font, fill="#52616b")
        draw.text((left - 46, bottom - 10), f"{y_min:g}", font=small_font, fill="#52616b")
    image.save(output_path)
    logging.info("Wrote figure %s", output_path)


def save_figures(comparison: pd.DataFrame, descriptives: pd.DataFrame) -> None:
    save_bar_figure(
        comparison,
        "standardized_beta",
        STANDARDIZED_BETA_FIGURE_PATH,
        "Standardized Beta for Variability Metrics",
        "OLS + HC3; ESS and variability metric scaled by their sample SD",
        "{:.3f}",
    )
    save_bar_figure(
        comparison,
        "delta_r_squared",
        DELTA_R2_FIGURE_PATH,
        "Delta R-squared vs Baseline",
        "Baseline: ESS ~ sleep duration mean + age + sex + BMI + AHI",
        "{:.4f}",
    )
    save_quartile_ess_figure(descriptives, QUARTILE_ESS_FIGURE_PATH)


def main() -> None:
    setup_logging()
    backup_existing_outputs()
    check_environment()
    primary = load_primary_dataset()
    comparison = build_metric_comparison(primary)
    effect_interpretation = build_effect_interpretation(comparison)
    cutpoints, descriptives, quartile_regression, trend_tests = build_quartile_analyses(primary)

    write_csv(comparison, METRIC_COMPARISON_PATH)
    write_csv(effect_interpretation, EFFECT_INTERPRETATION_PATH)
    write_csv(cutpoints, QUARTILE_CUTPOINTS_PATH)
    write_csv(descriptives, QUARTILE_DESCRIPTIVES_PATH)
    write_csv(quartile_regression, QUARTILE_REGRESSION_PATH)
    write_csv(trend_tests, TREND_TESTS_PATH)
    save_figures(comparison, descriptives)

    common = comparison.loc[comparison["analysis_sample_type"].eq("common_all_metrics")]
    print("Variability metric comparison complete")
    print(f"Primary window: {PRIMARY_FEATURE_SET}")
    print(f"Common complete-case n: {int(common['n'].iloc[0]) if not common.empty else 0}")
    for _, row in common.iterrows():
        print(
            f"{row['variability_metric']}: beta={float(row['coefficient']):.6f}, "
            f"std_beta={float(row['standardized_beta']):.6f}, "
            f"p={float(row['p_value']):.6g}, "
            f"delta_R2={float(row['delta_r_squared']):.6f}, "
            f"existing_match={row['matches_existing_result']}"
        )


if __name__ == "__main__":
    main()
