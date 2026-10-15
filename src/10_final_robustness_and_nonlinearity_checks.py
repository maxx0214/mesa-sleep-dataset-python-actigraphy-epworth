"""Final robustness and nonlinearity checks for sleep duration SD.

This script creates new outputs only. It reproduces the script 09 joint model,
then performs the requested VIF sensitivity, categorical quartile, and
quadratic nonlinearity checks without changing the validated sample, QC rule,
feature window, coding, raw data, or prior outputs.
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
PREVIOUS_JOINT_COEFFICIENTS_PATH = OUTPUT_TABLE_DIR / "joint_variability_model_coefficients.csv"
PREVIOUS_JOINT_FIT_PATH = OUTPUT_TABLE_DIR / "joint_variability_model_fit.csv"
PREVIOUS_JOINT_VIF_PATH = OUTPUT_TABLE_DIR / "joint_variability_vif.csv"

VIF_SENSITIVITY_SUMMARY_PATH = OUTPUT_TABLE_DIR / "joint_model_vif_sensitivity.csv"
VIF_SENSITIVITY_COEFFICIENTS_PATH = OUTPUT_TABLE_DIR / "joint_model_vif_sensitivity_coefficients.csv"
VIF_SENSITIVITY_FIT_PATH = OUTPUT_TABLE_DIR / "joint_model_vif_sensitivity_fit.csv"
VIF_SENSITIVITY_VIF_PATH = OUTPUT_TABLE_DIR / "joint_model_vif_sensitivity_vif.csv"
QUARTILE_CATEGORICAL_PATH = OUTPUT_TABLE_DIR / "sleep_duration_sd_quartile_categorical.csv"
NONLINEARITY_PATH = OUTPUT_TABLE_DIR / "sleep_duration_sd_nonlinearity.csv"

QUARTILE_FIGURE_PATH = OUTPUT_FIGURE_DIR / "sleep_duration_sd_adjusted_quartile_final.png"
RELATIONSHIP_FIGURE_PATH = OUTPUT_FIGURE_DIR / "sleep_duration_sd_adjusted_relationship.png"
LOG_PATH = OUTPUT_LOG_DIR / "10_final_robustness_and_nonlinearity_checks.log"

REQUESTED_OUTPUTS = [
    VIF_SENSITIVITY_SUMMARY_PATH,
    VIF_SENSITIVITY_COEFFICIENTS_PATH,
    VIF_SENSITIVITY_FIT_PATH,
    VIF_SENSITIVITY_VIF_PATH,
    QUARTILE_CATEGORICAL_PATH,
    NONLINEARITY_PATH,
    QUARTILE_FIGURE_PATH,
    RELATIONSHIP_FIGURE_PATH,
]

PRIMARY_FEATURE_SET = "consecutive_7_daybynoon"
OUTCOME = "ess"
COVARIATES = ["age", "sex_Male", "bmi", "ahi"]
MEAN_VARIABLES = [
    "sleep_duration_mean",
    "sleep_efficiency_mean",
    "waso_mean",
    "sleep_midpoint_circular_mean",
]
VARIABILITY_VARIABLES = [
    "sleep_duration_sd",
    "sleep_efficiency_sd",
    "waso_sd",
    "sleep_midpoint_circular_sd",
]
REFERENCE_JOINT_TERMS = [*MEAN_VARIABLES, *VARIABILITY_VARIABLES, *COVARIATES]
JOINT_REQUIRED = [OUTCOME, *REFERENCE_JOINT_TERMS]
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
class ModelSpec:
    model_id: str
    model_label: str
    terms: list[str]
    omitted_mean_variable: str


MODEL_SPECS = [
    ModelSpec(
        model_id="reference_joint_model",
        model_label="Reference joint model from script 09",
        terms=REFERENCE_JOINT_TERMS,
        omitted_mean_variable="none",
    ),
    ModelSpec(
        model_id="exclude_mean_sleep_efficiency",
        model_label="Joint model A: mean sleep efficiency excluded",
        terms=[
            "sleep_duration_mean",
            "waso_mean",
            "sleep_midpoint_circular_mean",
            *VARIABILITY_VARIABLES,
            *COVARIATES,
        ],
        omitted_mean_variable="sleep_efficiency_mean",
    ),
    ModelSpec(
        model_id="exclude_mean_waso",
        model_label="Joint model B: mean WASO excluded",
        terms=[
            "sleep_duration_mean",
            "sleep_efficiency_mean",
            "sleep_midpoint_circular_mean",
            *VARIABILITY_VARIABLES,
            *COVARIATES,
        ],
        omitted_mean_variable="waso_mean",
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
        logging.info("No requested outputs from script 10 existed before this run.")
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


def load_joint_complete_case() -> pd.DataFrame:
    if not FULL_DATASET_PATH.exists():
        raise FileNotFoundError(f"Required input does not exist: {FULL_DATASET_PATH}")
    df = read_csv_with_fallback(FULL_DATASET_PATH)
    primary = df.loc[df["feature_set"].eq(PRIMARY_FEATURE_SET)].copy()
    missing = [column for column in JOINT_REQUIRED if column not in primary.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in full_regression_dataset.csv: {missing}")
    for column in JOINT_REQUIRED:
        primary[column] = numeric_series(primary[column])
    joint = primary.dropna(subset=JOINT_REQUIRED).copy()
    logging.info("Primary rows: %s; joint complete-case rows: %s", len(primary), len(joint))
    return joint


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


def vif_table(design: pd.DataFrame, model_id: str, model_label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    values = design.astype(float).to_numpy()
    for idx, term in enumerate(design.columns):
        if term == "const":
            continue
        vif = float(variance_inflation_factor(values, idx))
        rows.append(
            {
                "model_id": model_id,
                "model_label": model_label,
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
                "notes": "Intercept excluded from VIF reporting.",
            }
        )
    return pd.DataFrame(rows).sort_values(["model_id", "vif"], ascending=[True, False])


def close_enough(a: float, b: float, tolerance: float = 1e-8) -> bool:
    return bool(math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance))


def previous_value(path: Path, filter_column: str, filter_value: str, value_column: str) -> float:
    if not path.exists():
        return float("nan")
    table = read_csv_with_fallback(path)
    match = table.loc[table[filter_column].astype(str).eq(filter_value)]
    if match.empty:
        return float("nan")
    return float(match.iloc[0][value_column])


def validate_reference_reproduction(
    coefficients: pd.DataFrame,
    fit: pd.DataFrame,
    vif: pd.DataFrame,
) -> dict[str, Any]:
    sd_row = coefficients.loc[coefficients["term"].eq("sleep_duration_sd")].iloc[0]
    expected_beta = previous_value(
        PREVIOUS_JOINT_COEFFICIENTS_PATH, "term", "sleep_duration_sd", "coefficient"
    )
    expected_std_beta = previous_value(
        PREVIOUS_JOINT_COEFFICIENTS_PATH, "term", "sleep_duration_sd", "standardized_beta"
    )
    expected_p = previous_value(
        PREVIOUS_JOINT_COEFFICIENTS_PATH, "term", "sleep_duration_sd", "p_value"
    )
    expected_sd_vif = previous_value(PREVIOUS_JOINT_VIF_PATH, "term", "sleep_duration_sd", "vif")
    expected_waso_mean_vif = previous_value(PREVIOUS_JOINT_VIF_PATH, "term", "waso_mean", "vif")
    expected_eff_mean_vif = previous_value(
        PREVIOUS_JOINT_VIF_PATH, "term", "sleep_efficiency_mean", "vif"
    )
    current_sd_vif = float(vif.loc[vif["term"].eq("sleep_duration_sd"), "vif"].iloc[0])
    current_waso_mean_vif = float(vif.loc[vif["term"].eq("waso_mean"), "vif"].iloc[0])
    current_eff_mean_vif = float(vif.loc[vif["term"].eq("sleep_efficiency_mean"), "vif"].iloc[0])
    output = {
        "reference_reproduced_sleep_duration_sd_beta": close_enough(
            float(sd_row["coefficient"]), expected_beta
        ),
        "reference_reproduced_sleep_duration_sd_standardized_beta": close_enough(
            float(sd_row["standardized_beta"]), expected_std_beta
        ),
        "reference_reproduced_sleep_duration_sd_p_value": close_enough(
            float(sd_row["p_value"]), expected_p
        ),
        "reference_reproduced_sleep_duration_sd_vif": close_enough(current_sd_vif, expected_sd_vif),
        "reference_reproduced_waso_mean_vif": close_enough(
            current_waso_mean_vif, expected_waso_mean_vif
        ),
        "reference_reproduced_sleep_efficiency_mean_vif": close_enough(
            current_eff_mean_vif, expected_eff_mean_vif
        ),
        "previous_joint_coefficients_source": relative_path(PREVIOUS_JOINT_COEFFICIENTS_PATH),
        "previous_joint_vif_source": relative_path(PREVIOUS_JOINT_VIF_PATH),
        "previous_sleep_duration_sd_beta": expected_beta,
        "current_sleep_duration_sd_beta": float(sd_row["coefficient"]),
        "previous_sleep_duration_sd_standardized_beta": expected_std_beta,
        "current_sleep_duration_sd_standardized_beta": float(sd_row["standardized_beta"]),
        "previous_sleep_duration_sd_p_value": expected_p,
        "current_sleep_duration_sd_p_value": float(sd_row["p_value"]),
        "previous_sleep_duration_sd_vif": expected_sd_vif,
        "current_sleep_duration_sd_vif": current_sd_vif,
        "previous_waso_mean_vif": expected_waso_mean_vif,
        "current_waso_mean_vif": current_waso_mean_vif,
        "previous_sleep_efficiency_mean_vif": expected_eff_mean_vif,
        "current_sleep_efficiency_mean_vif": current_eff_mean_vif,
        "reference_model_r_squared": float(fit["r_squared"].iloc[0]),
        "reference_model_adjusted_r_squared": float(fit["adjusted_r_squared"].iloc[0]),
    }
    if not all(
        bool(value)
        for key, value in output.items()
        if key.startswith("reference_reproduced_")
    ):
        raise RuntimeError(f"Reference joint model did not reproduce script 09 outputs: {output}")
    return output


def model_coefficients(
    sample: pd.DataFrame,
    spec: ModelSpec,
    classical: Any,
    robust: Any,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for term in VARIABILITY_VARIABLES:
        ci_lower, ci_upper = conf_int_values(robust, term)
        beta = float(robust.params[term])
        rows.append(
            {
                "model_id": spec.model_id,
                "model_label": spec.model_label,
                "omitted_mean_variable": spec.omitted_mean_variable,
                "term": term,
                "term_role": "variability",
                "n": int(classical.nobs),
                "coefficient": beta,
                "hc3_standard_error": float(robust.bse[term]),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_value": float(robust.pvalues[term]),
                "standardized_beta": standardized_beta(sample, beta, term),
                "reference_distribution": (
                    "t" if bool(getattr(robust, "use_t", False)) else "normal_z"
                ),
                "statsmodels_use_t": bool(getattr(robust, "use_t", False)),
            }
        )
    return pd.DataFrame(rows)


def model_fit_row(spec: ModelSpec, classical: Any) -> dict[str, Any]:
    return {
        "model_id": spec.model_id,
        "model_label": spec.model_label,
        "omitted_mean_variable": spec.omitted_mean_variable,
        "primary_window": PRIMARY_FEATURE_SET,
        "n": int(classical.nobs),
        "r_squared": float(classical.rsquared),
        "adjusted_r_squared": float(classical.rsquared_adj),
        "aic": float(classical.aic),
        "bic": float(classical.bic),
        "rmse": rmse_from_fit(classical),
        "predictor_count_excluding_intercept": len(spec.terms),
        "formula": "ESS ~ " + " + ".join(spec.terms),
    }


def build_vif_sensitivity(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coefficient_frames: list[pd.DataFrame] = []
    fit_rows: list[dict[str, Any]] = []
    vif_frames: list[pd.DataFrame] = []

    for spec in MODEL_SPECS:
        classical, robust, design = fit_ols_hc3(sample, spec.terms)
        coefficient_frames.append(model_coefficients(sample, spec, classical, robust))
        fit_rows.append(model_fit_row(spec, classical))
        vif_frames.append(vif_table(design, spec.model_id, spec.model_label))

    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    fit = pd.DataFrame(fit_rows)
    vif = pd.concat(vif_frames, ignore_index=True)
    reproduction = validate_reference_reproduction(
        coefficients.loc[coefficients["model_id"].eq("reference_joint_model")],
        fit.loc[fit["model_id"].eq("reference_joint_model")],
        vif.loc[vif["model_id"].eq("reference_joint_model")],
    )

    reference_sd = coefficients.loc[
        coefficients["model_id"].eq("reference_joint_model")
        & coefficients["term"].eq("sleep_duration_sd")
    ].iloc[0]
    reference_max_vif = float(
        vif.loc[vif["model_id"].eq("reference_joint_model"), "vif"].max()
    )
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        model_coefficients_df = coefficients.loc[coefficients["model_id"].eq(spec.model_id)]
        sd_row = model_coefficients_df.loc[
            model_coefficients_df["term"].eq("sleep_duration_sd")
        ].iloc[0]
        model_vif = vif.loc[vif["model_id"].eq(spec.model_id)].copy()
        sd_vif = float(model_vif.loc[model_vif["term"].eq("sleep_duration_sd"), "vif"].iloc[0])
        max_vif = float(model_vif["vif"].max())
        rows.append(
            {
                "model_id": spec.model_id,
                "model_label": spec.model_label,
                "omitted_mean_variable": spec.omitted_mean_variable,
                "n": int(sd_row["n"]),
                "sleep_duration_sd_coefficient": float(sd_row["coefficient"]),
                "sleep_duration_sd_standardized_beta": float(sd_row["standardized_beta"]),
                "sleep_duration_sd_hc3_standard_error": float(sd_row["hc3_standard_error"]),
                "sleep_duration_sd_ci_lower": float(sd_row["ci_lower"]),
                "sleep_duration_sd_ci_upper": float(sd_row["ci_upper"]),
                "sleep_duration_sd_p_value": float(sd_row["p_value"]),
                "sleep_duration_sd_positive": bool(float(sd_row["coefficient"]) > 0),
                "sleep_duration_sd_p_lt_0_05": bool(float(sd_row["p_value"]) < 0.05),
                "sleep_duration_sd_standardized_beta_difference_from_reference": float(
                    sd_row["standardized_beta"] - reference_sd["standardized_beta"]
                ),
                "sleep_duration_sd_vif": sd_vif,
                "max_vif": max_vif,
                "max_vif_term": str(model_vif.sort_values("vif", ascending=False)["term"].iloc[0]),
                "max_vif_reduction_from_reference": reference_max_vif - max_vif,
                "vif_gt_5_terms": "; ".join(model_vif.loc[model_vif["vif_gt_5"], "term"].tolist()),
                "vif_gt_10_terms": "; ".join(model_vif.loc[model_vif["vif_gt_10"], "term"].tolist()),
                **reproduction,
                "interpretation_note": (
                    "Reference rows reproduce script 09 before VIF sensitivity is interpreted."
                ),
            }
        )
    return pd.DataFrame(rows), coefficients, fit, vif


def assign_duration_sd_quartiles(sample: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = sample.copy()
    values = numeric_series(work["sleep_duration_sd"]).astype(float)
    try:
        quartiles = pd.qcut(values, q=4, labels=QUARTILE_LABELS, duplicates="raise")
        method = "pd.qcut(sleep_duration_sd, q=4, duplicates='raise')"
        failure_reason = ""
    except ValueError as exc:
        ranks = values.rank(method="first")
        quartiles = pd.qcut(ranks, q=4, labels=QUARTILE_LABELS, duplicates="raise")
        method = "pd.qcut(rank(method='first'), q=4) after tied cutpoint failure"
        failure_reason = str(exc)
    work["sleep_duration_sd_quartile"] = quartiles.astype(str)
    work["sleep_duration_sd_quartile_ordinal"] = work["sleep_duration_sd_quartile"].map(
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


def quartile_design(sample: pd.DataFrame, force_quartile: str | None = None) -> pd.DataFrame:
    x = pd.DataFrame(index=sample.index)
    for label in QUARTILE_LABELS[1:]:
        column = f"{label}_vs_Q1"
        if force_quartile is None:
            x[column] = sample["sleep_duration_sd_quartile"].eq(label).astype(float)
        else:
            x[column] = 1.0 if force_quartile == label else 0.0
    for term in ["sleep_duration_mean", *COVARIATES]:
        x[term] = numeric_series(sample[term]).astype(float)
    return sm.add_constant(x, has_constant="add")


def trend_design(sample: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=sample.index)
    x["quartile_ordinal"] = numeric_series(sample["sleep_duration_sd_quartile_ordinal"]).astype(float)
    for term in ["sleep_duration_mean", *COVARIATES]:
        x[term] = numeric_series(sample[term]).astype(float)
    return sm.add_constant(x, has_constant="add")


def marginal_estimate(
    robust_fit: Any,
    robust_cov: pd.DataFrame,
    sample: pd.DataFrame,
    quartile: str,
    q1_design_mean: pd.Series,
) -> dict[str, float]:
    forced_design = quartile_design(sample, force_quartile=quartile)
    l_vec = forced_design.mean(axis=0).reindex(robust_fit.params.index)
    params = robust_fit.params.reindex(l_vec.index)
    adjusted = float(np.dot(l_vec, params))
    cov = robust_cov.reindex(index=l_vec.index, columns=l_vec.index)
    cov_values = cov.to_numpy(dtype=float)
    se = float(np.sqrt(max(np.dot(l_vec, np.dot(cov_values, l_vec)), 0.0)))
    diff_vec = l_vec - q1_design_mean.reindex(l_vec.index)
    diff = float(np.dot(diff_vec, params))
    diff_se = float(np.sqrt(max(np.dot(diff_vec, np.dot(cov_values, diff_vec)), 0.0)))
    return {
        "adjusted_ess": adjusted,
        "adjusted_ess_standard_error": se,
        "adjusted_ess_ci_lower": adjusted - 1.96 * se,
        "adjusted_ess_ci_upper": adjusted + 1.96 * se,
        "adjusted_difference_vs_q1": diff,
        "adjusted_difference_vs_q1_standard_error": diff_se,
        "adjusted_difference_vs_q1_ci_lower": diff - 1.96 * diff_se,
        "adjusted_difference_vs_q1_ci_upper": diff + 1.96 * diff_se,
    }


def robust_overall_quartile_test(robust_fit: Any) -> dict[str, float | str]:
    terms = ["Q2_vs_Q1", "Q3_vs_Q1", "Q4_vs_Q1"]
    r_matrix = np.zeros((len(terms), len(robust_fit.params)))
    param_names = list(robust_fit.params.index)
    for row_idx, term in enumerate(terms):
        r_matrix[row_idx, param_names.index(term)] = 1.0
    wald = robust_fit.wald_test(r_matrix, use_f=False, scalar=True)
    return {
        "overall_quartile_test": "HC3 robust Wald chi-square test of Q2=Q3=Q4=0 vs Q1 reference",
        "overall_quartile_statistic": float(wald.statistic),
        "overall_quartile_df": int(len(terms)),
        "overall_quartile_p_value": float(wald.pvalue),
        "overall_quartile_p_value_interpretation": (
            "Tests whether any adjusted ESS differs across sleep duration SD quartiles "
            "relative to Q1, not whether the pattern is linear."
        ),
    }


def build_quartile_categorical(sample: pd.DataFrame) -> pd.DataFrame:
    sample, metadata = assign_duration_sd_quartiles(sample)
    y = numeric_series(sample[OUTCOME]).astype(float)
    x = quartile_design(sample)
    classical = sm.OLS(y, x).fit()
    robust = sm.OLS(y, x).fit(cov_type="HC3")
    robust_cov = robust.cov_params()
    q1_design_mean = quartile_design(sample, force_quartile="Q1").mean(axis=0)
    overall = robust_overall_quartile_test(robust)

    trend_x = trend_design(sample)
    trend_classical = sm.OLS(y, trend_x).fit()
    trend_robust = sm.OLS(y, trend_x).fit(cov_type="HC3")

    rows: list[dict[str, Any]] = []
    for label in QUARTILE_LABELS:
        group = sample.loc[sample["sleep_duration_sd_quartile"].eq(label)]
        estimates = marginal_estimate(robust, robust_cov, sample, label, q1_design_mean)
        coefficient = np.nan
        se = np.nan
        ci_lower = np.nan
        ci_upper = np.nan
        p_value = np.nan
        if label != "Q1":
            term = f"{label}_vs_Q1"
            coefficient = float(robust.params[term])
            se = float(robust.bse[term])
            ci_lower, ci_upper = conf_int_values(robust, term)
            p_value = float(robust.pvalues[term])
        rows.append(
            {
                "quartile": label,
                "quartile_ordinal": QUARTILE_LABELS.index(label) + 1,
                "n": int(len(group)),
                "model_n": int(classical.nobs),
                "raw_ess_mean": float(numeric_series(group[OUTCOME]).mean()),
                "adjusted_ess": estimates["adjusted_ess"],
                "adjusted_ess_standard_error": estimates["adjusted_ess_standard_error"],
                "adjusted_ess_ci_lower": estimates["adjusted_ess_ci_lower"],
                "adjusted_ess_ci_upper": estimates["adjusted_ess_ci_upper"],
                "q1_reference": "Q1",
                "coefficient_vs_q1": coefficient,
                "hc3_standard_error_vs_q1": se,
                "ci_lower_vs_q1": ci_lower,
                "ci_upper_vs_q1": ci_upper,
                "p_value_vs_q1": p_value,
                "adjusted_difference_vs_q1": estimates["adjusted_difference_vs_q1"],
                "adjusted_difference_vs_q1_standard_error": estimates[
                    "adjusted_difference_vs_q1_standard_error"
                ],
                "adjusted_difference_vs_q1_ci_lower": estimates[
                    "adjusted_difference_vs_q1_ci_lower"
                ],
                "adjusted_difference_vs_q1_ci_upper": estimates[
                    "adjusted_difference_vs_q1_ci_upper"
                ],
                "p_for_linear_trend": float(trend_robust.pvalues["quartile_ordinal"]),
                "trend_beta_per_quartile": float(trend_robust.params["quartile_ordinal"]),
                "trend_model_r_squared": float(trend_classical.rsquared),
                "categorical_model_r_squared": float(classical.rsquared),
                "categorical_model_adjusted_r_squared": float(classical.rsquared_adj),
                "adjustment": "Adjusted for mean sleep duration, age, sex, BMI, and AHI",
                "prediction_method": (
                    "Marginal standardization: covariates retained and only quartile set to Q1-Q4."
                ),
                "ci_method": "Delta method with statsmodels HC3 covariance matrix.",
                **overall,
                **metadata,
            }
        )
    return pd.DataFrame(rows)


def build_nonlinearity(sample: pd.DataFrame) -> pd.DataFrame:
    work = sample.copy()
    center = float(numeric_series(work["sleep_duration_sd"]).mean())
    work["duration_sd_centered"] = numeric_series(work["sleep_duration_sd"]) - center
    work["duration_sd_centered_squared"] = work["duration_sd_centered"] ** 2
    linear_terms = ["duration_sd_centered", "sleep_duration_mean", *COVARIATES]
    quadratic_terms = [
        "duration_sd_centered",
        "duration_sd_centered_squared",
        "sleep_duration_mean",
        *COVARIATES,
    ]
    linear_classical, linear_robust, _ = fit_ols_hc3(work, linear_terms)
    quad_classical, quad_robust, _ = fit_ols_hc3(work, quadratic_terms)
    rows: list[dict[str, Any]] = []
    for model_id, classical, robust, terms in [
        ("linear", linear_classical, linear_robust, linear_terms),
        ("quadratic", quad_classical, quad_robust, quadratic_terms),
    ]:
        rows.append(
            {
                "row_type": "model_fit",
                "model": model_id,
                "term": "",
                "n": int(classical.nobs),
                "coefficient": np.nan,
                "hc3_standard_error": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "p_value": np.nan,
                "r_squared": float(classical.rsquared),
                "adjusted_r_squared": float(classical.rsquared_adj),
                "aic": float(classical.aic),
                "bic": float(classical.bic),
                "delta_r_squared_vs_linear": float(classical.rsquared - linear_classical.rsquared),
                "delta_adjusted_r_squared_vs_linear": float(
                    classical.rsquared_adj - linear_classical.rsquared_adj
                ),
                "delta_aic_vs_linear": float(classical.aic - linear_classical.aic),
                "delta_bic_vs_linear": float(classical.bic - linear_classical.bic),
                "sleep_duration_sd_center_value": center,
                "formula": "ESS ~ " + " + ".join(terms),
                "interpretation_note": (
                    "Quadratic evidence should consider both the squared-term p-value and fit changes."
                ),
            }
        )
        for term in terms:
            if term not in ["duration_sd_centered", "duration_sd_centered_squared"]:
                continue
            ci_lower, ci_upper = conf_int_values(robust, term)
            rows.append(
                {
                    "row_type": "term",
                    "model": model_id,
                    "term": term,
                    "n": int(classical.nobs),
                    "coefficient": float(robust.params[term]),
                    "hc3_standard_error": float(robust.bse[term]),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "p_value": float(robust.pvalues[term]),
                    "r_squared": float(classical.rsquared),
                    "adjusted_r_squared": float(classical.rsquared_adj),
                    "aic": float(classical.aic),
                    "bic": float(classical.bic),
                    "delta_r_squared_vs_linear": float(classical.rsquared - linear_classical.rsquared),
                    "delta_adjusted_r_squared_vs_linear": float(
                        classical.rsquared_adj - linear_classical.rsquared_adj
                    ),
                    "delta_aic_vs_linear": float(classical.aic - linear_classical.aic),
                    "delta_bic_vs_linear": float(classical.bic - linear_classical.bic),
                    "sleep_duration_sd_center_value": center,
                    "formula": "ESS ~ " + " + ".join(terms),
                    "interpretation_note": (
                        "duration_sd_centered is sleep_duration_sd minus its complete-case mean."
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
        return "ref"
    if value < 0.001:
        return "p<0.001"
    return f"p={value:.3f}"


def save_quartile_figure(quartiles: pd.DataFrame) -> None:
    if not PIL_AVAILABLE:
        logging.warning("PIL unavailable; quartile figure was not created.")
        return
    width, height = 1120, 760
    image = Image.new("RGB", (width, height), "#f7f9fb")
    draw = ImageDraw.Draw(image)
    title_font = get_font(32, bold=True)
    subtitle_font = get_font(18)
    label_font = get_font(16)
    small_font = get_font(13)
    draw.text((54, 34), "Sleep Duration SD Quartiles and Adjusted ESS", font=title_font, fill="#1f2933")
    draw.text(
        (54, 82),
        "Adjusted for mean sleep duration, age, sex, BMI, and AHI",
        font=subtitle_font,
        fill="#52616b",
    )
    overall_p = float(quartiles["overall_quartile_p_value"].iloc[0])
    trend_p = float(quartiles["p_for_linear_trend"].iloc[0])
    draw.text(
        (54, 112),
        f"Marginal-standardized adjusted ESS; overall {format_p(overall_p)}, trend {format_p(trend_p)}",
        font=label_font,
        fill="#52616b",
    )

    left, top, right, bottom = 110, 180, 1030, 620
    y_min = math.floor((float(quartiles["adjusted_ess_ci_lower"].min()) - 0.35) * 2) / 2
    y_max = math.ceil((float(quartiles["adjusted_ess_ci_upper"].max()) + 0.35) * 2) / 2
    draw.line((left, bottom, right, bottom), fill="#90a4ae", width=2)
    draw.line((left, top, left, bottom), fill="#90a4ae", width=2)
    draw.text((left - 62, top - 8), f"{y_max:g}", font=small_font, fill="#52616b")
    draw.text((left - 62, bottom - 10), f"{y_min:g}", font=small_font, fill="#52616b")
    draw.text((left + 310, bottom + 92), "Sleep duration SD quartile", font=label_font, fill="#52616b")
    draw.text((20, top + 170), "Adjusted ESS", font=label_font, fill="#52616b")

    color = "#2f6f73"
    points: list[tuple[int, int]] = []
    for _, row in quartiles.sort_values("quartile_ordinal").iterrows():
        x = int(left + (int(row["quartile_ordinal"]) - 1) / 3 * (right - left))
        y = int(bottom - (float(row["adjusted_ess"]) - y_min) / (y_max - y_min) * (bottom - top))
        y_lo = int(
            bottom - (float(row["adjusted_ess_ci_lower"]) - y_min) / (y_max - y_min) * (bottom - top)
        )
        y_hi = int(
            bottom - (float(row["adjusted_ess_ci_upper"]) - y_min) / (y_max - y_min) * (bottom - top)
        )
        points.append((x, y))
        draw.line((x, y_hi, x, y_lo), fill=color, width=3)
        draw.line((x - 10, y_hi, x + 10, y_hi), fill=color, width=3)
        draw.line((x - 10, y_lo, x + 10, y_lo), fill=color, width=3)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color)
        draw_centered_text(draw, (x, y - 26), f"{float(row['adjusted_ess']):.2f}", label_font)
        draw_centered_text(draw, (x, bottom + 25), str(row["quartile"]), label_font, fill="#52616b")
        draw_centered_text(draw, (x, bottom + 48), f"n={int(row['n'])}", small_font, fill="#52616b")
        draw_centered_text(
            draw,
            (x, bottom + 69),
            format_p(float(row["p_value_vs_q1"])) if pd.notna(row["p_value_vs_q1"]) else "ref",
            small_font,
            fill="#52616b",
        )
    draw.line(points, fill=color, width=3)
    image.save(QUARTILE_FIGURE_PATH)
    logging.info("Wrote figure %s", QUARTILE_FIGURE_PATH)


def prediction_design(
    grid_sd: np.ndarray,
    sample: pd.DataFrame,
    center: float,
    quadratic: bool,
) -> pd.DataFrame:
    centered = grid_sd - center
    base = {
        "duration_sd_centered": centered,
        "sleep_duration_mean": np.repeat(float(sample["sleep_duration_mean"].mean()), len(grid_sd)),
        "age": np.repeat(float(sample["age"].mean()), len(grid_sd)),
        "sex_Male": np.repeat(float(sample["sex_Male"].mean()), len(grid_sd)),
        "bmi": np.repeat(float(sample["bmi"].mean()), len(grid_sd)),
        "ahi": np.repeat(float(sample["ahi"].mean()), len(grid_sd)),
    }
    if quadratic:
        base["duration_sd_centered_squared"] = centered**2
        order = [
            "duration_sd_centered",
            "duration_sd_centered_squared",
            "sleep_duration_mean",
            *COVARIATES,
        ]
    else:
        order = ["duration_sd_centered", "sleep_duration_mean", *COVARIATES]
    frame = pd.DataFrame(base)[order]
    return sm.add_constant(frame, has_constant="add")


def predict_with_ci(fit: Any, design: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = design.reindex(columns=fit.params.index)
    params = fit.params.reindex(design.columns).to_numpy(dtype=float)
    cov = fit.cov_params().reindex(index=design.columns, columns=design.columns).to_numpy(dtype=float)
    x = design.to_numpy(dtype=float)
    pred = x @ params
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", x, cov, x), 0.0))
    return pred, pred - 1.96 * se, pred + 1.96 * se


def map_point(
    x_value: float,
    y_value: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    bounds: tuple[int, int, int, int],
) -> tuple[int, int]:
    left, top, right, bottom = bounds
    x = int(left + (x_value - x_min) / (x_max - x_min) * (right - left))
    y = int(bottom - (y_value - y_min) / (y_max - y_min) * (bottom - top))
    return x, y


def save_relationship_figure(sample: pd.DataFrame) -> None:
    if not PIL_AVAILABLE:
        logging.warning("PIL unavailable; relationship figure was not created.")
        return
    work = sample.copy()
    center = float(work["sleep_duration_sd"].mean())
    work["duration_sd_centered"] = work["sleep_duration_sd"] - center
    work["duration_sd_centered_squared"] = work["duration_sd_centered"] ** 2
    linear_terms = ["duration_sd_centered", "sleep_duration_mean", *COVARIATES]
    quadratic_terms = [
        "duration_sd_centered",
        "duration_sd_centered_squared",
        "sleep_duration_mean",
        *COVARIATES,
    ]
    _, linear_fit, _ = fit_ols_hc3(work, linear_terms)
    _, quadratic_fit, _ = fit_ols_hc3(work, quadratic_terms)
    x_min = float(work["sleep_duration_sd"].quantile(0.01))
    x_max = float(work["sleep_duration_sd"].quantile(0.99))
    grid = np.linspace(x_min, x_max, 160)
    linear_pred, linear_low, linear_high = predict_with_ci(
        linear_fit, prediction_design(grid, work, center, quadratic=False)
    )
    quadratic_pred, quadratic_low, quadratic_high = predict_with_ci(
        quadratic_fit, prediction_design(grid, work, center, quadratic=True)
    )
    y_min = math.floor(
        min(float(linear_low.min()), float(quadratic_low.min()), float(linear_pred.min())) * 2
    ) / 2
    y_max = math.ceil(
        max(float(linear_high.max()), float(quadratic_high.max()), float(linear_pred.max())) * 2
    ) / 2

    width, height = 1220, 760
    image = Image.new("RGB", (width, height), "#f7f9fb")
    draw = ImageDraw.Draw(image)
    title_font = get_font(32, bold=True)
    subtitle_font = get_font(18)
    label_font = get_font(16)
    small_font = get_font(13)
    draw.text((54, 34), "Adjusted ESS Across Sleep Duration SD", font=title_font, fill="#1f2933")
    draw.text(
        (54, 82),
        "Adjusted for mean sleep duration, age, sex, BMI, and AHI; prediction range is observed 1st-99th percentile",
        font=subtitle_font,
        fill="#52616b",
    )

    bounds = (120, 155, 1120, 610)
    left, top, right, bottom = bounds
    draw.line((left, bottom, right, bottom), fill="#90a4ae", width=2)
    draw.line((left, top, left, bottom), fill="#90a4ae", width=2)
    draw.text((left - 70, top - 8), f"{y_max:g}", font=small_font, fill="#52616b")
    draw.text((left - 70, bottom - 10), f"{y_min:g}", font=small_font, fill="#52616b")
    draw.text((left, bottom + 36), f"{x_min:.1f}", font=small_font, fill="#52616b")
    draw.text((right - 44, bottom + 36), f"{x_max:.1f}", font=small_font, fill="#52616b")
    draw.text((left + 365, bottom + 72), "Sleep duration SD (minutes)", font=label_font, fill="#52616b")
    draw.text((24, top + 190), "Adjusted predicted ESS", font=label_font, fill="#52616b")

    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    upper = [map_point(x, y, x_min, x_max, y_min, y_max, bounds) for x, y in zip(grid, linear_high)]
    lower = [
        map_point(x, y, x_min, x_max, y_min, y_max, bounds)
        for x, y in zip(grid[::-1], linear_low[::-1])
    ]
    overlay_draw.polygon(upper + lower, fill=(47, 111, 115, 48))
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)

    linear_points = [
        map_point(x, y, x_min, x_max, y_min, y_max, bounds) for x, y in zip(grid, linear_pred)
    ]
    quadratic_points = [
        map_point(x, y, x_min, x_max, y_min, y_max, bounds) for x, y in zip(grid, quadratic_pred)
    ]
    draw.line(linear_points, fill="#2f6f73", width=4)
    for start in range(0, len(quadratic_points) - 1, 6):
        draw.line(quadratic_points[start : start + 4], fill="#c17c1c", width=3)
    draw.line((830, 132, 870, 132), fill="#2f6f73", width=4)
    draw.text((880, 122), "Linear prediction + 95% CI", font=small_font, fill="#1f2933")
    for x0 in range(830, 870, 12):
        draw.line((x0, 156, x0 + 7, 156), fill="#c17c1c", width=3)
    draw.text((880, 146), "Quadratic prediction", font=small_font, fill="#1f2933")
    image.save(RELATIONSHIP_FIGURE_PATH)
    logging.info("Wrote figure %s", RELATIONSHIP_FIGURE_PATH)


def main() -> None:
    setup_logging()
    backup_existing_outputs()
    check_environment()
    joint_sample = load_joint_complete_case()

    sensitivity, sensitivity_coefficients, sensitivity_fit, sensitivity_vif = build_vif_sensitivity(
        joint_sample
    )
    quartile_categorical = build_quartile_categorical(joint_sample)
    nonlinearity = build_nonlinearity(joint_sample)

    write_csv(sensitivity, VIF_SENSITIVITY_SUMMARY_PATH)
    write_csv(sensitivity_coefficients, VIF_SENSITIVITY_COEFFICIENTS_PATH)
    write_csv(sensitivity_fit, VIF_SENSITIVITY_FIT_PATH)
    write_csv(sensitivity_vif, VIF_SENSITIVITY_VIF_PATH)
    write_csv(quartile_categorical, QUARTILE_CATEGORICAL_PATH)
    write_csv(nonlinearity, NONLINEARITY_PATH)
    save_quartile_figure(quartile_categorical)
    save_relationship_figure(joint_sample)

    print("Final robustness and nonlinearity checks complete")
    print(f"Primary window: {PRIMARY_FEATURE_SET}")
    print(f"Joint complete-case n: {len(joint_sample)}")
    for _, row in sensitivity.iterrows():
        print(
            f"{row['model_id']}: sleep_duration_sd beta={float(row['sleep_duration_sd_coefficient']):.6f}, "
            f"std_beta={float(row['sleep_duration_sd_standardized_beta']):.6f}, "
            f"p={float(row['sleep_duration_sd_p_value']):.6g}, "
            f"max_VIF={float(row['max_vif']):.3f}, "
            f"duration_SD_VIF={float(row['sleep_duration_sd_vif']):.3f}"
        )
    q4 = quartile_categorical.loc[quartile_categorical["quartile"].eq("Q4")].iloc[0]
    print(
        f"Quartile overall p={float(q4['overall_quartile_p_value']):.6g}; "
        f"trend p={float(q4['p_for_linear_trend']):.6g}; "
        f"Q4-Q1 adjusted diff={float(q4['adjusted_difference_vs_q1']):.6f}"
    )
    quad = nonlinearity.loc[
        nonlinearity["term"].eq("duration_sd_centered_squared")
        & nonlinearity["model"].eq("quadratic")
    ].iloc[0]
    print(
        f"Quadratic term beta={float(quad['coefficient']):.8f}, "
        f"p={float(quad['p_value']):.6g}, "
        f"delta_adj_R2={float(quad['delta_adjusted_r_squared_vs_linear']):.6f}"
    )


if __name__ == "__main__":
    main()
