"""Validate pilot regression results with statsmodels.

This script does not rerun a new analysis strategy. It reproduces the existing
pilot OLS models from ``pilot_regression_dataset.csv`` using statsmodels and
compares the results with the direct numpy/scipy implementation saved by
``src/05_pilot_regression_analysis.py``.
"""

from __future__ import annotations

import importlib
import json
import logging
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
OUTPUT_LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
BACKUP_DIR = PROJECT_ROOT / "outputs" / "backups" / datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

PILOT_DATASET_PATH = PROCESSED_DIR / "pilot_regression_dataset.csv"
MANUAL_COEFFICIENTS_PATH = OUTPUT_TABLE_DIR / "pilot_regression_coefficients.csv"
MANUAL_MODEL_FIT_PATH = OUTPUT_TABLE_DIR / "pilot_regression_model_fit.csv"
MANUAL_INCREMENTAL_PATH = OUTPUT_TABLE_DIR / "pilot_incremental_value_comparison.csv"
MANUAL_SENSITIVITY_PATH = OUTPUT_TABLE_DIR / "pilot_sensitivity_analysis.csv"

STATSMODELS_COEFFICIENTS_PATH = OUTPUT_TABLE_DIR / "statsmodels_regression_coefficients.csv"
STATSMODELS_MODEL_FIT_PATH = OUTPUT_TABLE_DIR / "statsmodels_regression_model_fit.csv"
STATSMODELS_DESIGN_MATRIX_PATH = OUTPUT_TABLE_DIR / "statsmodels_design_matrix.csv"
COEFFICIENT_COMPARISON_PATH = OUTPUT_TABLE_DIR / "manual_vs_statsmodels_comparison.csv"
MODEL_FIT_COMPARISON_PATH = OUTPUT_TABLE_DIR / "manual_vs_statsmodels_model_fit_comparison.csv"
STATSMODELS_INCREMENTAL_PATH = OUTPUT_TABLE_DIR / "statsmodels_incremental_value_comparison.csv"
STATSMODELS_VIF_PATH = OUTPUT_TABLE_DIR / "statsmodels_vif.csv"
STATSMODELS_INFLUENTIAL_PATH = OUTPUT_TABLE_DIR / "statsmodels_influential_observations.csv"
STATSMODELS_DIAGNOSTICS_PATH = OUTPUT_TABLE_DIR / "statsmodels_diagnostics.csv"
STATSMODELS_SAMPLE_VALIDATION_PATH = OUTPUT_TABLE_DIR / "statsmodels_sample_validation.csv"
ENVIRONMENT_PATH = OUTPUT_TABLE_DIR / "statsmodels_environment.csv"
LOG_PATH = OUTPUT_LOG_DIR / "06_validate_regression_with_statsmodels.log"

REQUESTED_OUTPUTS = [
    STATSMODELS_COEFFICIENTS_PATH,
    STATSMODELS_MODEL_FIT_PATH,
    STATSMODELS_DESIGN_MATRIX_PATH,
    COEFFICIENT_COMPARISON_PATH,
    MODEL_FIT_COMPARISON_PATH,
    STATSMODELS_INCREMENTAL_PATH,
    STATSMODELS_VIF_PATH,
    STATSMODELS_INFLUENTIAL_PATH,
    STATSMODELS_DIAGNOSTICS_PATH,
    STATSMODELS_SAMPLE_VALIDATION_PATH,
    ENVIRONMENT_PATH,
]

CSV_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1")
EXPECTED_INFLUENTIAL_MODEL2 = {"159", "286", "6", "70"}
PRIMARY_COMPLETE_COLUMN = "primary_common_complete_case"

COEFFICIENT_TOL = 1e-6
R2_TOL = 1e-8
SE_TOL = 1e-5
P_TOL = 1e-5
CI_TOL = 1e-5

MODEL_TERMS = {
    "Model 0": ["age", "sex_Male", "bmi", "ahi"],
    "Model 1": ["sleep_duration_mean", "age", "sex_Male", "bmi", "ahi"],
    "Model 2": ["sleep_duration_mean", "sleep_duration_sd", "age", "sex_Male", "bmi", "ahi"],
    "Model 3": ["sleep_duration_mean", "sleep_duration_cv", "age", "sex_Male", "bmi", "ahi"],
}

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


def ensure_dirs() -> None:
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
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
        logging.info("No requested statsmodels validation outputs existed before this run.")
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for path in existing:
        destination = BACKUP_DIR / path.name
        shutil.copy2(path, destination)
        logging.info("Backed up %s to %s", path, destination)


def check_environment() -> pd.DataFrame:
    """Import statsmodels lazily so failures can report the Python executable."""
    global sm, statsmodels_module, variance_inflation_factor, OLSInfluence, het_breuschpagan
    try:
        statsmodels_module = importlib.import_module("statsmodels")
        sm = importlib.import_module("statsmodels.api")
        variance_inflation_factor = importlib.import_module(
            "statsmodels.stats.outliers_influence"
        ).variance_inflation_factor
        OLSInfluence = importlib.import_module(
            "statsmodels.stats.outliers_influence"
        ).OLSInfluence
        het_breuschpagan = importlib.import_module(
            "statsmodels.stats.diagnostic"
        ).het_breuschpagan
    except Exception as exc:
        message = (
            "statsmodels import failed. "
            f"sys.executable={sys.executable}; error={type(exc).__name__}: {exc}"
        )
        print(message)
        logging.error(message)
        raise SystemExit(1) from exc

    import numpy

    env = pd.DataFrame(
        [
            {"item": "sys.executable", "value": sys.executable},
            {"item": "python_version", "value": platform.python_version()},
            {"item": "numpy_version", "value": numpy.__version__},
            {"item": "scipy_version", "value": scipy.__version__},
            {"item": "pandas_version", "value": pd.__version__},
            {"item": "statsmodels_version", "value": statsmodels_module.__version__},
        ]
    )
    for _, row in env.iterrows():
        print(f"{row['item']}: {row['value']}")
        logging.info("%s: %s", row["item"], row["value"])
    return env


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype("string").str.lower().isin(["true", "1", "yes"])


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pilot = read_csv_with_fallback(PILOT_DATASET_PATH)
    manual_coef = read_csv_with_fallback(MANUAL_COEFFICIENTS_PATH)
    manual_fit = read_csv_with_fallback(MANUAL_MODEL_FIT_PATH)
    manual_incremental = read_csv_with_fallback(MANUAL_INCREMENTAL_PATH)
    manual_sensitivity = read_csv_with_fallback(MANUAL_SENSITIVITY_PATH)
    return pilot, manual_coef, manual_fit, manual_incremental, manual_sensitivity


def prepare_common_sample(pilot: pd.DataFrame, manual_fit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = pilot.loc[to_bool(pilot[PRIMARY_COMPLETE_COLUMN])].copy()
    sample["mesaid"] = sample["mesaid"].astype(str)
    sample["sex_Male"] = sample["sex_category"].astype(str).eq("Male").astype(float)

    required = [
        "ess",
        "age",
        "sex_Male",
        "bmi",
        "ahi",
        "sleep_duration_mean",
        "sleep_duration_sd",
        "sleep_duration_cv",
    ]
    rows: list[dict[str, Any]] = [
        {"check": "analysis_participant_count", "value": len(sample), "status": "ok"},
        {
            "check": "manual_model_raw_n_values",
            "value": json.dumps(
                manual_fit.loc[manual_fit["scale"].eq("raw_units"), ["model", "n"]].to_dict(
                    orient="records"
                )
            ),
            "status": "ok",
        },
        {
            "check": "mesaid_list_in_order",
            "value": json.dumps(sample["mesaid"].tolist()),
            "status": "ok",
        },
        {
            "check": "duplicate_mesaid_count",
            "value": int(sample["mesaid"].duplicated(keep=False).sum()),
            "status": "ok" if not sample["mesaid"].duplicated(keep=False).any() else "warning",
        },
    ]
    for column in required:
        values = numeric_series(sample[column])
        missing = int(values.isna().sum())
        infinite = int(np.isinf(values).sum())
        rows.append(
            {
                "check": f"variable_quality_{column}",
                "value": json.dumps({"missing": missing, "infinite": infinite}),
                "status": "ok" if missing == 0 and infinite == 0 else "warning",
            }
        )
    manual_raw_n = set(
        pd.to_numeric(manual_fit.loc[manual_fit["scale"].eq("raw_units"), "n"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )
    rows.append(
        {
            "check": "sample_n_matches_manual_raw_models",
            "value": int(len(sample)),
            "status": "ok" if manual_raw_n == {len(sample)} else "error",
        }
    )
    return sample, pd.DataFrame(rows)


def design_matrix(sample: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    columns = [pd.Series(1.0, index=sample.index, name="Intercept")]
    for term in terms:
        columns.append(numeric_series(sample[term]).astype(float).rename(term))
    return pd.concat(columns, axis=1)


def fit_models(sample: pd.DataFrame) -> tuple[dict[str, ModelResult], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_results: dict[str, ModelResult] = {}
    coef_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    design_rows: list[dict[str, Any]] = []
    y = numeric_series(sample["ess"]).astype(float)
    for model, terms in MODEL_TERMS.items():
        x = design_matrix(sample, terms)
        classical = sm.OLS(y, x).fit()
        robust = sm.OLS(y, x).fit(cov_type="HC3")
        model_results[model] = ModelResult(model, robust, classical, sample.copy(), x.copy())
        rmse = float(np.sqrt(np.mean(np.square(classical.resid))))

        conf = robust.conf_int(alpha=0.05)
        conf.columns = ["ci_lower", "ci_upper"]
        for term in robust.params.index:
            coef_rows.append(
                {
                    "model": model,
                    "scale": "raw_units",
                    "term": term,
                    "n": int(robust.nobs),
                    "coefficient": float(robust.params[term]),
                    "hc3_standard_error": float(robust.bse[term]),
                    "ci_lower": float(conf.loc[term, "ci_lower"]),
                    "ci_upper": float(conf.loc[term, "ci_upper"]),
                    "p_value": float(robust.pvalues[term]),
                    "r_squared": float(classical.rsquared),
                    "adjusted_r_squared": float(classical.rsquared_adj),
                    "aic": float(classical.aic),
                    "bic": float(classical.bic),
                    "rmse": rmse,
                    "residual_df": float(classical.df_resid),
                }
            )
        fit_rows.append(
            {
                "model": model,
                "scale": "raw_units",
                "n": int(classical.nobs),
                "r_squared": float(classical.rsquared),
                "adjusted_r_squared": float(classical.rsquared_adj),
                "aic": float(classical.aic),
                "bic": float(classical.bic),
                "rmse": rmse,
                "residual_df": float(classical.df_resid),
                "parameter_count": int(classical.df_model + 1),
                "observations_per_parameter": float(classical.nobs / (classical.df_model + 1)),
            }
        )
        for row_order, (idx, row) in enumerate(x.iterrows(), start=1):
            for column_order, column_name in enumerate(x.columns, start=1):
                design_rows.append(
                    {
                        "model": model,
                        "row_order": row_order,
                        "mesaid": sample.loc[idx, "mesaid"],
                        "column_order": column_order,
                        "column_name": column_name,
                        "value": float(row[column_name]),
                    }
                )
    return (
        model_results,
        pd.DataFrame(coef_rows),
        pd.DataFrame(fit_rows),
        pd.DataFrame(design_rows),
    )


def compare_coef_rows(manual: pd.DataFrame, statsmodels_coef: pd.DataFrame) -> pd.DataFrame:
    manual_raw = manual.loc[manual["scale"].eq("raw_units")].copy()
    keep = [
        "model",
        "term",
        "coefficient",
        "robust_se",
        "ci_lower",
        "ci_upper",
        "p_value",
        "r_squared",
        "adjusted_r_squared",
        "aic",
        "bic",
    ]
    manual_raw = manual_raw[keep].rename(
        columns={
            "coefficient": "manual_coefficient",
            "robust_se": "manual_hc3_standard_error",
            "ci_lower": "manual_ci_lower",
            "ci_upper": "manual_ci_upper",
            "p_value": "manual_p_value",
            "r_squared": "manual_r_squared",
            "adjusted_r_squared": "manual_adjusted_r_squared",
            "aic": "manual_aic",
            "bic": "manual_bic",
        }
    )
    stats = statsmodels_coef.rename(
        columns={
            "coefficient": "statsmodels_coefficient",
            "hc3_standard_error": "statsmodels_hc3_standard_error",
            "ci_lower": "statsmodels_ci_lower",
            "ci_upper": "statsmodels_ci_upper",
            "p_value": "statsmodels_p_value",
            "r_squared": "statsmodels_r_squared",
            "adjusted_r_squared": "statsmodels_adjusted_r_squared",
            "aic": "statsmodels_aic",
            "bic": "statsmodels_bic",
        }
    )
    merged = manual_raw.merge(stats, on=["model", "term"], how="outer", indicator=True)
    numeric_pairs = [
        ("coefficient", COEFFICIENT_TOL),
        ("hc3_standard_error", SE_TOL),
        ("ci_lower", CI_TOL),
        ("ci_upper", CI_TOL),
        ("p_value", P_TOL),
        ("r_squared", R2_TOL),
        ("adjusted_r_squared", R2_TOL),
        ("aic", np.inf),
        ("bic", np.inf),
    ]
    for metric, _ in numeric_pairs:
        manual_col = f"manual_{metric}"
        stats_col = f"statsmodels_{metric}"
        if manual_col in merged and stats_col in merged:
            merged[f"{metric}_difference"] = (
                pd.to_numeric(merged[stats_col], errors="coerce")
                - pd.to_numeric(merged[manual_col], errors="coerce")
            )
    for metric, tolerance in numeric_pairs:
        if metric in {"aic", "bic"}:
            continue
        diff_col = f"{metric}_difference"
        if diff_col in merged:
            merged[f"{metric}_within_tolerance"] = (
                pd.to_numeric(merged[diff_col], errors="coerce").abs() <= tolerance
            ).fillna(False)
    merged["coefficient_and_hc3_se_within_tolerance"] = (
        merged.get("coefficient_within_tolerance", False)
        & merged.get("hc3_standard_error_within_tolerance", False)
    )

    reasons: list[str] = []
    within: list[bool] = []
    for _, row in merged.iterrows():
        row_reasons: list[str] = []
        exceeded_metrics: list[str] = []
        if row["_merge"] != "both":
            row_reasons.append(f"term_merge_status={row['_merge']}")
        for metric, tolerance in numeric_pairs:
            if metric in {"aic", "bic"}:
                continue
            diff = row.get(f"{metric}_difference", np.nan)
            if pd.isna(diff) or abs(float(diff)) > tolerance:
                exceeded_metrics.append(metric)
                row_reasons.append(f"{metric}_diff_exceeds_{tolerance}")
        if set(exceeded_metrics).issubset({"ci_lower", "ci_upper", "p_value"}) and bool(
            row.get("coefficient_and_hc3_se_within_tolerance", False)
        ):
            row_reasons.insert(
                0,
                "statsmodels_default_HC3_uses_normal_z_for_p_and_ci; "
                "manual_direct_implementation_used_residual_df_t_distribution",
            )
        within.append(len(row_reasons) == 0)
        reasons.append("; ".join(row_reasons) if row_reasons else "within tolerance")
    merged["within_tolerance"] = within
    merged["discrepancy_reason"] = reasons
    return merged


def compare_model_fit(manual_fit: pd.DataFrame, stats_fit: pd.DataFrame) -> pd.DataFrame:
    manual = manual_fit.loc[manual_fit["scale"].eq("raw_units")].copy()
    manual = manual.rename(
        columns={
            "r_squared": "manual_r_squared",
            "adjusted_r_squared": "manual_adjusted_r_squared",
            "aic": "manual_aic",
            "bic": "manual_bic",
            "rmse": "manual_rmse",
            "residual_df": "manual_residual_df",
            "n": "manual_n",
        }
    )
    stats = stats_fit.rename(
        columns={
            "r_squared": "statsmodels_r_squared",
            "adjusted_r_squared": "statsmodels_adjusted_r_squared",
            "aic": "statsmodels_aic",
            "bic": "statsmodels_bic",
            "rmse": "statsmodels_rmse",
            "residual_df": "statsmodels_residual_df",
            "n": "statsmodels_n",
        }
    )
    merged = manual.merge(stats, on=["model", "scale"], how="outer", indicator=True)
    for metric in ["r_squared", "adjusted_r_squared", "aic", "bic", "rmse", "residual_df"]:
        merged[f"{metric}_difference"] = (
            pd.to_numeric(merged[f"statsmodels_{metric}"], errors="coerce")
            - pd.to_numeric(merged[f"manual_{metric}"], errors="coerce")
        )
    reasons: list[str] = []
    within: list[bool] = []
    for _, row in merged.iterrows():
        row_reasons: list[str] = []
        if row["_merge"] != "both":
            row_reasons.append(f"model_merge_status={row['_merge']}")
        if abs(float(row.get("r_squared_difference", np.nan))) > R2_TOL:
            row_reasons.append("r_squared_difference")
        if abs(float(row.get("adjusted_r_squared_difference", np.nan))) > R2_TOL:
            row_reasons.append("adjusted_r_squared_difference")
        if abs(float(row.get("rmse_difference", np.nan))) > 1e-8:
            row_reasons.append("rmse_difference")
        within.append(len(row_reasons) == 0)
        reasons.append("; ".join(row_reasons) if row_reasons else "within tolerance")
    merged["within_tolerance"] = within
    merged["discrepancy_reason"] = reasons
    merged["aic_bic_note"] = "AIC/BIC differences are recorded but are not automatic failure criteria."
    return merged


def incremental_comparison(model_results: dict[str, ModelResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = model_results["Model 1"]
    for model, added_term in [("Model 2", "sleep_duration_sd"), ("Model 3", "sleep_duration_cv")]:
        full = model_results[model]
        f_stat, p_value, df_diff = full.classical_fit.compare_f_test(base.classical_fit)
        conf = full.robust_fit.conf_int(alpha=0.05)
        conf.columns = ["ci_lower", "ci_upper"]
        rows.append(
            {
                "comparison": f"Model 1 vs {model}",
                "n": int(full.classical_fit.nobs),
                "delta_r_squared": float(full.classical_fit.rsquared - base.classical_fit.rsquared),
                "delta_adjusted_r_squared": float(
                    full.classical_fit.rsquared_adj - base.classical_fit.rsquared_adj
                ),
                "delta_aic": float(full.classical_fit.aic - base.classical_fit.aic),
                "delta_bic": float(full.classical_fit.bic - base.classical_fit.bic),
                "classical_partial_f_statistic": float(f_stat),
                "classical_partial_f_p_value": float(p_value),
                "classical_partial_f_df_diff": float(df_diff),
                "added_variable": added_term,
                "added_variable_coefficient": float(full.robust_fit.params[added_term]),
                "added_variable_hc3_standard_error": float(full.robust_fit.bse[added_term]),
                "added_variable_hc3_ci_lower": float(conf.loc[added_term, "ci_lower"]),
                "added_variable_hc3_ci_upper": float(conf.loc[added_term, "ci_upper"]),
                "added_variable_hc3_p_value": float(full.robust_fit.pvalues[added_term]),
                "notes": "Classical partial F-test and HC3 robust inference use different assumptions.",
            }
        )
    return pd.DataFrame(rows)


def vif_table(model_results: dict[str, ModelResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, result in model_results.items():
        exog = result.classical_fit.model.exog
        names = result.classical_fit.model.exog_names
        for idx, name in enumerate(names):
            if name == "Intercept":
                continue
            rows.append(
                {
                    "model": model,
                    "term": name,
                    "vif": float(variance_inflation_factor(exog, idx)),
                }
            )
    return pd.DataFrame(rows)


def statsmodels_diagnostics(model_results: dict[str, ModelResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    influence_rows: list[dict[str, Any]] = []
    for model, result in model_results.items():
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
        flagged = np.asarray(
            (cooks_distance > cooks_threshold)
            | (np.abs(studentized_internal) > studentized_threshold)
            | (leverage > leverage_threshold),
            dtype=bool,
        )
        data = result.data.reset_index(drop=True)
        flagged_positions = np.flatnonzero(flagged)
        detected = set(data.iloc[flagged_positions]["mesaid"].astype(str))
        rows.append(
            {
                "model": model,
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
                "expected_model2_influential_mesaids": json.dumps(
                    sorted(EXPECTED_INFLUENTIAL_MODEL2)
                )
                if model == "Model 2"
                else "",
                "detected_influential_mesaids": json.dumps(sorted(detected)),
                "model2_expected_set_matches": detected == EXPECTED_INFLUENTIAL_MODEL2
                if model == "Model 2"
                else "",
            }
        )
        for idx in flagged_positions:
            influence_rows.append(
                {
                    "model": model,
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
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(influence_rows)


def main() -> None:
    setup_logging()
    backup_existing_outputs()
    env = check_environment()
    write_csv(env, ENVIRONMENT_PATH)

    pilot, manual_coef, manual_fit, manual_incremental, manual_sensitivity = load_inputs()
    # These files are loaded intentionally to verify availability and preserve the exact
    # previous pilot-analysis context; this validation does not overwrite them.
    logging.info("Loaded manual incremental rows: %s", len(manual_incremental))
    logging.info("Loaded manual sensitivity rows: %s", len(manual_sensitivity))

    sample, sample_checks = prepare_common_sample(pilot, manual_fit)
    model_results, stats_coef, stats_fit, design = fit_models(sample)
    stats_coef = stats_coef.merge(sample_checks.iloc[[0]][["value"]], how="cross").rename(
        columns={"value": "analysis_sample_n_check"}
    )

    coef_comparison = compare_coef_rows(manual_coef, stats_coef)
    fit_comparison = compare_model_fit(manual_fit, stats_fit)
    incremental = incremental_comparison(model_results)
    vif = vif_table(model_results)
    diagnostics, influential = statsmodels_diagnostics(model_results)

    write_csv(stats_coef, STATSMODELS_COEFFICIENTS_PATH)
    write_csv(stats_fit, STATSMODELS_MODEL_FIT_PATH)
    write_csv(design, STATSMODELS_DESIGN_MATRIX_PATH)
    write_csv(coef_comparison, COEFFICIENT_COMPARISON_PATH)
    write_csv(fit_comparison, MODEL_FIT_COMPARISON_PATH)
    write_csv(incremental, STATSMODELS_INCREMENTAL_PATH)
    write_csv(vif, STATSMODELS_VIF_PATH)
    write_csv(influential, STATSMODELS_INFLUENTIAL_PATH)
    write_csv(diagnostics, STATSMODELS_DIAGNOSTICS_PATH)
    write_csv(sample_checks, STATSMODELS_SAMPLE_VALIDATION_PATH)

    print("Statsmodels validation complete")
    print(f"analysis_n: {len(sample)}")
    print(f"coefficient_rows_outside_tolerance: {int((~coef_comparison['within_tolerance']).sum())}")
    print(f"model_fit_rows_outside_tolerance: {int((~fit_comparison['within_tolerance']).sum())}")


if __name__ == "__main__":
    main()
