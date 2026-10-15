"""Create publication-style Table 1 for the final complete-case sample.

This is a descriptive summary only. It uses the existing full regression
dataset, keeps the validated primary window and complete-case definition, and
does not alter raw data or prior analysis outputs.
"""

from __future__ import annotations

import logging
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
BACKUP_DIR = PROJECT_ROOT / "outputs" / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")

FULL_DATASET_PATH = PROCESSED_DIR / "full_regression_dataset.csv"
TABLE1_CSV_PATH = OUTPUT_TABLE_DIR / "table1_full_analysis_descriptive_statistics.csv"
TABLE1_MD_PATH = OUTPUT_TABLE_DIR / "table1_full_analysis_descriptive_statistics.md"
LOG_PATH = OUTPUT_LOG_DIR / "11_create_table1_descriptive_statistics.log"

REQUESTED_OUTPUTS = [TABLE1_CSV_PATH, TABLE1_MD_PATH]
PRIMARY_FEATURE_SET = "consecutive_7_daybynoon"
EXPECTED_FINAL_N = 1527
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1")


@dataclass(frozen=True)
class ContinuousVariable:
    section: str
    variable: str
    label: str
    unit: str
    decimals: int = 1
    notes: str = ""


CONTINUOUS_VARIABLES = [
    ContinuousVariable("Demographics", "age", "Age", "years"),
    ContinuousVariable("Demographics", "bmi", "Body mass index", "kg/m^2"),
    ContinuousVariable("Sleep apnea", "ahi", "Apnea-hypopnea index", "events/hour"),
    ContinuousVariable("Sleepiness", "ess", "Epworth Sleepiness Scale", "score"),
    ContinuousVariable("Actigraphy sleep", "sleep_duration_mean", "Mean sleep duration", "minutes"),
    ContinuousVariable("Actigraphy sleep", "sleep_duration_sd", "Sleep duration SD", "minutes"),
    ContinuousVariable("Actigraphy sleep", "sleep_efficiency_mean", "Mean sleep efficiency", "percent"),
    ContinuousVariable(
        "Actigraphy sleep",
        "sleep_efficiency_sd",
        "Sleep efficiency SD",
        "percentage points",
    ),
    ContinuousVariable("Actigraphy sleep", "waso_mean", "Mean WASO", "minutes"),
    ContinuousVariable("Actigraphy sleep", "waso_sd", "WASO SD", "minutes"),
    ContinuousVariable(
        "Actigraphy sleep",
        "sleep_midpoint_circular_sd",
        "Sleep midpoint circular SD",
        "circular minutes on 24-hour clock",
        notes="Participant-level circular SD from existing feature column.",
    ),
]


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


def backup_existing_outputs() -> None:
    existing = [path for path in REQUESTED_OUTPUTS if path.exists()]
    if not existing:
        logging.info("No requested Table 1 outputs existed before this run.")
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for path in existing:
        destination = BACKUP_DIR / path.name
        shutil.copy2(path, destination)
        logging.info("Backed up %s to %s", path, destination)


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logging.info("Wrote %s rows to %s", len(df), path)


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def format_number(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{decimals}f}"


def clock_time(minutes: float) -> str:
    if pd.isna(minutes):
        return ""
    total = int(round(float(minutes))) % 1440
    return f"{total // 60:02d}:{total % 60:02d}"


def circular_stats_minutes(series: pd.Series) -> tuple[float, float]:
    values = numeric_series(series).dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    angles = values / 1440.0 * 2 * math.pi
    mean_sin = float(np.sin(angles).mean())
    mean_cos = float(np.cos(angles).mean())
    mean_angle = math.atan2(mean_sin, mean_cos)
    if mean_angle < 0:
        mean_angle += 2 * math.pi
    mean_minutes = mean_angle / (2 * math.pi) * 1440.0
    if len(values) < 2:
        return float(mean_minutes), float("nan")
    r = math.hypot(mean_sin, mean_cos)
    if r <= 0:
        return float(mean_minutes), float("nan")
    sd_minutes = math.sqrt(max(-2 * math.log(min(r, 1.0)), 0.0)) / (2 * math.pi) * 1440.0
    return float(mean_minutes), float(sd_minutes)


def complete_case_mask(df: pd.DataFrame) -> pd.Series:
    if "primary_common_complete_case" not in df.columns:
        raise RuntimeError("primary_common_complete_case column was not found.")
    value = df["primary_common_complete_case"]
    if value.dtype == bool:
        return value
    return value.astype(str).str.lower().isin(["true", "1", "yes"])


def load_final_sample() -> pd.DataFrame:
    if not FULL_DATASET_PATH.exists():
        raise FileNotFoundError(f"Required input not found: {FULL_DATASET_PATH}")
    df = read_csv_with_fallback(FULL_DATASET_PATH)
    primary = df.loc[df["feature_set"].astype(str).eq(PRIMARY_FEATURE_SET)].copy()
    sample = primary.loc[complete_case_mask(primary)].copy()
    if len(sample) != EXPECTED_FINAL_N:
        raise RuntimeError(
            f"Expected final complete-case n={EXPECTED_FINAL_N}, observed n={len(sample)}."
        )
    logging.info("Loaded final complete-case sample: %s rows.", len(sample))
    return sample


def continuous_summary_row(sample: pd.DataFrame, spec: ContinuousVariable) -> dict[str, Any]:
    values = numeric_series(sample[spec.variable]).dropna()
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    median = float(values.median())
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    minimum = float(values.min())
    maximum = float(values.max())
    return {
        "section": spec.section,
        "characteristic": spec.label,
        "category": "",
        "source_variable": spec.variable,
        "unit": spec.unit,
        "n": int(values.count()),
        "missing_n": int(len(sample) - values.count()),
        "summary": f"{format_number(mean, spec.decimals)} ({format_number(sd, spec.decimals)})",
        "summary_type": "mean (SD)",
        "mean": mean,
        "sd": sd,
        "median": median,
        "q1": q1,
        "q3": q3,
        "min": minimum,
        "max": maximum,
        "median_iqr": (
            f"{format_number(median, spec.decimals)} "
            f"[{format_number(q1, spec.decimals)}, {format_number(q3, spec.decimals)}]"
        ),
        "min_max": f"{format_number(minimum, spec.decimals)}, {format_number(maximum, spec.decimals)}",
        "notes": spec.notes,
    }


def sex_rows(sample: pd.DataFrame) -> list[dict[str, Any]]:
    counts = sample["sex_category"].value_counts(dropna=False)
    rows: list[dict[str, Any]] = []
    for category in ["Female", "Male"]:
        count = int(counts.get(category, 0))
        percent = count / len(sample) * 100
        rows.append(
            {
                "section": "Demographics",
                "characteristic": "Sex",
                "category": category,
                "source_variable": "sex_category",
                "unit": "n (%)",
                "n": count,
                "missing_n": int(sample["sex_category"].isna().sum()),
                "summary": f"{count:,} ({percent:.1f})",
                "summary_type": "n (%)",
                "mean": np.nan,
                "sd": np.nan,
                "median": np.nan,
                "q1": np.nan,
                "q3": np.nan,
                "min": np.nan,
                "max": np.nan,
                "median_iqr": "",
                "min_max": "",
                "notes": "Sex coding follows existing analysis: sex_Male=1 for Male, 0 for Female.",
            }
        )
    return rows


def midpoint_mean_row(sample: pd.DataFrame) -> dict[str, Any]:
    variable = "sleep_midpoint_circular_mean"
    values = numeric_series(sample[variable]).dropna()
    mean_minutes, circular_sd = circular_stats_minutes(values)
    median = float(values.median())
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    minimum = float(values.min())
    maximum = float(values.max())
    return {
        "section": "Actigraphy sleep",
        "characteristic": "Mean sleep midpoint",
        "category": "",
        "source_variable": variable,
        "unit": "clock time; circular summary",
        "n": int(values.count()),
        "missing_n": int(len(sample) - values.count()),
        "summary": f"{clock_time(mean_minutes)} (circular SD {format_number(circular_sd, 1)} min)",
        "summary_type": "circular mean (circular SD)",
        "mean": mean_minutes,
        "sd": circular_sd,
        "median": median,
        "q1": q1,
        "q3": q3,
        "min": minimum,
        "max": maximum,
        "median_iqr": f"{clock_time(median)} [{clock_time(q1)}, {clock_time(q3)}]",
        "min_max": f"{clock_time(minimum)}, {clock_time(maximum)}",
        "notes": (
            "Between-participant summary uses circular statistics on the existing "
            "participant-level circular mean column; ordinary clock-time mean was not calculated."
        ),
    }


def build_table1(sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "section": "Sample",
            "characteristic": "Final complete-case sample",
            "category": "",
            "source_variable": "primary_common_complete_case",
            "unit": "n",
            "n": len(sample),
            "missing_n": 0,
            "summary": f"{len(sample):,}",
            "summary_type": "n",
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median_iqr": "",
            "min_max": "",
            "notes": f"Primary window={PRIMARY_FEATURE_SET}; final full complete-case n={EXPECTED_FINAL_N}.",
        }
    )
    rows.append(continuous_summary_row(sample, CONTINUOUS_VARIABLES[0]))
    rows.extend(sex_rows(sample))
    for spec in CONTINUOUS_VARIABLES[1:10]:
        rows.append(continuous_summary_row(sample, spec))
    rows.append(midpoint_mean_row(sample))
    rows.append(continuous_summary_row(sample, CONTINUOUS_VARIABLES[10]))
    return pd.DataFrame(rows)


def markdown_table(table: pd.DataFrame) -> str:
    display = table[["section", "characteristic", "category", "unit", "summary", "median_iqr"]].copy()
    display = display.rename(
        columns={
            "section": "Section",
            "characteristic": "Characteristic",
            "category": "Category",
            "unit": "Unit",
            "summary": f"Overall (N={EXPECTED_FINAL_N})",
            "median_iqr": "Median [Q1, Q3]",
        }
    )
    lines = [
        f"# Table 1. Characteristics of the final analytic sample (N={EXPECTED_FINAL_N})",
        "",
        "| Section | Characteristic | Category | Unit | Overall | Median [Q1, Q3] |",
        "|---|---|---|---|---:|---:|",
    ]
    for _, row in display.iterrows():
        lines.append(
            "| "
            + " | ".join(str(row[column]) if pd.notna(row[column]) else "" for column in display.columns)
            + " |"
        )
    lines.extend(
        [
            "",
            "Notes: Values are mean (SD) unless otherwise noted. Sex is n (%). "
            "Mean sleep midpoint is summarized with circular statistics using the existing "
            "sleep_midpoint_circular_mean feature; no ordinary clock-time mean was recalculated.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    setup_logging()
    backup_existing_outputs()
    sample = load_final_sample()
    table1 = build_table1(sample)
    write_csv(table1, TABLE1_CSV_PATH)
    TABLE1_MD_PATH.write_text(markdown_table(table1), encoding="utf-8")
    logging.info("Wrote Markdown table to %s", TABLE1_MD_PATH)
    print("Table 1 descriptive statistics complete")
    print(f"Final analytic sample n={len(sample)}")
    for _, row in table1.iterrows():
        if row["characteristic"] in [
            "Age",
            "Sex",
            "Body mass index",
            "Apnea-hypopnea index",
            "Epworth Sleepiness Scale",
            "Mean sleep duration",
            "Sleep duration SD",
            "Mean sleep midpoint",
        ]:
            label = row["characteristic"] if not row["category"] else f"{row['characteristic']} - {row['category']}"
            print(f"{label}: {row['summary']}")


if __name__ == "__main__":
    main()
