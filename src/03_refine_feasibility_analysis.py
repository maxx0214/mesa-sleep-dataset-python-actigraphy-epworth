"""Refine and audit the first-pass actigraphy feasibility outputs.

This script reads prior feasibility outputs and raw metadata, then writes
second-pass review tables. It does not modify raw data, merge phenotypes, or run
regression models.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ACTIGRAPHY_DIR = RAW_DIR / "actigraphy"
DATASET_DIR = RAW_DIR / "datasets"
DOCUMENTATION_DIR = RAW_DIR / "documentation"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
BACKUP_DIR = PROJECT_ROOT / "outputs" / "backups" / datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

REST_S_BLOCKS_PATH = INTERIM_DIR / "rest_s_blocks.csv"
DAILY_METRICS_PATH = INTERIM_DIR / "provisional_daily_sleep_metrics.csv"

DEFINITION_REVIEW_PATH = OUTPUT_TABLE_DIR / "actigraphy_definition_review.csv"
QC_FLAGS_PATH = OUTPUT_TABLE_DIR / "qc_flags_by_analysis_level.csv"
SENSITIVITY_PATH = OUTPUT_TABLE_DIR / "valid_night_sensitivity_analysis.csv"
DESCRIPTIVE_PATH = OUTPUT_TABLE_DIR / "provisional_main_sleep_descriptive_statistics.csv"
PHENOTYPE_LOCATION_PATH = OUTPUT_TABLE_DIR / "phenotype_variable_location.csv"
LOG_PATH = OUTPUT_LOG_DIR / "03_refine_feasibility_analysis.log"

CSV_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1")
ACTIGRAPHY_PATTERN = "mesa-sleep-*.csv"

DEFINITION_TERMS = [
    "REST",
    "REST-S",
    "ACTIVE",
    "EXCLUDED",
    "wake",
    "offwrist",
    "daybynoon",
    "daybymidnight",
    "avgmainsleep5",
    "avgefficiency5",
    "avgwaso5",
]

PHENOTYPE_VARIABLES = [
    "epslpscl5c",
    "avgmainsleep5",
    "avgefficiency5",
    "avgwaso5",
    "avgsleepmidpoint5",
    "nsrr_age",
    "nsrr_sex",
    "nsrr_bmi",
    "nsrr_ahi_hp3r_aasm15",
    "mesaid",
    "nsrrid",
]

QC_FLAGS = [
    "duration_under_3h",
    "duration_over_16h",
    "high_missing_wake",
    "contains_offwrist",
    "non_contiguous_epochs",
    "partial_first_recording_day",
    "partial_last_recording_day",
    "multiple_blocks_same_day",
    "invalid_sleep_efficiency",
    "missing_sleep_epochs",
]

REQUESTED_OUTPUTS = [
    DEFINITION_REVIEW_PATH,
    QC_FLAGS_PATH,
    SENSITIVITY_PATH,
    DESCRIPTIVE_PATH,
    PHENOTYPE_LOCATION_PATH,
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


def backup_existing_outputs() -> list[dict[str, str]]:
    backups: list[dict[str, str]] = []
    existing = [path for path in REQUESTED_OUTPUTS if path.exists()]
    if not existing:
        logging.info("No requested output tables existed before this run.")
        return backups

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for path in existing:
        destination = BACKUP_DIR / path.name
        shutil.copy2(path, destination)
        backups.append(
            {
                "source": relative_path(path),
                "backup": relative_path(destination),
            }
        )
        logging.info("Backed up %s to %s", path, destination)
    return backups


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype("string").str.lower().isin(["true", "1", "yes"])


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def participant_key(df: pd.DataFrame) -> pd.Series:
    mesaid = df.get("mesaid", pd.Series(index=df.index, dtype="object")).astype("string")
    source_file = df.get("source_file", pd.Series(index=df.index, dtype="object")).astype("string")
    return mesaid.where(mesaid.notna() & mesaid.ne(""), source_file)


def file_count_and_examples(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "file_count": 0, "first_10_files": []}
    files = sorted(item for item in path.rglob("*") if item.is_file())
    return {
        "exists": True,
        "file_count": len(files),
        "first_10_files": [relative_path(item) for item in files[:10]],
    }


def inspect_raw_paths() -> dict[str, Any]:
    actigraphy_files = sorted(ACTIGRAPHY_DIR.glob(ACTIGRAPHY_PATTERN)) if ACTIGRAPHY_DIR.exists() else []
    path_summary = {
        "actigraphy": {
            **file_count_and_examples(ACTIGRAPHY_DIR),
            "mesa_sleep_csv_count": len(actigraphy_files),
            "first_10_mesa_sleep_csv": [path.name for path in actigraphy_files[:10]],
        },
        "datasets": file_count_and_examples(DATASET_DIR),
        "documentation": file_count_and_examples(DOCUMENTATION_DIR),
    }
    logging.info("Raw path summary: %s", json.dumps(path_summary, ensure_ascii=False))
    return path_summary


def get_current_dictionary() -> pd.DataFrame:
    dictionary_path = DATASET_DIR / "mesa-data-dictionary-0.8.0-variables.csv"
    if not dictionary_path.exists():
        return pd.DataFrame()
    dictionary = read_csv_with_fallback(dictionary_path)
    dictionary["_source_file"] = relative_path(dictionary_path)
    dictionary["_row_number"] = np.arange(2, len(dictionary) + 2)
    return dictionary


def definition_implication(term: str, confirmed: bool) -> str:
    if not confirmed:
        if term in {"REST", "REST-S", "ACTIVE", "EXCLUDED"}:
            return "Keep interval label interpretation provisional; do not treat this as an official scored-state definition."
        if term in {"wake", "offwrist", "daybynoon", "daybymidnight"}:
            return "Use as an observed raw column only; official meaning was not confirmed in available documentation."
        return "Do not rely on this term until a definition is found."
    implications = {
        "avgmainsleep5": "Supports comparing derived main-sleep duration with an official actigraphy phenotype, but it is not a raw epoch definition.",
        "avgefficiency5": "Confirms sleep efficiency as sleep duration divided by rest interval for main sleep.",
        "avgwaso5": "Confirms WASO as wake duration between sleep onset and sleep offset in main sleep.",
    }
    return implications.get(term, "Definition found in the current MESA data dictionary.")


def build_definition_review() -> pd.DataFrame:
    dictionary = get_current_dictionary()
    rows: list[dict[str, Any]] = []
    dictionary_by_id = {}
    if not dictionary.empty and "id" in dictionary:
        for _, item in dictionary.iterrows():
            dictionary_by_id[str(item["id"])] = item

    for term in DEFINITION_TERMS:
        if term in dictionary_by_id:
            item = dictionary_by_id[term]
            definition = str(item.get("display_name", "")).strip()
            description = str(item.get("description", "")).strip()
            if description and description.lower() != "nan":
                definition = f"{definition}: {description}"
            rows.append(
                {
                    "variable_or_label": term,
                    "official_definition": definition,
                    "source_file": item["_source_file"],
                    "source_section_or_row": f"id={term}; csv_row={int(item['_row_number'])}",
                    "implication_for_analysis": definition_implication(term, True),
                    "interpretation_confirmed": "confirmed",
                }
            )
        else:
            rows.append(
                {
                    "variable_or_label": term,
                    "official_definition": "not_confirmed",
                    "source_file": "not_found_in_data_raw_documentation_or_current_data_dictionary",
                    "source_section_or_row": "not_confirmed",
                    "implication_for_analysis": definition_implication(term, False),
                    "interpretation_confirmed": "not_confirmed",
                }
            )

    return pd.DataFrame(rows)


def qc_by_level(blocks: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    levels = [
        ("all_rest_s_candidate_blocks", blocks.copy(), "REST-S candidate block"),
        (
            "selected_provisional_main_sleep_blocks",
            blocks.loc[bool_series(blocks["selected_as_provisional_main_sleep"])].copy(),
            "selected main sleep block",
        ),
        (
            "provisional_valid_nights",
            daily.loc[bool_series(daily["provisional_valid_night"])].copy(),
            "valid selected night",
        ),
    ]

    rows: list[dict[str, Any]] = []
    for level_name, level_df, unit_label in levels:
        keys = participant_key(level_df) if not level_df.empty else pd.Series(dtype="string")
        total_participants = int(keys.nunique(dropna=True))
        total_units = int(len(level_df))
        for flag in QC_FLAGS:
            if level_df.empty or flag not in level_df:
                flagged = pd.DataFrame()
            else:
                flagged = level_df.loc[bool_series(level_df[flag])]
            flagged_keys = participant_key(flagged) if not flagged.empty else pd.Series(dtype="string")
            rows.append(
                {
                    "analysis_level": level_name,
                    "analysis_unit": unit_label,
                    "quality_flag": flag,
                    "participant_count": int(flagged_keys.nunique(dropna=True)),
                    "night_count": int(len(flagged)),
                    "total_participants_in_level": total_participants,
                    "total_units_in_level": total_units,
                }
            )
    return pd.DataFrame(rows)


def criterion_mask(selected: pd.DataFrame, criterion: str) -> pd.Series:
    selected = selected.copy()
    offwrist_ratio = (
        numeric_series(selected["offwrist_epochs"]) / numeric_series(selected["total_epochs"])
    ).replace([np.inf, -np.inf], np.nan).fillna(1)

    common = (
        ~bool_series(selected["duration_under_3h"])
        & ~bool_series(selected["duration_over_16h"])
        & ~bool_series(selected["high_missing_wake"])
        & ~bool_series(selected["non_contiguous_epochs"])
        & ~bool_series(selected["invalid_sleep_efficiency"])
        & ~bool_series(selected["missing_sleep_epochs"])
    )
    exclude_partial = (
        ~bool_series(selected["partial_first_recording_day"])
        & ~bool_series(selected["partial_last_recording_day"])
    )
    zero_offwrist = ~bool_series(selected["contains_offwrist"])
    low_offwrist = offwrist_ratio.le(0.05)

    if criterion == "A":
        return common & exclude_partial & zero_offwrist
    if criterion == "B":
        return common & zero_offwrist
    if criterion == "C":
        return common & exclude_partial & low_offwrist
    if criterion == "D":
        return common & low_offwrist
    raise ValueError(f"Unknown criterion: {criterion}")


def sd_available(df: pd.DataFrame, column: str) -> int:
    if df.empty:
        return 0
    counts = df.groupby("participant_key")[column].apply(
        lambda values: numeric_series(values).notna().sum()
    )
    return int(counts.ge(2).sum())


def cv_available(df: pd.DataFrame, column: str) -> int:
    if df.empty:
        return 0

    def can_compute(values: pd.Series) -> bool:
        numeric = numeric_series(values).dropna()
        return len(numeric) >= 2 and numeric.mean() > 0

    return int(df.groupby("participant_key")[column].apply(can_compute).sum())


def build_sensitivity(blocks: pd.DataFrame) -> pd.DataFrame:
    selected = blocks.loc[bool_series(blocks["selected_as_provisional_main_sleep"])].copy()
    selected["participant_key"] = participant_key(selected)
    criterion_descriptions = {
        "A": "Current criterion: exclude partial first/last recording days and require zero offwrist epochs.",
        "B": "Do not automatically exclude first/last recording days if the selected block itself passes duration, missingness, and continuity checks.",
        "C": "Keep partial-day exclusion but allow offwrist ratio <= 5% within the selected block.",
        "D": "Apply both B and C.",
    }

    rows: list[dict[str, Any]] = []
    for criterion, description in criterion_descriptions.items():
        valid = selected.loc[criterion_mask(selected, criterion)].copy()
        counts = valid.groupby("participant_key").size() if not valid.empty else pd.Series(dtype=float)
        rows.append(
            {
                "criterion": criterion,
                "criterion_description": description,
                "participants_with_1plus_valid_nights": int(counts.ge(1).sum()),
                "participants_with_2plus_valid_nights": int(counts.ge(2).sum()),
                "participants_with_5plus_valid_nights": int(counts.ge(5).sum()),
                "participants_with_6plus_valid_nights": int(counts.ge(6).sum()),
                "participants_with_7plus_valid_nights": int(counts.ge(7).sum()),
                "sleep_duration_sd_available_participant_count": sd_available(
                    valid, "sleep_duration_minutes"
                ),
                "sleep_duration_cv_available_participant_count": cv_available(
                    valid, "sleep_duration_minutes"
                ),
                "sleep_efficiency_sd_available_participant_count": sd_available(
                    valid, "sleep_efficiency"
                ),
                "waso_sd_available_participant_count": sd_available(valid, "waso_minutes"),
                "sleep_midpoint_sd_available_participant_count": sd_available(
                    valid, "sleep_midpoint_clock_minutes"
                ),
                "valid_night_count_mean": float(counts.mean()) if not counts.empty else 0,
                "valid_night_count_median": float(counts.median()) if not counts.empty else 0,
                "valid_night_count_min": int(counts.min()) if not counts.empty else 0,
                "valid_night_count_max": int(counts.max()) if not counts.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def descriptive_row(metric: str, series: pd.Series, units: str) -> dict[str, Any]:
    values = numeric_series(series).dropna()
    return {
        "metric": metric,
        "units": units,
        "count": int(values.count()),
        "mean": float(values.mean()) if not values.empty else np.nan,
        "sd": float(values.std(ddof=1)) if len(values) >= 2 else np.nan,
        "median": float(values.median()) if not values.empty else np.nan,
        "q1": float(values.quantile(0.25)) if not values.empty else np.nan,
        "q3": float(values.quantile(0.75)) if not values.empty else np.nan,
        "min": float(values.min()) if not values.empty else np.nan,
        "max": float(values.max()) if not values.empty else np.nan,
    }


def build_descriptive_statistics(blocks: pd.DataFrame) -> pd.DataFrame:
    selected = blocks.loc[bool_series(blocks["selected_as_provisional_main_sleep"])].copy()
    selected["participant_key"] = participant_key(selected)
    selected["wake_missing_percentage"] = numeric_series(selected["wake_missing_ratio"]) * 100
    selected["offwrist_percentage"] = (
        numeric_series(selected["offwrist_epochs"]) / numeric_series(selected["total_epochs"]) * 100
    )
    selected_night_counts = selected.groupby("participant_key").size()

    rows = [
        descriptive_row(
            "rest_interval_duration_minutes",
            selected["rest_interval_duration_minutes"],
            "minutes",
        ),
        descriptive_row("sleep_duration_minutes", selected["sleep_duration_minutes"], "minutes"),
        descriptive_row("sleep_efficiency", selected["sleep_efficiency"], "percent"),
        descriptive_row("waso_minutes", selected["waso_minutes"], "minutes"),
        descriptive_row(
            "sleep_midpoint_clock_minutes",
            selected["sleep_midpoint_clock_minutes"],
            "clock minutes after midnight",
        ),
        descriptive_row(
            "wake_missing_percentage",
            selected["wake_missing_percentage"],
            "percent",
        ),
        descriptive_row(
            "offwrist_percentage",
            selected["offwrist_percentage"],
            "percent",
        ),
        descriptive_row(
            "selected_night_count_per_participant",
            selected_night_counts,
            "nights",
        ),
    ]
    return pd.DataFrame(rows)


def csv_contains_values(path: Path, targets: set[str]) -> dict[str, list[int]]:
    found: dict[str, list[int]] = {target: [] for target in targets}
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as file_obj:
                reader = csv.reader(file_obj)
                for row_number, row in enumerate(reader, start=1):
                    row_values = {cell.strip() for cell in row}
                    for target in targets.intersection(row_values):
                        found[target].append(row_number)
            return found
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return found


def build_phenotype_variable_location() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    targets = set(PHENOTYPE_VARIABLES)
    dataset_files = sorted(DATASET_DIR.rglob("*.csv")) if DATASET_DIR.exists() else []

    for path in dataset_files:
        try:
            header = read_csv_with_fallback(path, nrows=0)
            columns = [str(column) for column in header.columns]
            value_hits = csv_contains_values(path, targets)
            file_label = relative_path(path)
            for variable in PHENOTYPE_VARIABLES:
                as_column = variable in columns
                rows_as_values = value_hits.get(variable, [])
                if as_column or rows_as_values:
                    rows.append(
                        {
                            "variable": variable,
                            "source_file": file_label,
                            "found_as_column": as_column,
                            "found_as_cell_value": bool(rows_as_values),
                            "cell_value_rows": json.dumps(rows_as_values[:20]),
                            "possible_actigraphy_link": (
                                "direct mesaid join candidate"
                                if variable == "mesaid"
                                else "requires a dataset containing mesaid/nsrrid mapping"
                                if variable == "nsrrid"
                                else "use source file IDs; actigraphy currently provides mesaid"
                            ),
                        }
                    )
        except Exception as exc:  # Keep location audit resilient across archived CSVs.
            rows.append(
                {
                    "variable": "__file_read_error__",
                    "source_file": relative_path(path),
                    "found_as_column": False,
                    "found_as_cell_value": False,
                    "cell_value_rows": "",
                    "possible_actigraphy_link": f"{type(exc).__name__}: {exc}",
                }
            )

    if not rows:
        rows.append(
            {
                "variable": "__no_dataset_csv_files_found__",
                "source_file": relative_path(DATASET_DIR),
                "found_as_column": False,
                "found_as_cell_value": False,
                "cell_value_rows": "",
                "possible_actigraphy_link": "No phenotype location can be assessed.",
            }
        )
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logging.info("Wrote %s rows to %s", len(df), path)


def main() -> None:
    setup_logging()
    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Current working directory: %s", Path.cwd())
    backup_existing_outputs()
    raw_summary = inspect_raw_paths()

    blocks = read_csv_with_fallback(REST_S_BLOCKS_PATH)
    daily = read_csv_with_fallback(DAILY_METRICS_PATH)

    definition_review = build_definition_review()
    qc_flags = qc_by_level(blocks, daily)
    sensitivity = build_sensitivity(blocks)
    descriptive = build_descriptive_statistics(blocks)
    phenotype_locations = build_phenotype_variable_location()

    write_csv(definition_review, DEFINITION_REVIEW_PATH)
    write_csv(qc_flags, QC_FLAGS_PATH)
    write_csv(sensitivity, SENSITIVITY_PATH)
    write_csv(descriptive, DESCRIPTIVE_PATH)
    write_csv(phenotype_locations, PHENOTYPE_LOCATION_PATH)

    confirmed = Counter(definition_review["interpretation_confirmed"])
    logging.info("Definition confirmation counts: %s", dict(confirmed))
    logging.info("Raw path summary: %s", json.dumps(raw_summary, ensure_ascii=False))
    print("Refined feasibility analysis complete")
    print(f"Actigraphy mesa-sleep CSV files: {raw_summary['actigraphy']['mesa_sleep_csv_count']}")
    print(f"Definition confirmed count: {confirmed.get('confirmed', 0)}")
    print(f"Definition not confirmed count: {confirmed.get('not_confirmed', 0)}")
    print(f"QC rows: {len(qc_flags)}")
    print(f"Sensitivity criteria rows: {len(sensitivity)}")


if __name__ == "__main__":
    main()
