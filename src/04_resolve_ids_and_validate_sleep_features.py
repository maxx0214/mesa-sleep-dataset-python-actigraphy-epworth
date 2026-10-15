"""Resolve ID linkage and validate derived sleep features before regression.

This script does not run ESS regression and does not modify raw data. It audits
ID linkage evidence, searches native covariate candidates, creates a boundary-
aware valid-night definition, builds fixed-window sleep features, and compares
derived actigraphy means with official main phenotype actigraphy variables.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ACTIGRAPHY_DIR = RAW_DIR / "actigraphy"
DATASET_DIR = RAW_DIR / "datasets"
DOCUMENTATION_DIR = RAW_DIR / "documentation"
OVERLAP_DIR = RAW_DIR / "overlap"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
BACKUP_DIR = PROJECT_ROOT / "outputs" / "backups" / datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

REST_S_BLOCKS_PATH = INTERIM_DIR / "rest_s_blocks.csv"
DAILY_METRICS_PATH = INTERIM_DIR / "provisional_daily_sleep_metrics.csv"
MAIN_DICTIONARY_PATH = DATASET_DIR / "mesa-data-dictionary-0.8.0-variables.csv"
MAIN_PHENOTYPE_PATH = DATASET_DIR / "mesa-sleep-dataset-0.8.0.csv"

RAW_PATH_INVENTORY_PATH = OUTPUT_TABLE_DIR / "raw_path_inventory.csv"
ID_MAPPING_REVIEW_PATH = OUTPUT_TABLE_DIR / "id_mapping_review.csv"
ID_MAPPING_QUALITY_PATH = OUTPUT_TABLE_DIR / "id_mapping_quality_summary.csv"
NATIVE_COVARIATE_PATH = OUTPUT_TABLE_DIR / "native_covariate_candidates.csv"
BOUNDARY_COMPARISON_PATH = OUTPUT_TABLE_DIR / "boundary_validity_comparison.csv"
SLEEP_FEATURES_PATH = INTERIM_DIR / "participant_sleep_features_by_window.csv"
SEVEN_NIGHT_SUMMARY_PATH = OUTPUT_TABLE_DIR / "seven_night_window_summary.csv"
METRIC_VALIDATION_PATH = OUTPUT_TABLE_DIR / "derived_vs_official_metric_validation.csv"
VALIDATION_DATASET_PATH = PROCESSED_DIR / "sleep_feature_validation_dataset.csv"
LOG_PATH = OUTPUT_LOG_DIR / "04_resolve_ids_and_validate_sleep_features.log"

REQUESTED_OUTPUTS = [
    RAW_PATH_INVENTORY_PATH,
    ID_MAPPING_REVIEW_PATH,
    ID_MAPPING_QUALITY_PATH,
    NATIVE_COVARIATE_PATH,
    BOUNDARY_COMPARISON_PATH,
    SLEEP_FEATURES_PATH,
    SEVEN_NIGHT_SUMMARY_PATH,
    METRIC_VALIDATION_PATH,
    VALIDATION_DATASET_PATH,
]

CSV_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1")
ID_SEARCH_TERMS = [
    "mesaid",
    "nsrrid",
    "subject_id",
    "participant_id",
    "crosswalk",
    "mapping",
]
OFFICIAL_VARIABLES = [
    "epslpscl5c",
    "avgmainsleep5",
    "avgefficiency5",
    "avgwaso5",
    "avgsleepmidpoint5",
]
QUALITY_FLAGS = [
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
EPOCH_SECONDS = 30
CONTEXT_MINUTES = 30
CONTEXT_EPOCHS = CONTEXT_MINUTES * 60 // EPOCH_SECONDS


def ensure_dirs() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
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
    existing = [path for path in REQUESTED_OUTPUTS if path.exists()]
    backups: list[dict[str, str]] = []
    if not existing:
        logging.info("No requested output files existed before this run.")
        return backups
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for path in existing:
        destination = BACKUP_DIR / path.name
        shutil.copy2(path, destination)
        backups.append({"source": relative_path(path), "backup": relative_path(destination)})
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
    source = df.get("source_file", pd.Series(index=df.index, dtype="object")).astype("string")
    return mesaid.where(mesaid.notna() & mesaid.ne(""), source)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logging.info("Wrote %s rows to %s", len(df), path)


def inspect_folder(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "folder": label,
            "path": relative_path(path),
            "exists": False,
            "file_count": 0,
            "extension_counts": "{}",
            "first_10_files": "[]",
        }
    files = sorted(item for item in path.rglob("*") if item.is_file())
    extension_counts = Counter(item.suffix.lower() or "<no_extension>" for item in files)
    return {
        "folder": label,
        "path": relative_path(path),
        "exists": True,
        "file_count": len(files),
        "extension_counts": json.dumps(dict(sorted(extension_counts.items())), sort_keys=True),
        "first_10_files": json.dumps([item.name for item in files[:10]], ensure_ascii=False),
    }


def build_raw_path_inventory() -> pd.DataFrame:
    rows = [
        inspect_folder(ACTIGRAPHY_DIR, "data/raw/actigraphy"),
        inspect_folder(DATASET_DIR, "data/raw/datasets"),
        inspect_folder(DOCUMENTATION_DIR, "data/raw/documentation"),
        inspect_folder(OVERLAP_DIR, "data/raw/overlap"),
    ]
    return pd.DataFrame(rows)


def iter_search_files() -> list[Path]:
    files: list[Path] = []
    for root in [DATASET_DIR, OVERLAP_DIR]:
        if root.exists():
            files.extend(sorted(item for item in root.rglob("*") if item.is_file()))
    return files


def read_header_and_count_rows(path: Path) -> tuple[list[str], int]:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as file_obj:
                reader = csv.reader(file_obj)
                try:
                    header = next(reader)
                except StopIteration:
                    return [], 0
                return [cell.strip() for cell in header], sum(1 for _ in reader)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return [], 0


def search_file_terms(path: Path, terms: list[str]) -> tuple[list[str], dict[str, int]]:
    term_counts = {term: 0 for term in terms}
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, errors="strict") as file_obj:
                for line in file_obj:
                    lower = line.lower()
                    for term in terms:
                        term_counts[term] += lower.count(term.lower())
            found = [term for term, count in term_counts.items() if count > 0]
            return found, term_counts
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return [], term_counts


def validate_mapping_file(path: Path) -> dict[str, Any]:
    df = read_csv_with_fallback(path, dtype=str)
    if "mesaid" not in df.columns or "nsrrid" not in df.columns:
        return {}
    mapping = df[["mesaid", "nsrrid"]].dropna()
    mapping = mapping.loc[mapping["mesaid"].astype(str).str.strip().ne("")]
    mapping = mapping.loc[mapping["nsrrid"].astype(str).str.strip().ne("")]
    duplicate_mesaid = int(mapping["mesaid"].duplicated(keep=False).sum())
    duplicate_nsrrid = int(mapping["nsrrid"].duplicated(keep=False).sum())
    many_to_one = int(mapping.groupby("nsrrid")["mesaid"].nunique().gt(1).sum())
    one_to_many = int(mapping.groupby("mesaid")["nsrrid"].nunique().gt(1).sum())
    return {
        "candidate_mapping_rows": int(len(mapping)),
        "unique_mesaid": int(mapping["mesaid"].nunique()),
        "unique_nsrrid": int(mapping["nsrrid"].nunique()),
        "duplicate_mesaid_rows": duplicate_mesaid,
        "duplicate_nsrrid_rows": duplicate_nsrrid,
        "one_to_many_mesaid_count": one_to_many,
        "many_to_one_nsrrid_count": many_to_one,
        "one_to_one_verified": bool(
            len(mapping) > 0
            and duplicate_mesaid == 0
            and duplicate_nsrrid == 0
            and one_to_many == 0
            and many_to_one == 0
        ),
    }


def build_id_mapping_review() -> tuple[pd.DataFrame, pd.DataFrame]:
    review_rows: list[dict[str, Any]] = []
    mapping_candidates: list[dict[str, Any]] = []

    for path in iter_search_files():
        row: dict[str, Any] = {
            "source_file": relative_path(path),
            "extension": path.suffix.lower() or "<no_extension>",
            "readable": False,
            "row_count": None,
            "columns_found": "[]",
            "string_hits": "[]",
            "string_hit_counts": "{}",
            "has_mesaid_column": False,
            "has_nsrrid_column": False,
            "has_subject_id_column": False,
            "has_participant_id_column": False,
            "candidate_official_crosswalk": False,
            "mapping_quality_notes": "",
        }
        try:
            if path.suffix.lower() == ".csv":
                header, row_count = read_header_and_count_rows(path)
                lower_header = [name.lower() for name in header]
                row["row_count"] = row_count
                row["columns_found"] = json.dumps(header, ensure_ascii=False)
                row["has_mesaid_column"] = "mesaid" in lower_header
                row["has_nsrrid_column"] = "nsrrid" in lower_header
                row["has_subject_id_column"] = "subject_id" in lower_header
                row["has_participant_id_column"] = "participant_id" in lower_header
                row["candidate_official_crosswalk"] = (
                    row["has_mesaid_column"] and row["has_nsrrid_column"] and row_count > 0
                )
                if row["candidate_official_crosswalk"]:
                    mapping_quality = validate_mapping_file(path)
                    row["mapping_quality_notes"] = json.dumps(mapping_quality, sort_keys=True)
                    mapping_candidates.append({"source_file": relative_path(path), **mapping_quality})
            hits, hit_counts = search_file_terms(path, ID_SEARCH_TERMS)
            row["string_hits"] = json.dumps(hits, ensure_ascii=False)
            row["string_hit_counts"] = json.dumps(hit_counts, sort_keys=True)
            row["readable"] = True
        except Exception as exc:
            row["mapping_quality_notes"] = f"{type(exc).__name__}: {exc}"
        review_rows.append(row)

    review_df = pd.DataFrame(review_rows)
    if mapping_candidates:
        best = pd.DataFrame(mapping_candidates).sort_values(
            ["one_to_one_verified", "candidate_mapping_rows"], ascending=[False, False]
        ).iloc[0]
        official_found = bool(best["one_to_one_verified"])
        harmonized_allowed = official_found
        mapping_source = best["source_file"]
        notes = "Official crosswalk candidate with mesaid and nsrrid columns was one-to-one."
    else:
        official_found = False
        harmonized_allowed = False
        mapping_source = ""
        notes = (
            "No readable data/raw/datasets or data/raw/overlap file contained both "
            "mesaid and nsrrid as data columns. Harmonized merge is not allowed."
        )

    quality_rows = [
        {
            "metric": "files_searched",
            "value": int(len(review_df)),
            "notes": "Files under data/raw/datasets and data/raw/overlap.",
        },
        {
            "metric": "files_with_mesaid_column",
            "value": int(review_df["has_mesaid_column"].fillna(False).sum())
            if not review_df.empty
            else 0,
            "notes": "",
        },
        {
            "metric": "files_with_nsrrid_column",
            "value": int(review_df["has_nsrrid_column"].fillna(False).sum())
            if not review_df.empty
            else 0,
            "notes": "",
        },
        {
            "metric": "files_with_both_mesaid_and_nsrrid_columns",
            "value": int(review_df["candidate_official_crosswalk"].fillna(False).sum())
            if not review_df.empty
            else 0,
            "notes": "",
        },
        {
            "metric": "official_crosswalk_found",
            "value": official_found,
            "notes": notes,
        },
        {
            "metric": "mapping_source_file",
            "value": mapping_source,
            "notes": "",
        },
        {
            "metric": "harmonized_dataset_merge_allowed",
            "value": harmonized_allowed,
            "notes": "Only true when a one-to-one mesaid/nsrrid crosswalk is found.",
        },
    ]
    return review_df, pd.DataFrame(quality_rows)


def load_dictionary() -> pd.DataFrame:
    dictionary = read_csv_with_fallback(MAIN_DICTIONARY_PATH)
    dictionary["_source_file"] = relative_path(MAIN_DICTIONARY_PATH)
    return dictionary


def load_main_phenotype(columns: list[str] | None = None) -> pd.DataFrame:
    if columns is None:
        return read_csv_with_fallback(MAIN_PHENOTYPE_PATH)
    return read_csv_with_fallback(MAIN_PHENOTYPE_PATH, usecols=columns)


def get_harmonized_source_refs(dictionary: pd.DataFrame) -> dict[str, str]:
    refs: dict[str, str] = {}
    targets = {
        "nsrr_age": "age",
        "nsrr_sex": "sex",
        "nsrr_bmi": "BMI",
        "nsrr_ahi_hp3r_aasm15": "AHI",
    }
    for variable, concept in targets.items():
        match = dictionary.loc[dictionary["id"].eq(variable)]
        if match.empty:
            continue
        text = " ".join(str(match.iloc[0].get(col, "")) for col in ["description", "calculation"])
        refs[concept] = ""
        marker_start = text.find("Source: :")
        if marker_start >= 0:
            marker_start += len("Source: :")
            marker_end = text.find(":", marker_start)
            if marker_end > marker_start:
                refs[concept] = text[marker_start:marker_end]
    return refs


def concept_matches(concept: str, text: str) -> bool:
    text = text.lower()
    if concept == "age":
        return "age" in text
    if concept == "sex":
        return "sex" in text or "gender" in text
    if concept == "BMI":
        return "bmi" in text or "body mass index" in text
    if concept == "apnea-hypopnea index":
        return "apnea-hypopnea index" in text or "apnea hypopnea index" in text
    if concept == "AHI":
        return "ahi" in text or "apnea-hypopnea index" in text
    if concept == "respiratory event index":
        return "respiratory event index" in text or "rdi" in text
    return False


def build_native_covariate_candidates(dictionary: pd.DataFrame, main_df: pd.DataFrame) -> pd.DataFrame:
    source_refs = get_harmonized_source_refs(dictionary)
    main_columns = set(main_df.columns)
    concepts = [
        "age",
        "sex",
        "BMI",
        "apnea-hypopnea index",
        "AHI",
        "respiratory event index",
    ]
    rows: list[dict[str, Any]] = []
    for _, item in dictionary.iterrows():
        variable = str(item.get("id", ""))
        if variable not in main_columns:
            continue
        search_text = " ".join(
            str(item.get(col, ""))
            for col in ["id", "display_name", "description", "domain", "units"]
        )
        for concept in concepts:
            if not concept_matches(concept, search_text):
                continue
            values = main_df[variable]
            non_missing = int(values.notna().sum())
            unique_count = int(values.nunique(dropna=True))
            priority = 3
            reason = "Keyword match in variable name, label, description, domain, or units."
            confirmed = False
            if source_refs.get(concept) == variable or (
                concept in {"apnea-hypopnea index", "AHI"} and source_refs.get("AHI") == variable
            ):
                priority = 1
                reason = "Matches source variable referenced by the harmonized NSRR variable definition."
                confirmed = True
            elif concept == "age" and variable == "sleepage5c":
                priority = 1
                reason = "Dictionary label directly identifies participant age at MESA Sleep Visit."
                confirmed = True
            elif concept == "sex" and variable == "gender1":
                priority = 1
                reason = "Dictionary label identifies participant gender; usable as native sex/gender covariate."
                confirmed = True
            elif concept == "BMI" and variable == "bmi5c":
                priority = 1
                reason = "Dictionary label and units identify body mass index."
                confirmed = True
            elif (
                concept in {"AHI", "apnea-hypopnea index"}
                and variable == "ahi_a0h3a"
            ):
                priority = 1
                reason = "Dictionary definition is the AASM-recommended AHI source used by the harmonized variable."
                confirmed = True
            elif "calculated" in str(item.get("description", "")).lower() and (
                "apnea-hypopnea index" in str(item.get("display_name", "")).lower()
            ):
                priority = 2
                reason = "Clear calculated AHI definition, but not selected as the primary source reference."

            rows.append(
                {
                    "concept": concept,
                    "variable_name": variable,
                    "label": item.get("display_name", ""),
                    "description": item.get("description", ""),
                    "units": item.get("units", ""),
                    "source_file": item.get("_source_file", relative_path(MAIN_DICTIONARY_PATH)),
                    "non_missing_count": non_missing,
                    "unique_value_count": unique_count,
                    "candidate_priority": priority,
                    "reason": reason,
                    "confirmed_for_use": bool(confirmed and non_missing > 0),
                }
            )
    return pd.DataFrame(rows).sort_values(["concept", "candidate_priority", "variable_name"])


def parse_actigraphy_datetime(df: pd.DataFrame) -> pd.Series:
    date_part = pd.to_datetime(df["date_shifted"], errors="coerce")
    date_text = date_part.dt.strftime("%Y-%m-%d").astype("string")
    time_text = df["linetime"].astype("string").str.strip()
    time_match = time_text.str.extract(r"(\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)")[0]
    combined = (date_text.fillna("") + " " + time_match.astype("string").fillna("")).str.strip()
    parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for date_format in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        missing = parsed.isna() & combined.ne("")
        if not missing.any():
            break
        parsed.loc[missing] = pd.to_datetime(combined.loc[missing], format=date_format, errors="coerce")
    return parsed


def load_source_timestamps(source_file: str) -> pd.Series:
    path = PROJECT_ROOT / source_file
    df = read_csv_with_fallback(path, usecols=["date_shifted", "linetime"])
    timestamps = parse_actigraphy_datetime(df).dropna().drop_duplicates().sort_values()
    return pd.Series(timestamps.array)


def add_boundary_context(blocks: pd.DataFrame) -> pd.DataFrame:
    selected = blocks.loc[bool_series(blocks["selected_as_provisional_main_sleep"])].copy()
    selected["block_start_dt"] = pd.to_datetime(selected["block_start"], errors="coerce")
    selected["block_end_dt"] = pd.to_datetime(selected["block_end"], errors="coerce")
    selected["participant_key"] = participant_key(selected)

    cache: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    for index, row in selected.iterrows():
        source_file = str(row["source_file"])
        if source_file not in cache:
            cache[source_file] = load_source_timestamps(source_file)
        timestamps = cache[source_file]
        first_timestamp = timestamps.iloc[0] if len(timestamps) else pd.NaT
        last_timestamp = timestamps.iloc[-1] if len(timestamps) else pd.NaT
        last_endpoint = last_timestamp + pd.Timedelta(seconds=EPOCH_SECONDS) if pd.notna(last_timestamp) else pd.NaT
        block_start = row["block_start_dt"]
        block_end = row["block_end_dt"]

        start_distance = abs((block_start - first_timestamp).total_seconds()) if pd.notna(block_start) and pd.notna(first_timestamp) else np.nan
        end_distance = min(
            abs((block_end - last_timestamp).total_seconds()) if pd.notna(block_end) and pd.notna(last_timestamp) else np.inf,
            abs((block_end - last_endpoint).total_seconds()) if pd.notna(block_end) and pd.notna(last_endpoint) else np.inf,
        )
        pre_start = block_start - pd.Timedelta(minutes=CONTEXT_MINUTES)
        post_end = block_end + pd.Timedelta(minutes=CONTEXT_MINUTES)
        pre_context_count = int(((timestamps >= pre_start) & (timestamps < block_start)).sum()) if pd.notna(block_start) else 0
        post_context_count = int(((timestamps >= block_end) & (timestamps < post_end)).sum()) if pd.notna(block_end) else 0
        boundary_start_truncated = bool(pd.notna(start_distance) and start_distance <= EPOCH_SECONDS)
        boundary_end_truncated = bool(np.isfinite(end_distance) and end_distance <= EPOCH_SECONDS)
        boundary_context_insufficient = bool(
            pre_context_count < CONTEXT_EPOCHS or post_context_count < CONTEXT_EPOCHS
        )
        rows.append(
            {
                "index": index,
                "recording_first_timestamp": first_timestamp.isoformat() if pd.notna(first_timestamp) else "",
                "recording_last_timestamp": last_timestamp.isoformat() if pd.notna(last_timestamp) else "",
                "pre_block_30min_epoch_count": pre_context_count,
                "post_block_30min_epoch_count": post_context_count,
                "boundary_start_truncated": boundary_start_truncated,
                "boundary_end_truncated": boundary_end_truncated,
                "boundary_context_insufficient": boundary_context_insufficient,
            }
        )
    context = pd.DataFrame(rows).set_index("index") if rows else pd.DataFrame()
    return selected.join(context, how="left")


def base_valid_mask(df: pd.DataFrame) -> pd.Series:
    offwrist_ok = numeric_series(df["offwrist_epochs"]).fillna(1).eq(0)
    return (
        ~bool_series(df["duration_under_3h"])
        & ~bool_series(df["duration_over_16h"])
        & ~bool_series(df["high_missing_wake"])
        & ~bool_series(df["non_contiguous_epochs"])
        & ~bool_series(df["missing_sleep_epochs"])
        & ~bool_series(df["invalid_sleep_efficiency"])
        & offwrist_ok
    )


def apply_valid_night_criteria(selected: pd.DataFrame) -> pd.DataFrame:
    selected = selected.copy()
    selected["valid_A"] = (
        base_valid_mask(selected)
        & ~bool_series(selected["partial_first_recording_day"])
        & ~bool_series(selected["partial_last_recording_day"])
    )
    selected["valid_B"] = base_valid_mask(selected)
    selected["valid_B2"] = (
        base_valid_mask(selected)
        & ~bool_series(selected["boundary_start_truncated"])
        & ~bool_series(selected["boundary_end_truncated"])
    )
    return selected


def criterion_summary(selected: pd.DataFrame, criterion_column: str, label: str) -> dict[str, Any]:
    valid = selected.loc[bool_series(selected[criterion_column])].copy()
    counts = valid.groupby("participant_key").size() if not valid.empty else pd.Series(dtype=int)
    return {
        "criterion": label,
        "valid_night_1plus_participants": int(counts.ge(1).sum()),
        "valid_night_2plus_participants": int(counts.ge(2).sum()),
        "valid_night_5plus_participants": int(counts.ge(5).sum()),
        "valid_night_6plus_participants": int(counts.ge(6).sum()),
        "valid_night_7plus_participants": int(counts.ge(7).sum()),
        "valid_night_count_mean": float(counts.mean()) if not counts.empty else 0.0,
        "valid_night_count_median": float(counts.median()) if not counts.empty else 0.0,
        "valid_night_count_min": int(counts.min()) if not counts.empty else 0,
        "valid_night_count_max": int(counts.max()) if not counts.empty else 0,
    }


def build_boundary_comparison(selected: pd.DataFrame) -> pd.DataFrame:
    rows = [
        criterion_summary(selected, "valid_A", "A_current_strict"),
        criterion_summary(selected, "valid_B", "B_no_partial_day_auto_exclusion"),
        criterion_summary(selected, "valid_B2", "B2_boundary_aware"),
    ]
    output = pd.DataFrame(rows)
    for column in [
        "boundary_start_truncated",
        "boundary_end_truncated",
        "boundary_context_insufficient",
    ]:
        flagged = selected.loc[bool_series(selected[column])]
        output[column + "_participant_count"] = int(participant_key(flagged).nunique()) if not flagged.empty else 0
        output[column + "_night_count"] = int(len(flagged))
    return output


def circular_stats(minutes: pd.Series) -> tuple[float, float]:
    values = numeric_series(minutes).dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    angles = values / 1440.0 * 2 * math.pi
    mean_sin = float(np.sin(angles).mean())
    mean_cos = float(np.cos(angles).mean())
    mean_angle = math.atan2(mean_sin, mean_cos)
    if mean_angle < 0:
        mean_angle += 2 * math.pi
    mean_minutes = mean_angle / (2 * math.pi) * 1440.0
    if len(values) < 2:
        return float(mean_minutes), np.nan
    r = min(math.hypot(mean_sin, mean_cos), 1.0)
    if r <= 0:
        return float(mean_minutes), np.nan
    sd_minutes = math.sqrt(max(-2 * math.log(r), 0.0)) / (2 * math.pi) * 1440.0
    return float(mean_minutes), float(sd_minutes)


def summarize_feature_window(feature_set: str, participant: str, rows: pd.DataFrame) -> dict[str, Any]:
    midpoint_mean, midpoint_sd = circular_stats(rows["sleep_midpoint_clock_minutes"])
    duration_mean = numeric_series(rows["sleep_duration_minutes"]).mean()
    duration_sd = numeric_series(rows["sleep_duration_minutes"]).std(ddof=1) if len(rows) >= 2 else np.nan
    return {
        "feature_set": feature_set,
        "mesaid": rows["mesaid"].iloc[0],
        "participant_key": participant,
        "valid_night_count": int(len(rows)),
        "window_start": rows["block_start_dt"].min().isoformat(),
        "window_end": rows["block_end_dt"].max().isoformat(),
        "first_daybynoon": rows["daybynoon"].iloc[0],
        "last_daybynoon": rows["daybynoon"].iloc[-1],
        "sleep_duration_mean": float(duration_mean),
        "sleep_duration_sd": float(duration_sd) if pd.notna(duration_sd) else np.nan,
        "sleep_duration_cv": float(duration_sd / duration_mean) if pd.notna(duration_sd) and duration_mean > 0 else np.nan,
        "sleep_efficiency_mean": float(numeric_series(rows["sleep_efficiency"]).mean()),
        "sleep_efficiency_sd": float(numeric_series(rows["sleep_efficiency"]).std(ddof=1)) if len(rows) >= 2 else np.nan,
        "waso_mean": float(numeric_series(rows["waso_minutes"]).mean()),
        "waso_sd": float(numeric_series(rows["waso_minutes"]).std(ddof=1)) if len(rows) >= 2 else np.nan,
        "sleep_midpoint_circular_mean": midpoint_mean,
        "sleep_midpoint_circular_sd": midpoint_sd,
    }


def consecutive_window(rows: pd.DataFrame, length: int = 7) -> pd.DataFrame:
    rows = rows.copy()
    rows["daybynoon_numeric"] = pd.to_numeric(rows["daybynoon"], errors="coerce")
    rows = rows.dropna(subset=["daybynoon_numeric"]).sort_values(["daybynoon_numeric", "block_start_dt"])
    by_day = rows.drop_duplicates(subset=["daybynoon_numeric"], keep="first")
    days = by_day["daybynoon_numeric"].astype(int).tolist()
    for start in days:
        wanted = list(range(start, start + length))
        if all(day in days for day in wanted):
            return by_day.loc[by_day["daybynoon_numeric"].astype(int).isin(wanted)].head(length)
    return pd.DataFrame(columns=rows.columns)


def build_sleep_features(selected: pd.DataFrame) -> pd.DataFrame:
    valid = selected.loc[bool_series(selected["valid_B2"])].copy()
    valid = valid.sort_values(["participant_key", "block_start_dt"])
    feature_rows: list[dict[str, Any]] = []
    for participant, group in valid.groupby("participant_key", dropna=False):
        group = group.sort_values("block_start_dt")
        if len(group) >= 7:
            feature_rows.append(summarize_feature_window("first_7_valid_nights", participant, group.head(7)))
        consecutive = consecutive_window(group, 7)
        if len(consecutive) == 7:
            feature_rows.append(summarize_feature_window("consecutive_7_daybynoon", participant, consecutive))
        if len(group) >= 1:
            feature_rows.append(summarize_feature_window("all_valid_nights", participant, group))
    return pd.DataFrame(feature_rows)


def build_window_summary(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_set, group in features.groupby("feature_set", dropna=False):
        rows.append(
            {
                "feature_set": feature_set,
                "participant_count": int(group["participant_key"].nunique()),
                "valid_night_count_mean": float(numeric_series(group["valid_night_count"]).mean()),
                "valid_night_count_median": float(numeric_series(group["valid_night_count"]).median()),
                "valid_night_count_min": int(numeric_series(group["valid_night_count"]).min()),
                "valid_night_count_max": int(numeric_series(group["valid_night_count"]).max()),
            }
        )
    return pd.DataFrame(rows)


def parse_time_to_minutes(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    parts = text.split(":")
    try:
        if len(parts) == 3:
            hour, minute, second = int(parts[0]), int(parts[1]), float(parts[2])
        elif len(parts) == 2:
            hour, minute, second = int(parts[0]), int(parts[1]), 0.0
        else:
            return np.nan
        return (hour % 24) * 60 + minute + second / 60.0
    except ValueError:
        return np.nan


def circular_difference_minutes(derived: pd.Series, official: pd.Series) -> pd.Series:
    diff = numeric_series(derived) - numeric_series(official)
    return ((diff + 720) % 1440) - 720


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    valid = pd.DataFrame({"x": numeric_series(x), "y": numeric_series(y)}).dropna()
    if len(valid) < 3 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return np.nan
    if method == "pearson":
        return float(stats.pearsonr(valid["x"], valid["y"]).statistic)
    if method == "spearman":
        return float(stats.spearmanr(valid["x"], valid["y"]).statistic)
    raise ValueError(method)


def dictionary_units(dictionary: pd.DataFrame, variable: str) -> str:
    match = dictionary.loc[dictionary["id"].eq(variable)]
    if match.empty:
        return ""
    return str(match.iloc[0].get("units", ""))


def validate_against_official(features: pd.DataFrame, dictionary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    phenotype_columns = ["mesaid", *OFFICIAL_VARIABLES]
    phenotype = load_main_phenotype(columns=phenotype_columns)
    phenotype["mesaid"] = phenotype["mesaid"].astype(str)
    features = features.copy()
    features["mesaid"] = features["mesaid"].astype(str)
    merged = features.merge(phenotype, on="mesaid", how="left", validate="many_to_one")
    merged["official_sleep_midpoint_minutes"] = merged["avgsleepmidpoint5"].apply(parse_time_to_minutes)

    comparisons = [
        ("sleep_duration_mean", "avgmainsleep5", "derived sleep_duration_mean vs avgmainsleep5", False),
        ("sleep_efficiency_mean", "avgefficiency5", "derived sleep_efficiency_mean vs avgefficiency5", False),
        ("waso_mean", "avgwaso5", "derived waso_mean vs avgwaso5", False),
        (
            "sleep_midpoint_circular_mean",
            "official_sleep_midpoint_minutes",
            "derived sleep_midpoint_circular_mean vs avgsleepmidpoint5",
            True,
        ),
    ]

    rows: list[dict[str, Any]] = []
    for feature_set, group in merged.groupby("feature_set", dropna=False):
        for derived_col, official_col, label, circular in comparisons:
            official_variable = "avgsleepmidpoint5" if circular else official_col
            paired = group[[derived_col, official_col]].copy()
            paired[derived_col] = numeric_series(paired[derived_col])
            paired[official_col] = numeric_series(paired[official_col])
            paired = paired.dropna()
            if circular:
                diff = circular_difference_minutes(paired[derived_col], paired[official_col])
                mean_derived, _ = circular_stats(paired[derived_col])
                mean_official, _ = circular_stats(paired[official_col])
            else:
                diff = paired[derived_col] - paired[official_col]
                mean_derived = float(paired[derived_col].mean()) if not paired.empty else np.nan
                mean_official = float(paired[official_col].mean()) if not paired.empty else np.nan
            rows.append(
                {
                    "feature_set": feature_set,
                    "comparison": label,
                    "official_variable": official_variable,
                    "official_units": dictionary_units(dictionary, official_variable),
                    "derived_units": "clock minutes after midnight" if circular else dictionary_units(dictionary, official_variable),
                    "unit_conversion": "Official hh:mm:ss converted to clock minutes; circular difference used."
                    if circular
                    else "No conversion required.",
                    "matched_sample_count": int(len(paired)),
                    "pearson_correlation": safe_corr(paired[derived_col], paired[official_col], "pearson"),
                    "spearman_correlation": safe_corr(paired[derived_col], paired[official_col], "spearman"),
                    "mean_derived_value": mean_derived,
                    "mean_official_value": mean_official,
                    "mean_difference": float(diff.mean()) if len(diff) else np.nan,
                    "median_absolute_error": float(diff.abs().median()) if len(diff) else np.nan,
                    "mean_absolute_error": float(diff.abs().mean()) if len(diff) else np.nan,
                    "rmse": float(np.sqrt(np.mean(np.square(diff)))) if len(diff) else np.nan,
                }
            )
    return pd.DataFrame(rows), merged


def main() -> None:
    setup_logging()
    logging.info("Project root: %s", PROJECT_ROOT)
    backup_existing_outputs()

    raw_inventory = build_raw_path_inventory()
    write_csv(raw_inventory, RAW_PATH_INVENTORY_PATH)

    id_review, id_quality = build_id_mapping_review()
    write_csv(id_review, ID_MAPPING_REVIEW_PATH)
    write_csv(id_quality, ID_MAPPING_QUALITY_PATH)

    dictionary = load_dictionary()
    main_df = load_main_phenotype()
    native_candidates = build_native_covariate_candidates(dictionary, main_df)
    write_csv(native_candidates, NATIVE_COVARIATE_PATH)

    blocks = read_csv_with_fallback(REST_S_BLOCKS_PATH)
    selected = add_boundary_context(blocks)
    selected = apply_valid_night_criteria(selected)
    boundary_comparison = build_boundary_comparison(selected)
    write_csv(boundary_comparison, BOUNDARY_COMPARISON_PATH)

    features = build_sleep_features(selected)
    write_csv(features, SLEEP_FEATURES_PATH)
    seven_summary = build_window_summary(features)
    write_csv(seven_summary, SEVEN_NIGHT_SUMMARY_PATH)

    metric_validation, validation_dataset = validate_against_official(features, dictionary)
    write_csv(metric_validation, METRIC_VALIDATION_PATH)
    write_csv(validation_dataset, VALIDATION_DATASET_PATH)

    print("ID and sleep-feature validation complete")
    print(f"ID mapping official crosswalk found: {id_quality.loc[id_quality['metric'].eq('official_crosswalk_found'), 'value'].iloc[0]}")
    print(f"Feature rows: {len(features)}")
    print(f"Metric validation rows: {len(metric_validation)}")


if __name__ == "__main__":
    main()
