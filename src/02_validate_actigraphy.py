"""Validate feasibility of deriving sleep variability from actigraphy files.

REST-S is treated only as a provisional main-sleep candidate until the official
MESA/NSRR documentation is checked. Raw CSV files are never modified.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ACTIGRAPHY_DIR = PROJECT_ROOT / "data" / "raw" / "actigraphy"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

REST_S_BLOCKS_PATH = INTERIM_DIR / "rest_s_blocks.csv"
DAILY_METRICS_PATH = INTERIM_DIR / "provisional_daily_sleep_metrics.csv"
PARTICIPANT_VARIABILITY_PATH = INTERIM_DIR / "participant_sleep_variability.csv"
FEASIBILITY_SUMMARY_PATH = OUTPUT_TABLE_DIR / "actigraphy_feasibility_summary.csv"
PARTICIPANT_QUALITY_PATH = OUTPUT_TABLE_DIR / "participant_quality_summary.csv"
VALID_NIGHT_DISTRIBUTION_PATH = OUTPUT_TABLE_DIR / "valid_night_distribution.csv"
LOG_PATH = OUTPUT_LOG_DIR / "02_validate_actigraphy.log"

ACTIGRAPHY_PATTERN = "mesa-sleep-*.csv"
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1")
REST_S_LABEL = "REST-S"
EPOCH_SECONDS = 30
EPOCH_MINUTES = EPOCH_SECONDS / 60
MIN_REST_INTERVAL_MINUTES = 180
MAX_REST_INTERVAL_MINUTES = 960
HIGH_MISSING_WAKE_THRESHOLD = 0.10

REQUIRED_ACTIGRAPHY_COLUMNS = [
    "mesaid",
    "linetime",
    "wake",
    "interval",
    "offwrist",
    "daybynoon",
    "date_shifted",
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

BLOCK_COLUMNS = [
    "source_file",
    "filename_participant_id",
    "mesaid",
    "daybynoon",
    "block_id",
    "block_start",
    "block_end",
    "total_epochs",
    "duration_minutes",
    "sleep_epochs",
    "wake_epochs",
    "missing_wake_epochs",
    "offwrist_epochs",
    "valid_scored_epochs",
    "wake_missing_ratio",
    "sleep_duration_minutes",
    "rest_interval_duration_minutes",
    "sleep_efficiency",
    "waso_minutes",
    "sleep_midpoint_datetime",
    "sleep_midpoint_clock_minutes",
    "sleep_midpoint_clock_minutes_is_circular",
    "selected_as_provisional_main_sleep",
    "main_sleep_selection_reason",
    *QUALITY_FLAGS,
    "provisional_valid_night",
]

DAILY_COLUMNS = [
    "source_file",
    "filename_participant_id",
    "mesaid",
    "daybynoon",
    "block_id",
    "block_start",
    "block_end",
    "sleep_duration_minutes",
    "rest_interval_duration_minutes",
    "sleep_efficiency",
    "waso_minutes",
    "sleep_midpoint_datetime",
    "sleep_midpoint_clock_minutes",
    "sleep_midpoint_clock_minutes_is_circular",
    "main_sleep_selection_reason",
    *QUALITY_FLAGS,
    "provisional_valid_night",
]

PARTICIPANT_QUALITY_COLUMNS = [
    "source_file",
    "filename_participant_id",
    "mesaid",
    "filename_id_matches_mesaid",
    "read_success",
    "parse_failed",
    "error_message",
    "missing_required_columns",
    "row_count",
    "unique_mesaid_count",
    "single_mesaid_in_file",
    "timestamp_parse_failed_count",
    "data_already_time_sorted",
    "duplicate_timestamp_count",
    "missing_timestamp_count",
    "gap_over_30s_count",
    "unexpected_epoch_interval_count",
    "epoch_interval_30s_ok",
    "interval_values",
    "wake_missing_count",
    "wake_invalid_count",
    "offwrist_missing_count",
    "offwrist_invalid_count",
    "has_rest_s",
    "rest_s_block_count",
    "provisional_valid_night_count",
    *[f"{flag}_count" for flag in QUALITY_FLAGS],
]

PARTICIPANT_VARIABILITY_COLUMNS = [
    "source_file",
    "filename_participant_id",
    "mesaid",
    "valid_night_count",
    "sleep_duration_mean",
    "sleep_duration_sd",
    "sleep_duration_cv",
    "sleep_efficiency_mean",
    "sleep_efficiency_sd",
    "waso_mean",
    "waso_sd",
    "sleep_midpoint_mean",
    "sleep_midpoint_sd",
    "sleep_midpoint_method",
    "can_sleep_duration_sd",
    "can_sleep_duration_cv",
    "can_sleep_efficiency_sd",
    "can_waso_sd",
    "can_sleep_midpoint_sd",
]


def ensure_output_dirs() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    ensure_output_dirs()
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


def json_string(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


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


def list_actigraphy_files() -> list[Path]:
    if not RAW_ACTIGRAPHY_DIR.exists():
        raise FileNotFoundError(f"Actigraphy directory does not exist: {RAW_ACTIGRAPHY_DIR}")
    files = sorted(RAW_ACTIGRAPHY_DIR.glob(ACTIGRAPHY_PATTERN))
    if not files:
        raise FileNotFoundError(
            f"No actigraphy files matched {RAW_ACTIGRAPHY_DIR / ACTIGRAPHY_PATTERN}"
        )
    return files


def extract_filename_participant_id(path: Path) -> str | None:
    match = re.match(r"^mesa-sleep-(\d+)\.csv$", path.name, flags=re.IGNORECASE)
    return match.group(1) if match else None


def timestamp_to_iso(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def parse_actigraphy_datetime(df: pd.DataFrame) -> pd.Series:
    date_part = pd.to_datetime(df["date_shifted"], errors="coerce")
    date_text = date_part.dt.strftime("%Y-%m-%d").astype("string")
    time_text = df["linetime"].astype("string").str.strip()
    time_match = time_text.str.extract(r"(\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)")[0]
    combined = date_text.fillna("") + " " + time_match.astype("string").fillna("")
    combined = combined.str.strip()

    parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for date_format in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ):
        missing = parsed.isna() & combined.ne("")
        if not missing.any():
            break
        parsed.loc[missing] = pd.to_datetime(
            combined.loc[missing],
            format=date_format,
            errors="coerce",
        )
    return parsed


def normalize_binary(series: pd.Series) -> tuple[pd.Series, int, int]:
    text = series.astype("string").str.strip()
    missing_mask = series.isna() | text.isna() | text.eq("")
    numeric = pd.to_numeric(text, errors="coerce")
    valid_mask = numeric.isin([0, 1])
    invalid_count = int((~missing_mask & ~valid_mask).sum())
    missing_count = int(missing_mask.sum())
    return numeric.where(valid_mask), missing_count, invalid_count


def clock_minutes(value: pd.Timestamp | None) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    timestamp = pd.Timestamp(value)
    return (
        timestamp.hour * 60
        + timestamp.minute
        + timestamp.second / 60
        + timestamp.microsecond / 60_000_000
    )


def ids_match(filename_id: str | None, mesaid: str | None) -> bool | None:
    if filename_id is None or mesaid is None:
        return None
    try:
        return int(filename_id) == int(float(mesaid))
    except ValueError:
        return filename_id.lstrip("0") == mesaid.lstrip("0")


def count_timestamp_gaps(valid_timestamps: pd.Series) -> dict[str, int | bool]:
    unique_sorted = valid_timestamps.drop_duplicates().sort_values()
    if len(unique_sorted) <= 1:
        return {
            "missing_timestamp_count": 0,
            "gap_over_30s_count": 0,
            "unexpected_epoch_interval_count": 0,
            "epoch_interval_30s_ok": True,
        }

    diffs = unique_sorted.diff().dropna().dt.total_seconds()
    gap_over_30 = diffs.gt(EPOCH_SECONDS)
    unexpected = diffs.ne(EPOCH_SECONDS)
    missing_timestamp_count = 0
    for diff_seconds in diffs[gap_over_30]:
        missing_timestamp_count += max(int(round(float(diff_seconds) / EPOCH_SECONDS)) - 1, 0)

    return {
        "missing_timestamp_count": int(missing_timestamp_count),
        "gap_over_30s_count": int(gap_over_30.sum()),
        "unexpected_epoch_interval_count": int(unexpected.sum()),
        "epoch_interval_30s_ok": bool(not unexpected.any()),
    }


def prepare_dataframe(path: Path) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    columns = list(read_csv_with_fallback(path, nrows=0).columns.astype(str))
    missing_required = [name for name in REQUIRED_ACTIGRAPHY_COLUMNS if name not in columns]
    if missing_required:
        return pd.DataFrame(), missing_required, {}

    df = read_csv_with_fallback(path, usecols=REQUIRED_ACTIGRAPHY_COLUMNS)
    df["row_number"] = np.arange(len(df), dtype=int)
    df["timestamp"] = parse_actigraphy_datetime(df)
    df["interval_clean"] = df["interval"].astype("string").str.strip().str.upper()
    df["daybynoon_key"] = df["daybynoon"].astype("string").str.strip().fillna("")
    df["wake_numeric"], wake_missing, wake_invalid = normalize_binary(df["wake"])
    df["offwrist_numeric"], offwrist_missing, offwrist_invalid = normalize_binary(
        df["offwrist"]
    )

    valid_timestamps = df["timestamp"].dropna()
    duplicate_timestamp_count = int(valid_timestamps.duplicated(keep=False).sum())
    data_already_sorted = bool(valid_timestamps.is_monotonic_increasing)
    gap_stats = count_timestamp_gaps(valid_timestamps)
    gap_stats["epoch_interval_30s_ok"] = bool(
        gap_stats["epoch_interval_30s_ok"] and duplicate_timestamp_count == 0
    )

    quality = {
        "row_count": int(len(df)),
        "timestamp_parse_failed_count": int(df["timestamp"].isna().sum()),
        "data_already_time_sorted": data_already_sorted,
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "wake_missing_count": wake_missing,
        "wake_invalid_count": wake_invalid,
        "offwrist_missing_count": offwrist_missing,
        "offwrist_invalid_count": offwrist_invalid,
        "interval_values": json_string(normalize_counts(df["interval_clean"])),
        **gap_stats,
    }
    return df, [], quality


def normalize_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False)
    output: dict[str, int] = {}
    for value, count in counts.items():
        label = "<NA>" if pd.isna(value) else str(value)
        output[label] = int(count)
    return dict(sorted(output.items(), key=lambda item: item[0]))


def recording_boundary_days(df: pd.DataFrame) -> tuple[str | None, str | None]:
    ordered = df.loc[df["timestamp"].notna()].sort_values(["timestamp", "row_number"])
    days = ordered["daybynoon_key"].replace("", pd.NA).dropna()
    if days.empty:
        return None, None
    return str(days.iloc[0]), str(days.iloc[-1])


def block_daybynoon(block: pd.DataFrame) -> str | None:
    days = block["daybynoon_key"].replace("", pd.NA).dropna()
    if days.empty:
        return None
    return str(days.iloc[0])


def block_non_contiguous(block: pd.DataFrame) -> bool:
    timestamps = block["timestamp"].sort_values()
    if len(timestamps) <= 1:
        return False
    diffs = timestamps.diff().dropna().dt.total_seconds()
    return bool(timestamps.duplicated().any() or diffs.ne(EPOCH_SECONDS).any())


def calculate_waso_minutes(block: pd.DataFrame) -> float:
    sleep_positions = np.flatnonzero(
        block["wake_numeric"].eq(0).fillna(False).to_numpy(dtype=bool)
    )
    if len(sleep_positions) == 0:
        return float("nan")
    first_sleep = int(sleep_positions[0])
    last_sleep = int(sleep_positions[-1])
    sleep_window = block.iloc[first_sleep : last_sleep + 1]
    return float(sleep_window["wake_numeric"].eq(1).sum() * EPOCH_MINUTES)


def summarize_block(
    block: pd.DataFrame,
    block_index: int,
    source_file: str,
    filename_participant_id: str | None,
    mesaid: str | None,
) -> dict[str, Any]:
    block = block.sort_values(["timestamp", "row_number"]).reset_index(drop=True)
    total_epochs = int(len(block))
    block_start = block["timestamp"].iloc[0]
    block_end = block["timestamp"].iloc[-1] + pd.Timedelta(seconds=EPOCH_SECONDS)
    midpoint = block_start + (block_end - block_start) / 2
    wake_numeric = block["wake_numeric"]
    offwrist_numeric = block["offwrist_numeric"]

    sleep_epochs = int(wake_numeric.eq(0).sum())
    wake_epochs = int(wake_numeric.eq(1).sum())
    missing_wake_epochs = int(wake_numeric.isna().sum())
    offwrist_epochs = int(offwrist_numeric.eq(1).sum())
    valid_scored_epochs = int(wake_numeric.isin([0, 1]).sum())
    sleep_duration_minutes = sleep_epochs * EPOCH_MINUTES
    rest_interval_duration_minutes = total_epochs * EPOCH_MINUTES
    sleep_efficiency = (
        sleep_epochs / valid_scored_epochs * 100 if valid_scored_epochs > 0 else float("nan")
    )
    wake_missing_ratio = (
        missing_wake_epochs / total_epochs if total_epochs > 0 else float("nan")
    )
    non_contiguous = block_non_contiguous(block)
    block_prefix = filename_participant_id or mesaid or "unknown"

    return {
        "source_file": source_file,
        "filename_participant_id": filename_participant_id,
        "mesaid": mesaid,
        "daybynoon": block_daybynoon(block),
        "block_id": f"{block_prefix}_RESTS_{block_index:04d}",
        "block_start": timestamp_to_iso(block_start),
        "block_end": timestamp_to_iso(block_end),
        "total_epochs": total_epochs,
        "duration_minutes": rest_interval_duration_minutes,
        "sleep_epochs": sleep_epochs,
        "wake_epochs": wake_epochs,
        "missing_wake_epochs": missing_wake_epochs,
        "offwrist_epochs": offwrist_epochs,
        "valid_scored_epochs": valid_scored_epochs,
        "wake_missing_ratio": wake_missing_ratio,
        "sleep_duration_minutes": sleep_duration_minutes,
        "rest_interval_duration_minutes": rest_interval_duration_minutes,
        "sleep_efficiency": sleep_efficiency,
        "waso_minutes": calculate_waso_minutes(block),
        "sleep_midpoint_datetime": timestamp_to_iso(midpoint),
        "sleep_midpoint_clock_minutes": clock_minutes(midpoint),
        "sleep_midpoint_clock_minutes_is_circular": True,
        "selected_as_provisional_main_sleep": False,
        "main_sleep_selection_reason": "",
        "duration_under_3h": rest_interval_duration_minutes < MIN_REST_INTERVAL_MINUTES,
        "duration_over_16h": rest_interval_duration_minutes > MAX_REST_INTERVAL_MINUTES,
        "high_missing_wake": wake_missing_ratio > HIGH_MISSING_WAKE_THRESHOLD,
        "contains_offwrist": offwrist_epochs > 0,
        "non_contiguous_epochs": non_contiguous,
        "partial_first_recording_day": False,
        "partial_last_recording_day": False,
        "multiple_blocks_same_day": False,
        "invalid_sleep_efficiency": not (
            pd.notna(sleep_efficiency) and 0 < sleep_efficiency <= 100
        ),
        "missing_sleep_epochs": sleep_epochs == 0,
        "provisional_valid_night": False,
    }


def identify_rest_s_blocks(
    df: pd.DataFrame,
    source_file: str,
    filename_participant_id: str | None,
    mesaid: str | None,
) -> pd.DataFrame:
    rest_s = df.loc[df["interval_clean"].eq(REST_S_LABEL) & df["timestamp"].notna()].copy()
    if rest_s.empty:
        return pd.DataFrame(columns=BLOCK_COLUMNS)

    rest_s = rest_s.sort_values(["timestamp", "row_number"]).reset_index(drop=True)
    timestamp_diff = rest_s["timestamp"].diff().dt.total_seconds()
    day_changed = rest_s["daybynoon_key"].ne(rest_s["daybynoon_key"].shift())
    new_block = timestamp_diff.isna() | timestamp_diff.gt(EPOCH_SECONDS) | day_changed
    rest_s["rest_s_block_number"] = new_block.cumsum().astype(int)

    rows: list[dict[str, Any]] = []
    for block_index, (_, block) in enumerate(rest_s.groupby("rest_s_block_number"), start=1):
        rows.append(
            summarize_block(
                block=block,
                block_index=block_index,
                source_file=source_file,
                filename_participant_id=filename_participant_id,
                mesaid=mesaid,
            )
        )
    return pd.DataFrame(rows, columns=BLOCK_COLUMNS)


def apply_block_flags_and_selection(
    blocks: pd.DataFrame,
    first_day: str | None,
    last_day: str | None,
) -> pd.DataFrame:
    if blocks.empty:
        return pd.DataFrame(columns=BLOCK_COLUMNS)

    blocks = blocks.copy()
    blocks["partial_first_recording_day"] = blocks["daybynoon"].eq(first_day)
    blocks["partial_last_recording_day"] = blocks["daybynoon"].eq(last_day)

    same_day_counts = blocks.groupby(["source_file", "daybynoon"], dropna=False)[
        "block_id"
    ].transform("count")
    blocks["multiple_blocks_same_day"] = same_day_counts.gt(1)

    ranked = blocks.sort_values(
        ["source_file", "daybynoon", "rest_interval_duration_minutes", "block_start"],
        ascending=[True, True, False, True],
        na_position="last",
    )
    selected_indices = (
        ranked.groupby(["source_file", "daybynoon"], dropna=False).head(1).index
    )
    blocks.loc[selected_indices, "selected_as_provisional_main_sleep"] = True
    blocks.loc[selected_indices, "main_sleep_selection_reason"] = (
        "longest REST-S block within source_file and daybynoon"
    )

    valid = blocks["selected_as_provisional_main_sleep"].astype(bool)
    for flag in QUALITY_FLAGS:
        if flag in {"duration_under_3h", "duration_over_16h"}:
            valid &= ~blocks[flag].astype(bool)
        elif flag in {
            "high_missing_wake",
            "contains_offwrist",
            "non_contiguous_epochs",
            "partial_first_recording_day",
            "partial_last_recording_day",
            "invalid_sleep_efficiency",
            "missing_sleep_epochs",
        }:
            valid &= ~blocks[flag].astype(bool)
    blocks["provisional_valid_night"] = valid
    return blocks.reindex(columns=BLOCK_COLUMNS)


def create_daily_metrics(blocks: pd.DataFrame) -> pd.DataFrame:
    if blocks.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    daily = blocks.loc[blocks["selected_as_provisional_main_sleep"].astype(bool)].copy()
    return daily.reindex(columns=DAILY_COLUMNS)


def summarize_failed_participant(
    path: Path,
    error_message: str,
    missing_required_columns: list[str] | None = None,
) -> dict[str, Any]:
    row = {column: None for column in PARTICIPANT_QUALITY_COLUMNS}
    row.update(
        {
            "source_file": relative_path(path),
            "filename_participant_id": extract_filename_participant_id(path),
            "read_success": False,
            "parse_failed": True,
            "error_message": error_message,
            "missing_required_columns": json_string(missing_required_columns or []),
            "row_count": 0,
            "has_rest_s": False,
            "rest_s_block_count": 0,
            "provisional_valid_night_count": 0,
        }
    )
    for flag in QUALITY_FLAGS:
        row[f"{flag}_count"] = 0
    return row


def summarize_successful_participant(
    path: Path,
    df: pd.DataFrame,
    blocks: pd.DataFrame,
    quality: dict[str, Any],
) -> dict[str, Any]:
    mesaid_values = sorted(df["mesaid"].dropna().astype(str).unique().tolist())
    mesaid = mesaid_values[0] if len(mesaid_values) == 1 else None
    source_file = relative_path(path)

    row = {column: None for column in PARTICIPANT_QUALITY_COLUMNS}
    row.update(
        {
            "source_file": source_file,
            "filename_participant_id": extract_filename_participant_id(path),
            "mesaid": mesaid,
            "filename_id_matches_mesaid": ids_match(extract_filename_participant_id(path), mesaid),
            "read_success": True,
            "parse_failed": False,
            "error_message": "",
            "missing_required_columns": json_string([]),
            "unique_mesaid_count": len(mesaid_values),
            "single_mesaid_in_file": len(mesaid_values) == 1,
            "has_rest_s": bool(not blocks.empty),
            "rest_s_block_count": int(len(blocks)),
            "provisional_valid_night_count": int(
                blocks["provisional_valid_night"].fillna(False).astype(bool).sum()
            )
            if not blocks.empty
            else 0,
            **quality,
        }
    )
    for flag in QUALITY_FLAGS:
        row[f"{flag}_count"] = (
            int(blocks[flag].fillna(False).astype(bool).sum()) if not blocks.empty else 0
        )
    return row


def process_actigraphy_file(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    logging.info("Processing %s", relative_path(path))
    try:
        df, missing_required, quality = prepare_dataframe(path)
        if missing_required:
            message = f"Missing required columns: {missing_required}"
            logging.warning("%s: %s", relative_path(path), message)
            return (
                pd.DataFrame(columns=BLOCK_COLUMNS),
                pd.DataFrame(columns=DAILY_COLUMNS),
                summarize_failed_participant(path, message, missing_required),
            )

        mesaid_values = sorted(df["mesaid"].dropna().astype(str).unique().tolist())
        mesaid = mesaid_values[0] if len(mesaid_values) == 1 else None
        first_day, last_day = recording_boundary_days(df)
        df = df.sort_values(["timestamp", "row_number"], na_position="last").reset_index(
            drop=True
        )
        blocks = identify_rest_s_blocks(
            df=df,
            source_file=relative_path(path),
            filename_participant_id=extract_filename_participant_id(path),
            mesaid=mesaid,
        )
        blocks = apply_block_flags_and_selection(blocks, first_day, last_day)
        daily = create_daily_metrics(blocks)
        participant = summarize_successful_participant(path, df, blocks, quality)
        return blocks, daily, participant
    except (OSError, EmptyDataError, ParserError, UnicodeDecodeError, ValueError) as exc:
        logging.exception("Failed to process %s", relative_path(path))
        return (
            pd.DataFrame(columns=BLOCK_COLUMNS),
            pd.DataFrame(columns=DAILY_COLUMNS),
            summarize_failed_participant(path, f"{type(exc).__name__}: {exc}"),
        )


def sample_sd(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return float("nan")
    return float(values.std(ddof=1))


def mean_value(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def circular_midpoint_stats(minutes: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(minutes, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")

    angles = values / 1440 * 2 * math.pi
    mean_sin = float(np.sin(angles).mean())
    mean_cos = float(np.cos(angles).mean())
    mean_angle = math.atan2(mean_sin, mean_cos)
    if mean_angle < 0:
        mean_angle += 2 * math.pi
    mean_minutes = mean_angle / (2 * math.pi) * 1440

    if len(values) < 2:
        return float(mean_minutes), float("nan")
    r = math.hypot(mean_sin, mean_cos)
    if r <= 0:
        return float(mean_minutes), float("nan")
    circular_sd_radians = math.sqrt(max(-2 * math.log(min(r, 1.0)), 0.0))
    circular_sd_minutes = circular_sd_radians / (2 * math.pi) * 1440
    return float(mean_minutes), float(circular_sd_minutes)


def create_participant_variability(
    participant_quality: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    logging.info(
        "Sleep midpoint variability uses circular statistics on 24-hour clock minutes."
    )

    for _, participant in participant_quality.iterrows():
        source_file = participant["source_file"]
        participant_daily = daily.loc[
            daily["source_file"].eq(source_file)
            & daily["provisional_valid_night"].fillna(False).astype(bool)
        ].copy()

        valid_night_count = int(len(participant_daily))
        duration_mean = mean_value(participant_daily["sleep_duration_minutes"])
        duration_sd = sample_sd(participant_daily["sleep_duration_minutes"])
        duration_cv = (
            duration_sd / duration_mean
            if pd.notna(duration_mean) and duration_mean > 0 and pd.notna(duration_sd)
            else float("nan")
        )
        efficiency_mean = mean_value(participant_daily["sleep_efficiency"])
        efficiency_sd = sample_sd(participant_daily["sleep_efficiency"])
        waso_mean = mean_value(participant_daily["waso_minutes"])
        waso_sd = sample_sd(participant_daily["waso_minutes"])
        midpoint_mean, midpoint_sd = circular_midpoint_stats(
            participant_daily["sleep_midpoint_clock_minutes"]
        )

        rows.append(
            {
                "source_file": source_file,
                "filename_participant_id": participant["filename_participant_id"],
                "mesaid": participant["mesaid"],
                "valid_night_count": valid_night_count,
                "sleep_duration_mean": duration_mean,
                "sleep_duration_sd": duration_sd,
                "sleep_duration_cv": duration_cv,
                "sleep_efficiency_mean": efficiency_mean,
                "sleep_efficiency_sd": efficiency_sd,
                "waso_mean": waso_mean,
                "waso_sd": waso_sd,
                "sleep_midpoint_mean": midpoint_mean,
                "sleep_midpoint_sd": midpoint_sd,
                "sleep_midpoint_method": "circular mean and circular SD on 24-hour clock minutes",
                "can_sleep_duration_sd": pd.notna(duration_sd),
                "can_sleep_duration_cv": pd.notna(duration_cv),
                "can_sleep_efficiency_sd": pd.notna(efficiency_sd),
                "can_waso_sd": pd.notna(waso_sd),
                "can_sleep_midpoint_sd": pd.notna(midpoint_sd),
            }
        )

    return pd.DataFrame(rows).reindex(columns=PARTICIPANT_VARIABILITY_COLUMNS)


def participant_count(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df:
        return 0
    return int(df[column].fillna(False).astype(bool).sum())


def participant_problem_count(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df:
        return 0
    return int(pd.to_numeric(df[column], errors="coerce").fillna(0).gt(0).sum())


def create_valid_night_distribution(variability: pd.DataFrame) -> pd.DataFrame:
    if variability.empty:
        return pd.DataFrame(
            columns=["valid_night_count", "participant_count", "participant_percent"]
        )
    counts = (
        variability["valid_night_count"].fillna(0).astype(int).value_counts().sort_index()
    )
    total = int(counts.sum())
    distribution = counts.rename_axis("valid_night_count").reset_index(name="participant_count")
    distribution["participant_percent"] = (
        distribution["participant_count"] / total * 100 if total > 0 else np.nan
    )
    return distribution


def create_feasibility_summary(
    actigraphy_files: list[Path],
    participant_quality: pd.DataFrame,
    variability: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "metric": "total_actigraphy_file_count",
            "value": len(actigraphy_files),
            "notes": f"Pattern: {ACTIGRAPHY_PATTERN}",
        },
        {
            "metric": "successfully_parsed_file_count",
            "value": participant_count(participant_quality, "read_success"),
            "notes": "",
        },
        {
            "metric": "parse_failed_file_count",
            "value": participant_count(participant_quality, "parse_failed"),
            "notes": "",
        },
        {
            "metric": "participants_with_rest_s",
            "value": participant_count(participant_quality, "has_rest_s"),
            "notes": "REST-S remains provisional.",
        },
        {
            "metric": "participants_with_1plus_valid_nights",
            "value": int(variability["valid_night_count"].fillna(0).astype(int).ge(1).sum())
            if not variability.empty
            else 0,
            "notes": "",
        },
        {
            "metric": "participants_with_5plus_valid_nights",
            "value": int(variability["valid_night_count"].fillna(0).astype(int).ge(5).sum())
            if not variability.empty
            else 0,
            "notes": "",
        },
        {
            "metric": "participants_with_7plus_valid_nights",
            "value": int(variability["valid_night_count"].fillna(0).astype(int).ge(7).sum())
            if not variability.empty
            else 0,
            "notes": "",
        },
        {
            "metric": "sleep_duration_sd_available_participant_count",
            "value": participant_count(variability, "can_sleep_duration_sd"),
            "notes": "Sample SD, ddof=1.",
        },
        {
            "metric": "sleep_duration_cv_available_participant_count",
            "value": participant_count(variability, "can_sleep_duration_cv"),
            "notes": "SD / mean when mean > 0.",
        },
        {
            "metric": "sleep_efficiency_sd_available_participant_count",
            "value": participant_count(variability, "can_sleep_efficiency_sd"),
            "notes": "Sample SD, ddof=1.",
        },
        {
            "metric": "waso_sd_available_participant_count",
            "value": participant_count(variability, "can_waso_sd"),
            "notes": "Sample SD, ddof=1.",
        },
        {
            "metric": "sleep_midpoint_sd_available_participant_count",
            "value": participant_count(variability, "can_sleep_midpoint_sd"),
            "notes": "Circular SD on 24-hour clock minutes.",
        },
        {
            "metric": "rest_s_interpretation_status",
            "value": "provisional",
            "notes": "No phenotype merge or regression is performed.",
        },
        {
            "metric": "main_sleep_selection_rule",
            "value": "longest REST-S block per daybynoon",
            "notes": "All REST-S blocks are retained in rest_s_blocks.csv.",
        },
    ]

    for flag in QUALITY_FLAGS:
        rows.append(
            {
                "metric": f"participants_with_{flag}",
                "value": participant_problem_count(participant_quality, f"{flag}_count"),
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    blocks: pd.DataFrame,
    daily: pd.DataFrame,
    participant_quality: pd.DataFrame,
    variability: pd.DataFrame,
    feasibility: pd.DataFrame,
    valid_night_distribution: pd.DataFrame,
) -> None:
    ensure_output_dirs()
    blocks.reindex(columns=BLOCK_COLUMNS).to_csv(
        REST_S_BLOCKS_PATH, index=False, encoding="utf-8-sig"
    )
    daily.reindex(columns=DAILY_COLUMNS).to_csv(
        DAILY_METRICS_PATH, index=False, encoding="utf-8-sig"
    )
    variability.reindex(columns=PARTICIPANT_VARIABILITY_COLUMNS).to_csv(
        PARTICIPANT_VARIABILITY_PATH, index=False, encoding="utf-8-sig"
    )
    feasibility.to_csv(FEASIBILITY_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    participant_quality.reindex(columns=PARTICIPANT_QUALITY_COLUMNS).to_csv(
        PARTICIPANT_QUALITY_PATH, index=False, encoding="utf-8-sig"
    )
    valid_night_distribution.to_csv(
        VALID_NIGHT_DISTRIBUTION_PATH, index=False, encoding="utf-8-sig"
    )


def print_run_summary(feasibility: pd.DataFrame) -> None:
    metrics = dict(zip(feasibility["metric"], feasibility["value"]))
    print("Actigraphy validation complete")
    print(f"Actigraphy CSV files: {metrics.get('total_actigraphy_file_count')}")
    print(f"Successfully parsed files: {metrics.get('successfully_parsed_file_count')}")
    print(f"Parse failed files: {metrics.get('parse_failed_file_count')}")
    print(f"Participants with >=5 valid nights: {metrics.get('participants_with_5plus_valid_nights')}")
    print(f"Participants with >=7 valid nights: {metrics.get('participants_with_7plus_valid_nights')}")
    print(f"Outputs written under: {relative_path(INTERIM_DIR)} and {relative_path(OUTPUT_TABLE_DIR)}")


def main() -> None:
    setup_logging()
    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Current working directory: %s", Path.cwd())
    logging.info(
        "REST-S is provisional; all REST-S blocks are retained before selecting longest daily candidates."
    )

    actigraphy_files = list_actigraphy_files()
    all_blocks: list[pd.DataFrame] = []
    all_daily: list[pd.DataFrame] = []
    participant_rows: list[dict[str, Any]] = []

    for path in actigraphy_files:
        blocks, daily, participant = process_actigraphy_file(path)
        all_blocks.append(blocks)
        all_daily.append(daily)
        participant_rows.append(participant)

    blocks_df = (
        pd.concat(all_blocks, ignore_index=True)
        if all_blocks
        else pd.DataFrame(columns=BLOCK_COLUMNS)
    )
    daily_df = (
        pd.concat(all_daily, ignore_index=True)
        if all_daily
        else pd.DataFrame(columns=DAILY_COLUMNS)
    )
    participant_quality = pd.DataFrame(participant_rows).reindex(
        columns=PARTICIPANT_QUALITY_COLUMNS
    )
    variability = create_participant_variability(participant_quality, daily_df)
    valid_night_distribution = create_valid_night_distribution(variability)
    feasibility = create_feasibility_summary(
        actigraphy_files=actigraphy_files,
        participant_quality=participant_quality,
        variability=variability,
    )

    write_outputs(
        blocks=blocks_df,
        daily=daily_df,
        participant_quality=participant_quality,
        variability=variability,
        feasibility=feasibility,
        valid_night_distribution=valid_night_distribution,
    )

    logging.info("Wrote %s", REST_S_BLOCKS_PATH)
    logging.info("Wrote %s", DAILY_METRICS_PATH)
    logging.info("Wrote %s", PARTICIPANT_VARIABILITY_PATH)
    logging.info("Wrote %s", FEASIBILITY_SUMMARY_PATH)
    logging.info("Wrote %s", PARTICIPANT_QUALITY_PATH)
    logging.info("Wrote %s", VALID_NIGHT_DISTRIBUTION_PATH)
    print_run_summary(feasibility)


if __name__ == "__main__":
    main()
