"""Inspect raw MESA actigraphy and dataset CSV files.

The script is intentionally read-only for data/raw. It uses the location of
this file to resolve the project root, so it does not depend on the shell's
current working directory.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pandas.errors import EmptyDataError, ParserError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ACTIGRAPHY_DIR = RAW_DIR / "actigraphy"
DATASET_DIR = RAW_DIR / "datasets"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
REPORT_PATH = OUTPUT_TABLE_DIR / "data_inspection_report.csv"
LOG_PATH = OUTPUT_LOG_DIR / "01_inspect_data.log"

ACTIGRAPHY_PATTERN = "mesa-sleep-*.csv"
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1")

EXPECTED_ACTIGRAPHY_COLUMNS = [
    "mesaid",
    "line",
    "linetime",
    "offwrist",
    "activity",
    "wake",
    "interval",
    "dayofweek",
    "daybymidnight",
    "daybynoon",
    "date_shifted",
    "actigraphy_month_start",
]

TARGET_DATASET_VARIABLES = [
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


@dataclass
class ReportRow:
    section: str
    file: str
    metric: str
    value: str
    status: str
    details: str = ""


def ensure_output_dirs() -> None:
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


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def row(
    section: str,
    file: str,
    metric: str,
    value: Any,
    status: str,
    details: Any = "",
) -> ReportRow:
    return ReportRow(section, file, metric, serialize(value), status, serialize(details))


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


def read_header(path: Path) -> list[str]:
    return [str(column) for column in read_csv_with_fallback(path, nrows=0).columns]


def count_csv_rows(path: Path) -> int:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as file_obj:
                reader = csv.reader(file_obj)
                try:
                    next(reader)
                except StopIteration:
                    return 0
                return sum(1 for _ in reader)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return 0


def normalize_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False)
    normalized: dict[str, int] = {}
    for value, count in counts.items():
        label = "<NA>" if pd.isna(value) else str(value)
        normalized[label] = int(count)
    return dict(sorted(normalized.items(), key=lambda item: item[0]))


def find_values_in_csv(path: Path, target_variables: Iterable[str]) -> list[str]:
    targets = set(target_variables)
    found: set[str] = set()
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as file_obj:
                for csv_row in csv.reader(file_obj):
                    for cell in csv_row:
                        if cell.strip() in targets:
                            found.add(cell.strip())
                    if found == targets:
                        return sorted(found)
            return sorted(found)
        except UnicodeDecodeError as exc:
            last_error = exc
            found.clear()
    if last_error is not None:
        raise last_error
    return sorted(found)


def list_actigraphy_files() -> list[Path]:
    if not ACTIGRAPHY_DIR.exists():
        raise FileNotFoundError(f"Actigraphy directory does not exist: {ACTIGRAPHY_DIR}")
    files = sorted(ACTIGRAPHY_DIR.glob(ACTIGRAPHY_PATTERN))
    if not files:
        raise FileNotFoundError(
            f"No actigraphy files matched {ACTIGRAPHY_DIR / ACTIGRAPHY_PATTERN}"
        )
    return files


def inspect_actigraphy_schema(paths: list[Path]) -> list[ReportRow]:
    rows: list[ReportRow] = []
    expected = set(EXPECTED_ACTIGRAPHY_COLUMNS)
    reference_columns: list[str] | None = None
    schema_counter: Counter[tuple[str, ...]] = Counter()
    failed_files = 0

    for path in paths:
        label = relative_path(path)
        try:
            columns = read_header(path)
            schema_counter[tuple(columns)] += 1
            if reference_columns is None:
                reference_columns = columns

            missing = sorted(expected.difference(columns))
            unexpected = sorted(set(columns).difference(expected))
            rows.append(row("actigraphy_schema", label, "column_count", len(columns), "ok"))
            rows.append(
                row(
                    "actigraphy_schema",
                    label,
                    "schema_matches_first_file",
                    columns == reference_columns,
                    "ok" if columns == reference_columns else "warning",
                )
            )
            rows.append(
                row(
                    "actigraphy_schema",
                    label,
                    "missing_required_columns",
                    missing,
                    "warning" if missing else "ok",
                )
            )
            rows.append(
                row(
                    "actigraphy_schema",
                    label,
                    "unexpected_columns",
                    unexpected,
                    "info" if unexpected else "ok",
                )
            )
        except (OSError, EmptyDataError, ParserError, UnicodeDecodeError, ValueError) as exc:
            failed_files += 1
            logging.exception("Failed to read actigraphy header: %s", path)
            rows.append(
                row(
                    "actigraphy_schema",
                    label,
                    "schema_read_error",
                    type(exc).__name__,
                    "error",
                    str(exc),
                )
            )

    rows.append(row("actigraphy_summary", "", "unique_schema_count", len(schema_counter), "ok"))
    rows.append(
        row(
            "actigraphy_summary",
            "",
            "schema_consistent",
            len(schema_counter) == 1 and failed_files == 0,
            "ok" if len(schema_counter) == 1 and failed_files == 0 else "warning",
        )
    )
    rows.append(row("actigraphy_summary", "", "schema_read_failed_file_count", failed_files, "ok"))
    return rows


def inspect_actigraphy_details(paths: list[Path]) -> list[ReportRow]:
    rows: list[ReportRow] = []
    expected = set(EXPECTED_ACTIGRAPHY_COLUMNS)
    detail_columns = ["mesaid", "interval", "wake", "offwrist", "date_shifted"]

    for path in paths[:5]:
        label = relative_path(path)
        try:
            columns = read_header(path)
            available_detail_columns = [name for name in detail_columns if name in columns]
            df = read_csv_with_fallback(path, usecols=available_detail_columns)
            missing = sorted(expected.difference(columns))
            unexpected = sorted(set(columns).difference(expected))

            rows.extend(
                [
                    row("actigraphy_file", label, "row_count", len(df), "ok"),
                    row("actigraphy_file", label, "column_count", len(columns), "ok"),
                    row(
                        "actigraphy_file",
                        label,
                        "mesaid_unique_count",
                        int(df["mesaid"].nunique(dropna=True)) if "mesaid" in df else "",
                        "ok" if "mesaid" in df and df["mesaid"].nunique(dropna=True) == 1 else "warning",
                    ),
                    row(
                        "actigraphy_file",
                        label,
                        "missing_required_columns",
                        missing,
                        "warning" if missing else "ok",
                    ),
                    row(
                        "actigraphy_file",
                        label,
                        "unexpected_columns",
                        unexpected,
                        "info" if unexpected else "ok",
                    ),
                ]
            )

            for column in ["interval", "wake", "offwrist"]:
                if column in df:
                    rows.append(
                        row(
                            "actigraphy_file",
                            label,
                            f"{column}_value_distribution",
                            normalize_counts(df[column]),
                            "ok",
                        )
                    )
                else:
                    rows.append(
                        row(
                            "actigraphy_file",
                            label,
                            f"{column}_value_distribution",
                            "",
                            "warning",
                            "Column missing.",
                        )
                    )

            if "date_shifted" in df:
                parsed_dates = pd.to_datetime(df["date_shifted"], errors="coerce")
                rows.append(
                    row(
                        "actigraphy_file",
                        label,
                        "date_shifted_min",
                        parsed_dates.min() if parsed_dates.notna().any() else "",
                        "ok" if parsed_dates.notna().any() else "warning",
                    )
                )
                rows.append(
                    row(
                        "actigraphy_file",
                        label,
                        "date_shifted_max",
                        parsed_dates.max() if parsed_dates.notna().any() else "",
                        "ok" if parsed_dates.notna().any() else "warning",
                    )
                )

        except (OSError, EmptyDataError, ParserError, UnicodeDecodeError, ValueError) as exc:
            logging.exception("Failed to inspect actigraphy file: %s", path)
            rows.append(
                row(
                    "actigraphy_file",
                    label,
                    "parse_error",
                    type(exc).__name__,
                    "error",
                    str(exc),
                )
            )

    return rows


def inspect_dataset_files() -> list[ReportRow]:
    rows: list[ReportRow] = []
    variable_to_column_files: dict[str, list[str]] = {
        variable: [] for variable in TARGET_DATASET_VARIABLES
    }
    variable_to_value_files: dict[str, list[str]] = {
        variable: [] for variable in TARGET_DATASET_VARIABLES
    }

    if not DATASET_DIR.exists():
        rows.append(
            row(
                "dataset_summary",
                relative_path(DATASET_DIR),
                "dataset_directory_exists",
                False,
                "warning",
            )
        )
        return rows

    dataset_files = sorted(DATASET_DIR.rglob("*.csv"))
    rows.append(
        row(
            "dataset_summary",
            relative_path(DATASET_DIR),
            "dataset_csv_file_count",
            len(dataset_files),
            "ok" if dataset_files else "warning",
        )
    )

    for path in dataset_files:
        label = relative_path(path)
        try:
            columns = read_header(path)
            row_count = count_csv_rows(path)
            column_hits = [name for name in TARGET_DATASET_VARIABLES if name in columns]
            value_hits = find_values_in_csv(path, TARGET_DATASET_VARIABLES)

            for name in column_hits:
                variable_to_column_files[name].append(label)
            for name in value_hits:
                variable_to_value_files[name].append(label)

            rows.extend(
                [
                    row("dataset_file", label, "row_count", row_count, "ok"),
                    row("dataset_file", label, "column_count", len(columns), "ok"),
                    row("dataset_file", label, "target_variables_as_columns", column_hits, "ok"),
                    row("dataset_file", label, "target_variables_as_values", value_hits, "info"),
                ]
            )
        except (OSError, EmptyDataError, ParserError, UnicodeDecodeError, ValueError) as exc:
            logging.exception("Failed to inspect dataset file: %s", path)
            rows.append(
                row(
                    "dataset_file",
                    label,
                    "parse_error",
                    type(exc).__name__,
                    "error",
                    str(exc),
                )
            )

    for variable in TARGET_DATASET_VARIABLES:
        rows.append(
            row(
                "dataset_variable_summary",
                "",
                variable,
                {
                    "as_column": variable_to_column_files[variable],
                    "as_cell_value": variable_to_value_files[variable],
                },
                "ok"
                if variable_to_column_files[variable] or variable_to_value_files[variable]
                else "warning",
            )
        )

    return rows


def write_report(rows: list[ReportRow]) -> None:
    report = pd.DataFrame([asdict(item) for item in rows])
    report.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    setup_logging()
    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Current working directory: %s", Path.cwd())
    logging.info("Actigraphy directory: %s", ACTIGRAPHY_DIR)

    actigraphy_files = list_actigraphy_files()
    rows: list[ReportRow] = [
        row("path_check", "", "project_root", PROJECT_ROOT, "ok"),
        row("path_check", "", "current_working_directory", Path.cwd(), "ok"),
        row("path_check", relative_path(ACTIGRAPHY_DIR), "exists", ACTIGRAPHY_DIR.exists(), "ok"),
        row(
            "actigraphy_summary",
            relative_path(ACTIGRAPHY_DIR),
            "actigraphy_csv_file_count",
            len(actigraphy_files),
            "ok",
            f"Pattern: {ACTIGRAPHY_PATTERN}",
        ),
        row(
            "actigraphy_summary",
            relative_path(ACTIGRAPHY_DIR),
            "first_10_actigraphy_files",
            [path.name for path in actigraphy_files[:10]],
            "ok",
        ),
    ]

    rows.extend(inspect_actigraphy_schema(actigraphy_files))
    rows.extend(inspect_actigraphy_details(actigraphy_files))
    rows.extend(inspect_dataset_files())
    write_report(rows)

    status_counts = Counter(item.status for item in rows)
    logging.info("Wrote inspection report: %s", REPORT_PATH)
    logging.info("Actigraphy CSV files: %s", len(actigraphy_files))
    logging.info("Report rows: %s", len(rows))
    logging.info("Warnings: %s", status_counts.get("warning", 0))
    logging.info("Errors: %s", status_counts.get("error", 0))
    print(f"Inspection report written: {relative_path(REPORT_PATH)}")
    print(f"Actigraphy CSV files: {len(actigraphy_files)}")
    print(f"Report rows: {len(rows)}")
    print(f"Warnings: {status_counts.get('warning', 0)}")
    print(f"Errors: {status_counts.get('error', 0)}")


if __name__ == "__main__":
    main()
