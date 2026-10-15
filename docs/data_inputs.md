# Data Inputs

All paths are relative to the repository root. Data must be obtained separately
with the required study access permissions. No automated data download is included.

## Raw inputs

| Path | Role |
| --- | --- |
| data/raw/actigraphy/mesa-sleep-*.csv | Participant-level 30-second epoch recordings |
| data/raw/datasets/mesa-sleep-dataset-0.8.0.csv | ESS, covariates, official actigraphy summary variables |
| data/raw/datasets/mesa-data-dictionary-0.8.0-variables.csv | Variable descriptions |
| data/raw/datasets/mesa-data-dictionary-0.8.0-domains.csv | Categorical code definitions, including gender1 |
| data/raw/datasets/mesa-data-dictionary-0.8.0-forms.csv | Supplementary dictionary forms |
| data/raw/documentation/ | MESA manuals and data documentation used in definition checks |

The full regression uses the native 0.8.0 phenotype dataset, not the harmonized
dataset or archived earlier releases. The prior local input inventory contained
2,159 actigraphy CSV files. This is a historical input count, not an eligibility rule.

Native phenotype columns used by the full analysis include:

| Native column | Analysis role |
| --- | --- |
| mesaid | Participant linkage |
| epslpscl5c | ESS outcome |
| sleepage5c | age |
| gender1 | sex; official domains must confirm 0=Female, 1=Male |
| bmi5c | BMI |
| ahi_a0h3a | AHI |
| avgmainsleep5 | Official mean sleep duration for validation |
| avgefficiency5 | Official mean sleep efficiency for validation |
| avgwaso5 | Official mean WASO for validation |
| avgsleepmidpoint5 | Official mean midpoint for validation |

Do not replace these columns with similarly named harmonized variables without
an explicitly specified new analysis. Required actigraphy columns and schema
validation are defined in scripts 01 and 02.

## Generated inputs

- `data/interim/rest_s_blocks.csv`: all candidate blocks and flags.
- `data/interim/provisional_daily_sleep_metrics.csv`: selected daily main-sleep metrics.
- `data/interim/participant_sleep_variability.csv`: initial variability summary.
- `data/interim/participant_sleep_features_by_window.csv`: boundary-aware window features.
- `data/processed/full_regression_dataset.csv`: merged full analysis input.
- `data/processed/pilot_regression_dataset.csv`: historical pilot analysis input.
- `data/processed/sleep_feature_validation_dataset.csv`: feature-validation input.

`full_regression_dataset.csv` contains eligibility indicators and is not necessarily
limited to final complete cases. Script 11 filters `feature_set` to
`consecutive_7_daybynoon` and selects `primary_common_complete_case` before checking
that the historical final sample has 1,527 rows.

`outputs/tables` also contains participant-level audits, mappings and diagnostic
outputs in addition to aggregate tables. The entire outputs directory is excluded
from Git; the repository does not assume every results CSV is aggregate-only.
