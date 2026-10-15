# MESA Sleep Variability and Daytime Sleepiness

MESA Sleep Study의 손목 액티그래피에서 개인 내 수면 변동성을 산출하고,
Epworth Sleepiness Scale(ESS)과의 관련성을 분석한 Python 코드입니다.
초기 파일럿, 전체 표본 분석, 추가 보정 및 논문 Table 1 생성 코드를 포함합니다.

이 저장소에는 코드와 문서만 들어 있습니다. 원자료, 참가자별 파생 데이터,
실행 로그, 분석 결과 및 논문 파일은 포함하지 않습니다.
데이터는 적절한 접근 권한을 확보한 뒤 별도로 준비해야 합니다.

## 분석 개요

- Primary window: `consecutive_7_daybynoon`
- Outcome: ESS (`epslpscl5c`)
- 기본 보정: 평균 수면시간, 나이, 성별, BMI, AHI
- 추정: OLS + HC3 robust standard errors
- 주요 변동성 지표: 수면시간 SD; 추가 지표: 수면효율 SD, WASO SD, midpoint circular SD
- 최종 기술통계 스크립트가 기대하는 기존 complete-case 표본: 1,527명
- 관찰자료의 관련성 분석이며, 인과효과를 추정한 것으로 해석하지 않습니다.

세부 정의와 유효 night 기준은 [분석 방법](docs/analysis_methods.md)을 참고하세요.

## 설치

Python 3.10 이상이 필요합니다. Windows PowerShell 예시:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS에서는 가상환경 활성화에 `source .venv/bin/activate`를 사용합니다.
그림 스크립트는 Windows 글꼴을 우선 탐색하며, 없는 경우 Pillow 기본 글꼴로
대체하므로 다른 운영체제에서는 글꼴과 그림의 외형이 달라질 수 있습니다.

## 로컬 데이터 배치

```text
data/
  raw/
    actigraphy/       # mesa-sleep-*.csv, 30-second epochs
    datasets/
      mesa-sleep-dataset-0.8.0.csv
      mesa-data-dictionary-0.8.0-variables.csv
      mesa-data-dictionary-0.8.0-domains.csv
      mesa-data-dictionary-0.8.0-forms.csv
    documentation/    # Official MESA manuals and data documentation
  interim/            # Generated REST-S blocks and sleep features
  processed/          # Generated regression and validation datasets
outputs/
  tables/
  figures/
  logs/
  backups/
```

사용한 파일명과 필요한 변수는 [데이터 안내](docs/data_inputs.md)에 정리했습니다.
각 스크립트는 자신의 파일 위치에서 프로젝트 루트를 계산합니다.
원자료 폴더와 파일명은 위 구조에 맞춰야 합니다.

## 스크립트 순서

| 번호 | 파일 | 역할 |
| --- | --- | --- |
| 01 | `01_inspect_data.py` | 파일, 컬럼, ID, phenotype 변수 점검 |
| 02 | `02_validate_actigraphy.py` | REST-S 후보 구간, main sleep, 일별 지표 및 초기 QC |
| 03 | `03_refine_feasibility_analysis.py` | 공식 정의, QC 수준별 집계, 유효 night 민감도 |
| 04 | `04_resolve_ids_and_validate_sleep_features.py` | ID 검증, 경계 QC, 분석 window, 공식 지표와 비교 |
| 05 | `05_pilot_regression_analysis.py` | 초기 파일럿 모델과 수동 회귀 계산 |
| 06 | `06_validate_regression_with_statsmodels.py` | 파일럿 계산을 statsmodels와 비교 |
| 07 | `07_full_regression_analysis.py` | 전체 표본 Model 0-3 및 진단 |
| 08 | `08_compare_variability_metrics_and_quartiles.py` | 변동성별 비교, 공통/가용 표본, FDR, 사분위수 |
| 09 | `09_mean_adjusted_and_joint_variability_analysis.py` | 각 지표 평균 보정, joint model, 상관, VIF, 보정 사분위수 |
| 10 | `10_final_robustness_and_nonlinearity_checks.py` | joint model 민감도, 사분위수, 이차항, 최종 Figure 1 |
| 11 | `11_create_table1_descriptive_statistics.py` | 최종 표본의 논문 Table 1 |

전체 표본 분석 경로는 다음과 같습니다. 각 명령은 앞 단계가 성공한 뒤 실행하세요.

```powershell
python src/01_inspect_data.py
python src/02_validate_actigraphy.py
python src/03_refine_feasibility_analysis.py
python src/04_resolve_ids_and_validate_sleep_features.py
python src/07_full_regression_analysis.py
python src/08_compare_variability_metrics_and_quartiles.py
python src/09_mean_adjusted_and_joint_variability_analysis.py
python src/10_final_robustness_and_nonlinearity_checks.py
python src/11_create_table1_descriptive_statistics.py
```

`05`와 `06`은 기존 파일럿의 계산 및 검증 기록입니다. 해당 비교를 재현하려면
동일한 파일럿 입력 범위를 별도 작업 복사본에 준비하고 `05`, `06` 순서로 실행하세요.
현재 전체 원자료로 `05`를 실행한다고 과거의 소규모 파일럿이 자동 복원되지는 않습니다.

## 모델

모든 모델의 종속변수는 ESS입니다.

| 모델 | 설명변수 |
| --- | --- |
| Model 0 | age + sex_Male + bmi + ahi |
| Model 1 | Model 0 + sleep_duration_mean |
| Model 2 | Model 1 + sleep_duration_sd |
| Model 3 | Model 1 + sleep_duration_cv |

`08`은 평균 수면시간을 공통으로 보정한 지표 비교입니다.
`09`는 각 변동성 지표의 대응 평균을 보정하고, 평균 4종과 변동성 4종을 함께
넣은 joint model을 추가합니다. `10`은 평균 수면효율 또는 평균 WASO를 제외한
joint model과 선형/이차항 모델을 비교합니다. 따라서 단계별 모델은 동일하지 않습니다.

## 출력과 재실행

- `data/processed/full_regression_dataset.csv`: 전체 회귀 입력과 분석대상 표시
- `outputs/tables/`: 회귀계수, 표본 흐름, FDR, 사분위수, VIF, Table 1
- `outputs/figures/sleep_duration_sd_adjusted_quartile_final.png`: 최종 보정 사분위수 그림
- `outputs/logs/`: 실행 기록

원자료를 수정하는 코드는 없습니다. 재실행하면 생성 파일은 갱신됩니다.
여러 후속 스크립트는 `outputs/backups/<timestamp>/`에 기존 출력을 보관하지만,
초기 `01`, `02`는 모든 출력을 자동 백업하지 않으므로 재실행 전 작업 폴더를 보관하세요.
`10`은 `09`의 joint model 결과 재현을 검사하고, `11`은 표본 수가 1,527명이 아니면
중단합니다. 다른 데이터 릴리스에서 표본 수를 맞추기 위해 QC나 변수를 바꾸면 안 됩니다.

## 코드 확인

```powershell
python -m compileall -q src tests
python -m unittest discover -s tests -v
```

테스트는 합성 입력으로 SD/CV, 원형통계, 연속 window 및 모델 정의를 확인합니다.
연구 원자료를 사용한 전체 재분석이나 기존 추정치의 수치적 재현을 대신하지 않습니다.
설치된 패키지 버전을 보관하려면 실행 환경에서 `python -m pip freeze > requirements.lock.txt`를
실행하세요. 현재 requirements는 버전을 고정하지 않았습니다.

## 저장소 범위

분석 프로젝트의 11개 스크립트를 정리한 코드 배포본입니다.
기존 분석 작업 폴더와 원자료는 그대로 보존했습니다.
배포를 위한 변경은 의존성 목록 보완, Codex 전용 패키지 경로 제거, 문서 및 테스트 추가입니다.
별도의 재배포 라이선스는 아직 지정하지 않았습니다.
