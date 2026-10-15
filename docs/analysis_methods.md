# Analysis Methods

## Main sleep and valid nights

1. Retain contiguous intervals labeled `REST-S` as candidate blocks.
2. Select the longest candidate within a participant's `daybynoon` as provisional main sleep.
3. Exclude intervals shorter than 3 hours or longer than 16 hours, excessive missing wake
   scores, non-contiguous epochs, missing sleep epochs, invalid efficiency, and any offwrist epochs.
4. Use the `valid_B2` boundary-aware rule from script 04: exclude blocks touching the
   recording start or end within the implemented 30-second tolerance. The first or last
   recording day is not excluded solely because it is the first or last day.

The high-missing-wake threshold is 10%. Script 04 also records the availability of
30-minute pre/post context, but `boundary_context_insufficient` is a diagnostic flag:
it is not an additional exclusion in `valid_B2`. See `base_valid_mask` and
`apply_valid_night_criteria` for the operative conditions.

`REST-S` is used operationally as a candidate-interval label. Its exact official label
definition, including the meaning of the suffix, was not confirmed in the initial
definition review. Do not describe the entire interval as confirmed sleep.

## Analysis windows

- Primary: the earliest available run of seven valid, consecutive numeric `daybynoon` values.
- Sensitivity: first seven valid nights, without requiring consecutive daybynoon values.
- Sensitivity: all valid nights.

The primary window is selected by temporal availability, not ESS or statistical significance.

## Features and units

| Feature | Definition | Unit |
| --- | --- | --- |
| sleep_duration_mean | Mean daily sleep duration; sleep epochs are wake=0 | minutes |
| sleep_duration_sd | Sample SD across the selected nights, ddof=1 | minutes |
| sleep_duration_cv | Duration SD divided by mean duration | ratio, not percent |
| sleep_efficiency_mean | Mean of daily sleep-epoch percentages among valid wake scores | percent |
| sleep_efficiency_sd | Sample SD of daily efficiency, ddof=1 | percentage points |
| waso_mean | Mean wake time between the first and last sleep epochs | minutes |
| waso_sd | Sample SD of daily WASO, ddof=1 | minutes |
| sleep_midpoint_circular_mean | Circular mean of interval midpoints over a 24-hour cycle | minutes after midnight |
| sleep_midpoint_circular_sd | Circular SD converted from angular units | minutes |

For daily midpoint clock minutes `t`, use `theta = 2*pi*t/1440`,
`R = hypot(mean(sin(theta)), mean(cos(theta)))`, and
`circular_sd_minutes = sqrt(-2*log(R))*1440/(2*pi)`.
This is circular dispersion, not an ordinary clock-time sample SD with ddof=1.
The implemented midpoint is the midpoint of the candidate interval boundaries.

## Inference and interpretation

OLS coefficients use HC3 covariance for inference; R-squared, AIC, BIC and related
fit statistics describe the fitted OLS model. Standardized beta is the raw coefficient
multiplied by predictor SD and divided by outcome SD on the relevant analytic sample.
BH-FDR is implemented in the scripts; correction families depend on the output column.

Quartile models compare Q2, Q3, Q4 with Q1. Adjusted ESS uses marginal standardized
predictions over the analysis sample. Trend tests enter quartile rank as 1-4.
A significant trend test alone does not establish a linear dose-response shape.
The quadratic comparison in script 10 addresses one specific nonlinear form.

The joint model includes the mean midpoint as an existing scalar clock-minute
covariate while preserving circular dispersion for midpoint variability. A fully
periodic sine/cosine model for the mean midpoint is not implemented here.

These are observational associations. Statistical significance, effect size, and
incremental explained variance must be reported separately.
