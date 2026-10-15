"""Small synthetic checks; no participant data or analysis outputs are read."""

import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_script(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'src' / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


features = load_script('04_resolve_ids_and_validate_sleep_features.py', 'sleep_features')
models = load_script('07_full_regression_analysis.py', 'full_models')


class FeatureContractTests(unittest.TestCase):
    def test_midpoints_wrap_at_midnight(self):
        mean, sd = features.circular_stats(pd.Series([1430.0, 10.0]))
        self.assertLess(min(abs(mean), abs(1440.0 - mean)), 1e-8)
        self.assertAlmostEqual(sd, 10.0, delta=0.01)

    def test_single_midpoint_has_no_variability_estimate(self):
        mean, sd = features.circular_stats(pd.Series([195.0]))
        self.assertAlmostEqual(mean, 195.0)
        self.assertTrue(pd.isna(sd))

    def test_consecutive_window_selects_first_complete_run(self):
        days = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
        rows = pd.DataFrame({
            'daybynoon': days,
            'block_start_dt': pd.date_range('2020-01-01', periods=len(days)),
        })
        selected = features.consecutive_window(rows)
        self.assertEqual(selected['daybynoon'].tolist(), list(range(4, 11)))

    def test_seven_nights_with_gap_are_not_consecutive(self):
        rows = pd.DataFrame({
            'daybynoon': [1, 2, 3, 5, 6, 7, 8],
            'block_start_dt': pd.date_range('2020-01-01', periods=7),
        })
        self.assertTrue(features.consecutive_window(rows).empty)

    def test_sample_sd_and_cv_units(self):
        rows = pd.DataFrame({
            'mesaid': ['synthetic', 'synthetic'],
            'daybynoon': [1, 2],
            'block_start_dt': pd.to_datetime(['2020-01-01', '2020-01-02']),
            'block_end_dt': pd.to_datetime(['2020-01-01 07:00', '2020-01-02 07:00']),
            'sleep_duration_minutes': [300.0, 420.0],
            'sleep_efficiency': [90.0, 94.0],
            'waso_minutes': [10.0, 20.0],
            'sleep_midpoint_clock_minutes': [180.0, 200.0],
        })
        result = features.summarize_feature_window('synthetic', 'synthetic', rows)
        self.assertAlmostEqual(result['sleep_duration_mean'], 360.0)
        self.assertAlmostEqual(result['sleep_duration_sd'], 120.0 / 2 ** 0.5)
        self.assertAlmostEqual(result['sleep_duration_cv'], result['sleep_duration_sd'] / 360.0)


class ModelContractTests(unittest.TestCase):
    def test_duration_sd_and_cv_are_separate_extensions(self):
        baseline = models.MODEL_TERMS['Model 1']
        self.assertEqual(set(models.MODEL_TERMS['Model 2']), set(baseline) | {'sleep_duration_sd'})
        self.assertEqual(set(models.MODEL_TERMS['Model 3']), set(baseline) | {'sleep_duration_cv'})
        self.assertEqual(models.PRIMARY_FEATURE_SET, 'consecutive_7_daybynoon')


if __name__ == '__main__':
    unittest.main()
