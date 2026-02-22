# Changelog

## 4.0.1

### Breaking Changes

- **Decision tree restructured**: Independent groups now check normality **before** variance homogeneity (Levene). Non-normal data always routes to Kruskal-Wallis regardless of variance. Previously, non-normal data with unequal variances was incorrectly routed to Welch's ANOVA.
- **Error handling**: `print()` calls replaced with `warnings.warn()` for non-critical issues and `raise ValueError()` for invalid inputs (missing columns, non-numeric/non-dichotomous data).
- **Non-parametric result text**: Dunn, Wilcoxon, and Nemenyi posthoc results now report **medians** instead of means, consistent with rank-based test assumptions.

### Bug Fixes

- **McNemar return type crash** (`significance_tests.py`): `get_mcnemar_results()` now correctly returns `(float, PosthocResults)` tuple. Previously returned only `PosthocResults`, causing `ValueError: not enough values to unpack` when called from `analyze_dependent_groups()`.
- **Wilcoxon false positives** (`significance_tests.py`): Wilcoxon signed-rank results are now gated by `p_value < p_value_threshold`. Previously, all results were added to `significant_results` unconditionally.
- **Decision tree order** (`analysis.py`): Normality is now checked before Levene's test. This fixes the bug where non-normal data with unequal variances was routed to Welch's ANOVA (parametric) instead of Kruskal-Wallis (non-parametric). Also ensures `gaussian_flag` is always set for numeric data.

### Statistical Correctness

- **Medians for non-parametric tests**: `get_dunn_posthoc_results()`, `get_wilcoxon_results()`, and `get_nemenyi_results()` now sort by and display median values instead of means.
- **Sphericity correction for RM ANOVA**: Replaced `statsmodels.AnovaRM` with `pingouin.rm_anova()`, which automatically computes Mauchly's sphericity test and applies Greenhouse-Geisser correction when sphericity is violated.
- **Normality of differences**: Dependent groups now tests normality of **pairwise differences** (via new `are_differences_gaussian()`) instead of raw variables, which is the correct assumption for repeated measures designs.
- **Fisher's exact fallback**: Chi-square test now falls back to Fisher's exact test for 2x2 contingency tables when any expected frequency is below 5.
- **Small sample normality guard**: `is_gaussian()` returns `False` for samples with n < 20 (policy: use safer non-parametric path with small samples).
- **McNemar exact test**: Uses `exact=True` when discordant cell count (b + c) < 25 for more accurate small-sample results.
- **Cochran's Q omnibus test**: Added `get_cochrans_q_significance()` for >2 dependent dichotomous variables. Cochran's Q is now used as the omnibus test before pairwise McNemar posthoc comparisons.
- **McNemar Bonferroni correction**: When >2 dependent dichotomous variables are compared pairwise, p-values are now Bonferroni-corrected.

### Improvements

- `a_priori_test` is now populated for all dependent group paths (McNemar, Cochran's Q, RM ANOVA, Wilcoxon, Friedman).
- Consistent non-significant result message format: `"not significant (p={pvalue:.3f})"` across both independent and dependent group analyses.
- Chi-square posthoc positive value detection is now dynamic (handles bool, int, and string binary encodings) instead of hardcoded `1` / `True`.
- Eliminated `locals()` antipattern and walrus operator overuse in `analysis.py`.

### Tests

- Added comprehensive test suite with 70 tests across 3 files:
  - `test_analysis.py` (21 tests): Full decision tree coverage for both independent and dependent groups
  - `test_significance_tests.py` (38 tests): Unit tests for all statistical functions
  - `test_utils.py` (8 tests): Formatting and data transformation utilities
- Shared fixtures in `conftest.py` for reproducible test data

---

## 3.0.6

- Updated dependencies in setup.py
- Improved Tukey HSD results handling
- Fixed typo in utils.py documentation

## 3.0.5

- Changed: means displayed as .3f instead of .2f

## 3.0.4

- Fixed: restored `convert_long_to_wide`

## 3.0.3

- Improved: README

## 3.0.2

- Fixed: missing tabulate requirement

## 3.0.1

- Added: support for other table formats (incl. HTML)
- Improved: number formatting in table
- Fixed: example.py

## 3.0.0

- Improved: formatting of table (uses tabulate)
- Improved: returning a priori tests
- Fixed: Dunn returned results twice

## 2.0.0

- Added: analysis method is now returned
- Added: a priori test is returned
- Fixed: typo (Kruskal-Wallis)
