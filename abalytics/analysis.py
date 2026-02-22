import warnings
from typing import Optional, List

import pandas as pd
from pydantic import BaseModel, Field

from .significance_tests import (
    is_dichotomous,
    is_numeric,
    is_levene_significant,
    is_gaussian,
    are_differences_gaussian,
    get_chi_square_significance,
    get_chi_square_posthoc_results,
    get_welch_anova_significance,
    get_oneway_anova_significance,
    get_kruskal_wallis_significance,
    get_games_howell_posthoc_results,
    get_tukeyhsd_posthoc_results,
    get_dunn_posthoc_results,
    get_cochrans_q_significance,
    get_mcnemar_results,
    get_repeated_measures_anova_significance,
    get_repeated_measures_anova_posthoc_results,
    get_wilcoxon_results,
    get_friedman_significance,
    get_nemenyi_results,
)


class AnalysisResults(BaseModel):
    significant_results: Optional[List] = Field(
        default=[],
        description="A list of significant results from the statistical tests.",
    )
    info: Optional[str] = Field(
        default=None,
        description="Additional information or notes regarding the analysis.",
    )
    a_priori_test: Optional[str] = Field(
        default=None,
        description="The name of the statistical test used in the analysis.",
    )
    sample_size: Optional[int] = Field(
        default=None, description="The sample size used in the analysis."
    )
    dichotomous_flag: Optional[bool] = Field(
        default=None,
        description="Indicates if the data analyzed is dichotomous.",
    )
    levene_flag: Optional[bool] = Field(
        default=None,
        description="Indicates if Levene's test for homogeneity of variances is significant.",
    )
    gaussian_flag: Optional[bool] = Field(
        default=None,
        description="Indicates if the data follows a Gaussian distribution.",
    )


def analyze_independent_groups(
    df: pd.DataFrame,
    variable_to_analyze: str,
    group_column: str,
    p_value_threshold: float = 0.05,
    min_sample_size: int = 25,
) -> Optional[AnalysisResults]:
    """
    Returns a text with the results of the statistical significance tests.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data to analyze.
    variable_to_analyze (str): The column name in df that is to be analyzed for statistical significance.
    group_column (str): The name of the column in df that contains the grouping variable.
    p_value_threshold (float, optional): The threshold for determining statistical significance. Defaults to 0.05.
    min_sample_size (int, optional): The minimum sample size for the data. Defaults to 25.

    Returns:
    AnalysisResults: An instance of the AnalysisResults class containing the following attributes:
        - significant_results: A list of results of the statistical significance tests.
        - info: A string containing information about the data if no results were found.
        - a_priori_test: The name of the statistical test used in the analysis.
        - sample_size: The sample size of the data.
        - dichotomous_flag: A boolean flag indicating if the data is dichotomous (e.g. boolean).
        - levene_flag: A boolean flag indicating if Levene's test for homogeneity of variances is significant.
        - gaussian_flag: A boolean flag indicating if the data has a Gaussian distribution.
    """
    if p_value_threshold > 0.05:
        warnings.warn(
            "p_value_threshold is set to a value higher than the conventional alpha level of 0.05.",
            UserWarning,
            stacklevel=2,
        )

    if variable_to_analyze not in df.columns or group_column not in df.columns:
        raise ValueError(
            f"Both columns '{variable_to_analyze}' and '{group_column}' must be present in the DataFrame."
        )

    df = df.dropna(subset=[variable_to_analyze])
    sample_size = len(df)
    pvalue = None
    info = None
    a_priori_test = None
    results = None
    dichotomous_flag = False
    levene_flag = None
    gaussian_flag = None

    # Check if any group has less than the minimum sample size
    if (
        df.groupby(group_column)[variable_to_analyze]
        .apply(lambda x: len(x) < min_sample_size)
        .any()
    ):
        info = "not enough data"
    # Check if the variable is dichotomous (e.g. boolean)
    elif is_dichotomous(df, variable_to_analyze):
        dichotomous_flag = True
        pvalue = get_chi_square_significance(df, group_column, variable_to_analyze)
        a_priori_test = f"Chi-square, p-value = {pvalue:.3f}"
        if pvalue < p_value_threshold:
            results = get_chi_square_posthoc_results(
                df, group_column, variable_to_analyze, p_value_threshold
            )
    elif is_numeric(df, variable_to_analyze):
        # Step 1: Check normality first
        gaussian_flag = is_gaussian(
            df, variable_to_analyze, p_value_threshold, group_column
        )

        if gaussian_flag:
            # Data is normal — check variance homogeneity
            levene_flag = is_levene_significant(
                df, group_column, variable_to_analyze, p_value_threshold
            )

            if levene_flag:
                # Normal but unequal variances — Welch's ANOVA + Games-Howell
                pvalue = get_welch_anova_significance(
                    df, group_column, variable_to_analyze
                )
                a_priori_test = f"Welch's ANOVA, p-value = {pvalue:.3f}"
                if pvalue < p_value_threshold:
                    results = get_games_howell_posthoc_results(
                        df, group_column, variable_to_analyze, p_value_threshold
                    )
            else:
                # Normal and equal variances — One-way ANOVA + Tukey HSD
                pvalue = get_oneway_anova_significance(
                    df, group_column, variable_to_analyze
                )
                a_priori_test = f"One-way ANOVA, p-value = {pvalue:.3f}"
                if pvalue < p_value_threshold:
                    results = get_tukeyhsd_posthoc_results(
                        df, group_column, variable_to_analyze, p_value_threshold
                    )
        else:
            # Non-normal — non-parametric regardless of variance
            pvalue = get_kruskal_wallis_significance(
                df, group_column, variable_to_analyze
            )
            a_priori_test = f"Kruskal-Wallis, p-value = {pvalue:.3f}"
            if pvalue < p_value_threshold:
                results = get_dunn_posthoc_results(
                    df, group_column, variable_to_analyze, p_value_threshold
                )
    else:
        raise ValueError(
            f"Variable '{variable_to_analyze}' must be dichotomous (e.g. boolean) or numeric."
        )

    # If no results were found, return a message
    if results is None or len(results.significant_results) == 0:
        if pvalue is not None:
            info = f"not significant (p={pvalue:.3f})"

    significant_results = results.significant_results if results else []

    return AnalysisResults(
        significant_results=significant_results,
        info=info,
        a_priori_test=a_priori_test,
        sample_size=sample_size,
        dichotomous_flag=dichotomous_flag,
        levene_flag=levene_flag,
        gaussian_flag=gaussian_flag,
    )


def analyze_dependent_groups(
    df: pd.DataFrame,
    variables_to_compare: List[str],
    p_value_threshold: float = 0.05,
    min_sample_size: int = 25,
) -> Optional[AnalysisResults]:
    """
    Analyzes dependent groups using appropriate statistical tests and returns an AnalysisResults object.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data to analyze.
    variables_to_compare (List[str]): The column names in df that are to be analyzed for statistical significance.
    p_value_threshold (float, optional): The threshold for determining statistical significance. Defaults to 0.05.
    min_sample_size (int, optional): The minimum sample size for the data. Defaults to 25.

    Returns:
    AnalysisResults: An instance of the AnalysisResults class containing the following attributes:
        - significant_results: A list of results of the statistical significance tests.
        - info: A string containing information about the data if no results were found.
        - sample_size: The sample size of the data.
        - dichotomous_flag: A boolean flag indicating if the data is dichotomous (e.g. boolean).
        - levene_flag: A boolean flag indicating if Levene's test for homogeneity of variances is significant.
        - gaussian_flag: A boolean flag indicating if the data has a Gaussian distribution.
    """
    if p_value_threshold > 0.05:
        warnings.warn(
            "p_value_threshold is set to a value higher than the conventional alpha level of 0.05.",
            UserWarning,
            stacklevel=2,
        )

    missing_columns = [
        variable for variable in variables_to_compare if variable not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Columns not found in DataFrame: {', '.join(missing_columns)}"
        )

    df = df.dropna(subset=variables_to_compare)
    sample_size = len(df)
    pvalue = None
    info = None
    a_priori_test = None
    dichotomous_flag = False
    levene_flag = False
    gaussian_flag = False
    results = None

    # Check if any variable has less than the minimum sample size
    if any(df[variable].count() < min_sample_size for variable in variables_to_compare):
        info = "not enough data"
    # Check if the variables are dichotomous (e.g. boolean)
    elif all(is_dichotomous(df, variable) for variable in variables_to_compare):
        dichotomous_flag = True
        if len(variables_to_compare) > 2:
            # Use Cochran's Q as omnibus test, then McNemar posthoc
            pvalue = get_cochrans_q_significance(df, variables_to_compare)
            a_priori_test = f"Cochran's Q, p-value = {pvalue:.3f}"
            if pvalue < p_value_threshold:
                _, results = get_mcnemar_results(
                    df, variables_to_compare, p_value_threshold
                )
        else:
            # Two variables: McNemar directly
            pvalue, results = get_mcnemar_results(
                df, variables_to_compare, p_value_threshold
            )
            a_priori_test = f"McNemar, p-value = {pvalue:.3f}"
    elif all(is_numeric(df, variable) for variable in variables_to_compare):
        # Check if pairwise differences follow a Gaussian distribution
        gaussian_flag = are_differences_gaussian(
            df, variables_to_compare, p_value_threshold
        )
        if gaussian_flag:
            pvalue = get_repeated_measures_anova_significance(df, variables_to_compare)
            a_priori_test = f"Repeated measures ANOVA, p-value = {pvalue:.3f}"
            if pvalue < p_value_threshold:
                results = get_repeated_measures_anova_posthoc_results(
                    df, variables_to_compare, p_value_threshold
                )
        elif len(variables_to_compare) == 2:
            pvalue, results = get_wilcoxon_results(
                df, variables_to_compare, p_value_threshold
            )
            a_priori_test = f"Wilcoxon, p-value = {pvalue:.3f}"
        else:
            pvalue = get_friedman_significance(df, variables_to_compare)
            a_priori_test = f"Friedman, p-value = {pvalue:.3f}"
            if pvalue < p_value_threshold:
                results = get_nemenyi_results(
                    df, variables_to_compare, p_value_threshold
                )
    else:
        raise ValueError(
            "All variables must be dichotomous (e.g. boolean) or numeric."
        )

    # If no results were found, return a message
    if not results or len(results.significant_results) == 0:
        if pvalue is not None:
            info = f"not significant (p={pvalue:.3f})"

    significant_results = results.significant_results if results else []

    return AnalysisResults(
        significant_results=significant_results,
        info=info,
        a_priori_test=a_priori_test,
        sample_size=sample_size,
        dichotomous_flag=dichotomous_flag,
        levene_flag=levene_flag,
        gaussian_flag=gaussian_flag,
    )
