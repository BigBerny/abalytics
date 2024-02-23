from typing import Optional, List

import pandas as pd
from pydantic import BaseModel, Field

from .significance_tests import (
    is_dichotomous,
    is_numeric,
    is_levene_significant,
    is_gaussian,
    get_chi_square_significance,
    get_chi_square_posthoc_results,
    get_welch_anova_significance,
    get_oneway_anova_significance,
    get_crushal_wallis_significance,
    get_games_howell_posthoc_results,
    get_tukeyhsd_posthoc_results,
    get_dunn_posthoc_results,
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
        - sample_size: The sample size of the data.
        - dichotomous_flag: A boolean flag indicating if the data is dichotomous (e.g. boolean).
        - levene_flag: A boolean flag indicating if Levene's test for homogeneity of variances is significant.
        - gaussian_flag: A boolean flag indicating if the data has a Gaussian distribution.
    """
    if p_value_threshold > 0.05:
        print(
            "Warning: p_value_threshold is set to a value higher than the conventional alpha level of 0.05."
        )

    if variable_to_analyze not in df.columns or group_column not in df.columns:
        print(
            f"Both columns {variable_to_analyze} and {group_column} have to be present in the DataFrame."
        )
        return None

    df = df.dropna(subset=[variable_to_analyze])
    sample_size = len(df)
    pvalue = None
    info = None
    dichotomous_flag = False
    levene_flag = False
    gaussian_flag = False
    # Check if any group has less than the minimum sample size
    if (
        df.groupby(group_column)[variable_to_analyze]
        .apply(lambda x: len(x) < min_sample_size)
        .any()
    ):
        info = "not enough data"
    # Check if the variable is dichotomous (e.g. boolean)
    elif dichotomous_flag := is_dichotomous(df, variable_to_analyze):
        # Perform chi-square test
        pvalue = get_chi_square_significance(
            df, group_column, variable_to_analyze
        )
        if pvalue < p_value_threshold:
            results = get_chi_square_posthoc_results(
                df, group_column, variable_to_analyze, p_value_threshold
            )
    elif is_numeric(df, variable_to_analyze):
        # Check homogeneity of variances. If significant, use non-parametric tests
        if levene_flag := is_levene_significant(
            df, group_column, variable_to_analyze, p_value_threshold
        ):
            pvalue = get_welch_anova_significance(
                df, group_column, variable_to_analyze
            )
            if pvalue < p_value_threshold:
                results = get_games_howell_posthoc_results(
                    df, group_column, variable_to_analyze, p_value_threshold
                )
        # Check if the data has Gaussian distribution, if not use non-parametric tests
        elif gaussian_flag := is_gaussian(
            df, variable_to_analyze, p_value_threshold, group_column
        ):
            pvalue = get_oneway_anova_significance(
                df, group_column, variable_to_analyze
            )
            if pvalue < p_value_threshold:
                results = get_tukeyhsd_posthoc_results(
                    df, group_column, variable_to_analyze, p_value_threshold
                )
        else:
            pvalue = get_crushal_wallis_significance(
                df, group_column, variable_to_analyze
            )
            if pvalue < p_value_threshold:
                results = get_dunn_posthoc_results(
                    df, group_column, variable_to_analyze, p_value_threshold
                )
    else:
        print("Variable has to be dichotomous (e.g. boolean) or numeric")
        info = "Variable has to be dichotomous (e.g. boolean) or numeric"

    # If no results were found, return a message
    if not "results" in locals() or len(results.significant_results) == 0:
        if pvalue is not None:
            info = f"not significant (p={pvalue:.3f})"
    # If results were found, return them
    if "results" in locals():
        significant_results = results.significant_results
    else:
        significant_results = []

    return AnalysisResults(
        significant_results=significant_results,
        info=info,
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
        print(
            "Warning: p_value_threshold is set to a value higher than the conventional alpha level of 0.05."
        )

    missing_columns = [
        variable
        for variable in variables_to_compare
        if variable not in df.columns
    ]
    if missing_columns:
        print(f"Columns not found in DataFrame: {', '.join(missing_columns)}")
        return None

    df = df.dropna(subset=variables_to_compare)
    sample_size = len(df)
    pvalue = None
    info = None
    dichotomous_flag = False
    levene_flag = False
    gaussian_flag = False
    results = None

    # Check if any variable has less than the minimum sample size
    if any(
        df[variable].count() < min_sample_size
        for variable in variables_to_compare
    ):
        info = "not enough data"
    # Check if the variables are dichotomous (e.g. boolean)
    elif all(is_dichotomous(df, variable) for variable in variables_to_compare):
        dichotomous_flag = True
        # Perform McNemar's test or its extension for more than two groups
        pvalue, results = get_mcnemar_results(
            df, variables_to_compare, p_value_threshold
        )
    elif all(is_numeric(df, variable) for variable in variables_to_compare):
        # Check if the data has Gaussian distribution, if not use non-parametric tests
        if gaussian_flag := all(
            is_gaussian(df, variable, p_value_threshold)
            for variable in variables_to_compare
        ):
            pvalue = get_repeated_measures_anova_significance(
                df, variables_to_compare
            )
            if pvalue < p_value_threshold:
                results = get_repeated_measures_anova_posthoc_results(
                    df, variables_to_compare, p_value_threshold
                )
        elif len(variables_to_compare) == 2:
            pvalue, results = get_wilcoxon_results(
                df, variables_to_compare, p_value_threshold
            )
        else:
            pvalue = get_friedman_significance(df, variables_to_compare)
            if pvalue < p_value_threshold:
                results = get_nemenyi_results(
                    df, variables_to_compare, p_value_threshold
                )
    else:
        print("All variables have to be dichotomous (e.g. boolean) or numeric")
        info = "All variables have to be dichotomous (e.g. boolean) or numeric"

    # If no results were found, return a message
    if not results or len(results.significant_results) == 0:
        if pvalue is not None:
            info = f"not significant (p={pvalue:.3f})"
    # If results were found, return them
    significant_results = results.significant_results if results else []

    return AnalysisResults(
        significant_results=significant_results,
        info=info,
        sample_size=sample_size,
        dichotomous_flag=dichotomous_flag,
        levene_flag=levene_flag,
        gaussian_flag=gaussian_flag,
    )
