import pandas as pd
from .significance_tests import (
    is_boolean,
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
)


class AnalysisResults:
    def __init__(self, significant_results=None, info=None, sample_size=None, boolean_flag=None, levene_flag=None, gaussian_flag=None):
        self.significant_results = significant_results if significant_results is not None else []
        self.info = info
        self.sample_size = sample_size
        self.boolean_flag = boolean_flag
        self.levene_flag = levene_flag
        self.gaussian_flag = gaussian_flag


def get_results(df, variable_to_analyze, group_column, p_value_threshold=0.05):
    """
    Returns a text with the results of the statistical significance tests.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data to analyze.
    variable_to_analyze (str): The column name in df that is to be analyzed for statistical significance.
    group_column (str): The name of the column in df that contains the grouping variable.
    p_value_threshold (float, optional): The threshold for determining statistical significance. Defaults to 0.05.

    Returns:
    AnalysisResults: An instance of the AnalysisResults class containing the following attributes:
        - significant_results: A list of results of the statistical significance tests.
        - info: A string containing information about the data if no results were found.
        - sample_size: The sample size of the data.
        - boolean_flag: A boolean flag indicating if the variable is boolean.
        - levene_flag: A boolean flag indicating if Levene's test for homogeneity of variances is significant.
        - gaussian_flag: A boolean flag indicating if the data has a Gaussian distribution.
    """
    if p_value_threshold > 0.05:
        print(
            "Warning: p_value_threshold is set to a value higher than the conventional alpha level of 0.05."
        )

    df = df.dropna(subset=[variable_to_analyze])
    sample_size = len(df)
    pvalue = None
    info = None
    boolean_flag = False
    levene_flag = False
    gaussian_flag = False
    # Check if there is enough data
    if len(df) < 25 * df[group_column].nunique():
        info = "not enough data"
    # Check if the variable is dichotomous (e.g. boolean)
    elif boolean_flag := is_boolean(df, variable_to_analyze):
        # Perform chi-square test
        pvalue = get_chi_square_significance(df, group_column, variable_to_analyze)
        if pvalue < p_value_threshold:
            results = get_chi_square_posthoc_results(
                df, group_column, variable_to_analyze, p_value_threshold
            )
    elif is_numeric(df, variable_to_analyze):
        # Check homogeneity of variances. If significant, use non-parametric tests
        if levene_flag := is_levene_significant(
            df, group_column, variable_to_analyze, p_value_threshold
        ):
            pvalue = get_welch_anova_significance(df, group_column, variable_to_analyze)
            if pvalue < p_value_threshold:
                results = get_games_howell_posthoc_results(
                    df, group_column, variable_to_analyze, p_value_threshold
                )
        # Check if the data has Gaussian distribution, if not use non-parametric tests
        elif gaussian_flag := is_gaussian(
            df, group_column, variable_to_analyze, p_value_threshold
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
        print("Variable has to be boolean or numeric")

    # If no results were found, return a message
    if not 'results' in locals() or len(results.significant_results) == 0:
        if pvalue is not None:
            info = f"not significant (p={pvalue:.3f})"
    # If results were found, return them
    if 'results' in locals():
        significant_results = results.significant_results
    else:
        significant_results = []

    return AnalysisResults(
        significant_results, info, sample_size, boolean_flag, levene_flag, gaussian_flag
    )


def get_results_pretty_text_header(identifiers=None):
    """
    Returns a text with the results of the statistical significance tests.

    Parameters:
    identifiers (list of str, optional): A list of strings that are to be used for identifying the data. Defaults to [].

    Returns:
    str: A formatted text string with the header of the pretty text table.
    """
    identifier_string = ""
    identifier_string_empty = ""
    if identifiers:
        identifier_string = "    ".join(identifiers)
        identifier_string += "    "
        identifier_string_empty = "    ".join([""] * len(identifiers))
        identifier_string_empty += "    "
        output_string = f"{'Identifier':<{len(identifier_string)}}{'n':>10}    {'Boolean':<7}    {'Levene':<6}    {'Gaussian':<8}    {'Result'}"
    else:
        output_string = f"{'n':>10}    {'Boolean':<7}    {'Levene':<6}    {'Gaussian':<8}    {'Result'}"

    return output_string


def get_results_pretty_text(
    df,
    variable_to_analyze,
    group_column,
    identifiers=None,
    p_value_threshold=0.05,
    header=True,
):
    """
    Returns a text with the results of the statistical significance tests.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data to analyze.
    variable_to_analyze (str): The column name in df that is to be analyzed for statistical significance.
    group_column (str): The name of the column in df that contains the grouping variable.
    identifiers (list of str, optional): A list of strings that are to be used for identifying the data. Defaults to [].
    p_value_threshold (float, optional): The threshold for determining statistical significance. Defaults to 0.05.

    Returns:
    str: A formatted text string with the results of the statistical significance tests.
    """

    analysis_results = get_results(
        df, variable_to_analyze, group_column, p_value_threshold
    )
    significant_results = analysis_results.significant_results
    info = analysis_results.info
    sample_size = analysis_results.sample_size
    boolean_flag = analysis_results.boolean_flag
    levene_flag = analysis_results.levene_flag
    gaussian_flag = analysis_results.gaussian_flag

    identifier_string = ""
    identifier_string_empty = ""
    if identifiers:
        identifier_string = "    ".join(identifiers)
        identifier_string += "    "
        identifier_string_empty = "    ".join([""] * len(identifiers))
        identifier_string_empty += "    "

    output_string = ""
    if header:
        output_string += get_results_pretty_text_header(identifiers)
        output_string += "\n"

    boolean_output = "X" if boolean_flag else ""
    levene_output = "X" if levene_flag else ""
    gaussian_output = "X" if gaussian_flag else ""

    output_results = []
    if len(significant_results) > 0:
        for i, result in enumerate(significant_results):
            if i == 0:
                output_results.append(
                    f"{identifier_string}{sample_size:>10}    {boolean_output:<7}    {levene_output:<6}    {gaussian_output:<8}    {result.result_pretty}"
                )
            else:
                output_results.append(
                    f"{identifier_string}{sample_size:>10}    {boolean_output:<7}    {levene_output:<6}    {gaussian_output:<8}    {result.result_pretty}"
                )
    else:
        output_results.append(
            f"{identifier_string}{sample_size:>10}    {boolean_output:<7}    {levene_output:<6}    {gaussian_output:<8}    {info}"
        )
    output_string += "\n".join(output_results)
    return output_string
