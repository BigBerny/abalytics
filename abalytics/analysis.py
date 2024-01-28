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


def get_results(df, variable_to_analyze, group_column, p_value_threshold=0.05):
    """
    Returns a text with the results of the statistical significance tests.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data to analyze.
    variable_to_analyze (str): The column name in df that are to be analyzed for statistical significance.
    group_column (str): The name of the column in df that contains the grouping variable.
    p_value_threshold (float, optional): The threshold for determining statistical significance. Defaults to 0.05.

    Returns:
    tuple: A tuple containing the results of the statistical significance tests for each variable, and flags indicating boolean, Levene's test, and Gaussian distribution status.
    """
    if p_value_threshold > 0.05:
        print(
            "Warning: p_value_threshold is set to a value higher than the conventional alpha level of 0.05."
        )

    results = []
    boolean_flag = False
    levene_flag = False
    gaussian_flag = False
    # Check if there is enough data
    if len(df) < 20 * df[group_column].nunique():
        results.append("not enough data")
    # Check if the variable is dichotomous (e.g. boolean)
    elif boolean_flag := is_boolean(df, variable_to_analyze):
        # Perform chi-square test
        pvalue = get_chi_square_significance(df, group_column, variable_to_analyze)
        if pvalue < p_value_threshold:
            results = get_chi_square_posthoc_results(
                df, group_column, variable_to_analyze
            )
    elif is_numeric(df, variable_to_analyze):
        # Check homogeneity of variances. If significant, use non-parametric tests
        if levene_flag := is_levene_significant(df, group_column, variable_to_analyze):
            pvalue = get_welch_anova_significance(df, group_column, variable_to_analyze)
            if pvalue < p_value_threshold:
                results = get_games_howell_posthoc_results(
                    df, group_column, variable_to_analyze
                )
        # Check if the data has Gaussian distribution, if not use non-parametric tests
        elif gaussian_flag := is_gaussian(df, group_column, variable_to_analyze):
            pvalue = get_oneway_anova_significance(
                df, group_column, variable_to_analyze
            )
            if pvalue < p_value_threshold:
                results = get_tukeyhsd_posthoc_results(
                    df, group_column, variable_to_analyze
                )
        else:
            pvalue = get_crushal_wallis_significance(
                df, group_column, variable_to_analyze
            )
            if pvalue < p_value_threshold:
                results = get_dunn_posthoc_results(
                    df, group_column, variable_to_analyze
                )
    else:
        results.append("Variable has to be boolean or numeric")

    if len(results) == 0:
        results.append(f"not significant (p={pvalue:.3f})")

    return results, boolean_flag, levene_flag, gaussian_flag


def get_results_pretty_text_header(identifiers=None):
    """
    Returns a text with the results of the statistical significance tests.

    Parameters:
    identifiers (list of str, optional): A list of strings that are to be used for identifying the data. Defaults to [].

    Returns:
    str: A formatted text string with the results of the statistical significance tests for each variable.
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
    variable_to_analyze (str): The column name in df that are to be analyzed for statistical significance.
    group_column (str): The name of the column in df that contains the grouping variable.
    identifiers (list of str, optional): A list of strings that are to be used for identifying the data. Defaults to [].
    p_value_threshold (float, optional): The threshold for determining statistical significance. Defaults to 0.05.

    Returns:
    str: A formatted text string with the results of the statistical significance tests for each variable.
    """

    results, boolean_flag, levene_flag, gaussian_flag = get_results(
        df, variable_to_analyze, group_column, p_value_threshold
    )

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
    for i, result in enumerate(results):
        if i == 0:
            output_results.append(f"{identifier_string}{len(df):>10}    {boolean_output:<7}    {levene_output:<6}    {gaussian_output:<8}    {result}")
        else:
            output_results.append(f"{identifier_string}{len(df):>10}    {boolean_output:<7}    {levene_output:<6}    {gaussian_output:<8}    {result}")
    output_string += "\n".join(output_results)
    return output_string
