from itertools import combinations

# Define classes to structure the results of posthoc tests
from typing import List, Optional
from typing import Tuple

import pandas as pd
from pingouin import pairwise_gameshowell, rm_anova, welch_anova
from pydantic import BaseModel
from scikit_posthocs import posthoc_dunn, posthoc_nemenyi_friedman
from scipy import stats
from scipy.stats import (
    normaltest,
    chi2_contingency,
    fisher_exact,
    wilcoxon,
    friedmanchisquare,
)
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest


class GroupResult(BaseModel):
    """Represents the result for a single group within a posthoc test."""

    name: str
    sample_size: int
    mean: float
    median: Optional[float] = None


class PosthocResult(BaseModel):
    """Represents a single posthoc test result."""

    analysis_method: str
    result_pretty_text: str
    p_value: float
    groups: List[GroupResult]


class PosthocResults(BaseModel):
    """Container for multiple posthoc test results."""

    significant_results: List[PosthocResult] = []

    def add_result(
        self,
        analysis_method: str,
        result_pretty_text: str,
        p_value: float,
        groups_info: List[dict],
    ):
        groups = [GroupResult(**group) for group in groups_info]
        result = PosthocResult(
            analysis_method=analysis_method,
            result_pretty_text=result_pretty_text,
            p_value=p_value,
            groups=groups,
        )
        self.significant_results.append(result)


# Define functions to perform various statistical tests and analyses
def is_dichotomous(df: pd.DataFrame, variable: str) -> bool:
    """Check if a variable in a DataFrame is dichotomous."""
    return df[variable].nunique() == 2


def is_numeric(df: pd.DataFrame, variable: str) -> bool:
    """Check if a variable in a DataFrame is numeric."""
    return pd.api.types.is_numeric_dtype(df[variable])


def is_gaussian(
    df: pd.DataFrame,
    variable: str,
    p_value_threshold: float,
    group_column: str = None,
) -> bool:
    """Check if a variable or grouped variables in a DataFrame follow a Gaussian distribution.
    Returns False for samples with n < 20 as a conservative policy choice:
    normality tests are unreliable with small samples, so we default to non-parametric."""
    MIN_NORMALTEST_N = 20

    if group_column:
        for group in df[group_column].unique():
            group_data = df[df[group_column] == group][variable].dropna()
            if len(group_data) < MIN_NORMALTEST_N:
                return False
            if normaltest(group_data)[1] < p_value_threshold:
                return False
        return True

    data = df[variable].dropna()
    if len(data) < MIN_NORMALTEST_N:
        return False
    return normaltest(data)[1] >= p_value_threshold


def get_chi_square_significance(
    df: pd.DataFrame, group_column: str, variable: str
) -> float:
    """Calculate the chi-square test significance for a given variable and group.
    Falls back to Fisher's exact test for 2x2 tables with expected frequencies < 5."""
    contingency_table = pd.crosstab(df[group_column], df[variable])
    chi2, p_value, dof, expected_freq = chi2_contingency(contingency_table)

    if not (expected_freq >= 5).all():
        if contingency_table.shape == (2, 2):
            _, p_value = fisher_exact(contingency_table)
        else:
            import logging

            logging.warning(
                "Chi-square expected frequencies < 5 in some cells. "
                "Results may be unreliable. Consider collecting more data."
            )

    return p_value


def get_chi_square_posthoc_results(
    df: pd.DataFrame, group_column: str, variable: str, p_value_threshold: float
) -> PosthocResults:
    """Perform posthoc analysis for chi-square test results."""
    posthoc_results = PosthocResults()
    contingency_table = pd.crosstab(df[group_column], df[variable])
    group_combinations = list(combinations(contingency_table.index, 2))
    p_values = []

    # Determine the "positive" column dynamically
    truthy_values = [True, 1, 1.0, "true", "True", "yes", "Yes", "Y", "y"]
    positive_col = None
    for col in contingency_table.columns:
        if col in truthy_values:
            positive_col = col
            break
    if positive_col is None:
        # Fallback: use the column with the higher overall count
        positive_col = contingency_table.sum().idxmax()

    for group_a, group_b in group_combinations:
        count = [
            contingency_table.loc[group_a, positive_col],
            contingency_table.loc[group_b, positive_col],
        ]
        nobs = [
            contingency_table.loc[group_a].sum(),
            contingency_table.loc[group_b].sum(),
        ]
        rate_a, rate_b = (count[0] / nobs[0]) * 100, (count[1] / nobs[1]) * 100
        _, p_value = proportions_ztest(count, nobs)
        p_values.append(p_value)

        if p_value < p_value_threshold:
            groups_info = sorted(
                [
                    {"name": group_a, "sample_size": nobs[0], "mean": rate_a},
                    {"name": group_b, "sample_size": nobs[1], "mean": rate_b},
                ],
                key=lambda x: x["mean"],
                reverse=True,
            )
            result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.1f}%) > {groups_info[1]['name']} ({groups_info[1]['mean']:.1f}%)"
            posthoc_results.add_result(
                "Chi-square", result_pretty_text, p_value, groups_info
            )

    # Apply Bonferroni correction if there are any p-values to correct
    if p_values:
        corrected_alpha = p_value_threshold / len(p_values)
        posthoc_results.significant_results = [
            result
            for result in posthoc_results.significant_results
            if result.p_value < corrected_alpha
        ]

    return posthoc_results


def are_differences_gaussian(
    df: pd.DataFrame, variables: List[str], p_value_threshold: float
) -> bool:
    """Check if the pairwise differences between variables follow a Gaussian distribution.
    This is the correct assumption to test for repeated measures ANOVA."""
    MIN_NORMALTEST_N = 20
    for v1, v2 in combinations(variables, 2):
        diff = df[v1] - df[v2]
        diff = diff.dropna()
        if len(diff) < MIN_NORMALTEST_N:
            return False
        _, p = normaltest(diff)
        if p < p_value_threshold:
            return False
    return True


def is_levene_significant(
    df: pd.DataFrame, group_column: str, variable: str, p_value_threshold: float
) -> bool:
    """
    Determines if the variances across groups are significantly different using Levene's test.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - group_column (str): The column name representing the groups.
    - variable (str): The column name of the variable to test for homogeneity of variances.
    - p_value_threshold (float): The significance level.

    Returns:
    - bool: True if variances are significantly different, False otherwise.
    """
    # Extract data for each group
    data_groups = [
        df[df[group_column] == group][variable] for group in df[group_column].unique()
    ]
    # Perform Levene's test
    _, p_value = stats.levene(*data_groups)
    # Determine significance
    return p_value < p_value_threshold


def get_welch_anova_significance(
    df: pd.DataFrame, group_column: str, variable: str
) -> float:
    """
    Calculates the significance of differences between group means using Welch's ANOVA.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - group_column (str): The column name representing the groups.
    - variable (str): The column name of the variable to test.

    Returns:
    - float: The p-value from Welch's ANOVA.
    """
    # Perform Welch's ANOVA and return the p-value
    return welch_anova(data=df, dv=variable, between=group_column).loc[0, "p-unc"]


def get_games_howell_posthoc_results(
    df: pd.DataFrame, group_column: str, variable: str, p_value_threshold: float
) -> PosthocResults:
    """
    Performs the Games-Howell posthoc test to identify differences between pairs of groups.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - group_column (str): The column name representing the groups.
    - variable (str): The column name of the variable to test.
    - p_value_threshold (float): The significance level for identifying significant differences.

    Returns:
    - PosthocResults: An object containing significant posthoc test results.
    """
    posthoc_results = PosthocResults()
    # Perform Games-Howell posthoc test
    games_howell_results = pairwise_gameshowell(
        data=df, dv=variable, between=group_column
    )
    # Aggregate group statistics
    group_stats = (
        df.groupby(group_column)[variable].agg(["mean", "median", "size"]).reset_index()
    )

    # Filter significant results and add to posthoc results
    significant_results = games_howell_results[
        games_howell_results["pval"] < p_value_threshold
    ]
    for _, row in significant_results.iterrows():
        group_a_stats = group_stats[group_stats[group_column] == row["A"]].iloc[0]
        group_b_stats = group_stats[group_stats[group_column] == row["B"]].iloc[0]
        groups_info = sorted(
            [
                {
                    "name": row["A"],
                    "sample_size": group_a_stats["size"],
                    "mean": group_a_stats["mean"],
                    "median": group_a_stats["median"],
                },
                {
                    "name": row["B"],
                    "sample_size": group_b_stats["size"],
                    "mean": group_b_stats["mean"],
                    "median": group_b_stats["median"],
                },
            ],
            key=lambda x: x["mean"],
            reverse=True,
        )
        result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.3f}) > {groups_info[1]['name']} ({groups_info[1]['mean']:.3f})"
        posthoc_results.add_result(
            "Games-Howell", result_pretty_text, row["pval"], groups_info
        )

    return posthoc_results


def get_oneway_anova_significance(
    df: pd.DataFrame, group_column: str, variable: str
) -> float:
    """
    Calculates the significance of differences between group means using one-way ANOVA.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - group_column (str): The column name representing the groups.
    - variable (str): The column name of the variable to test.

    Returns:
    - float: The p-value from the one-way ANOVA test.
    """
    group_data = [
        df[df[group_column] == group][variable] for group in df[group_column].unique()
    ]
    _, p_value = stats.f_oneway(*group_data)
    return p_value


def get_kruskal_wallis_significance(
    df: pd.DataFrame, group_column: str, variable: str
) -> float:
    """
    Calculates the significance of differences between group medians using Kruskal-Wallis H-test.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - group_column (str): The column name representing the groups.
    - variable (str): The column name of the variable to test.

    Returns:
    - float: The p-value from the Kruskal-Wallis H-test.
    """
    group_data = [
        df[df[group_column] == group][variable] for group in df[group_column].unique()
    ]
    _, p_value = stats.kruskal(*group_data)
    return p_value


def get_tukeyhsd_posthoc_results(
    df: pd.DataFrame, group_column: str, variable: str, p_value_threshold: float
) -> PosthocResults:
    """
    Performs Tukey's Honest Significant Difference (HSD) test to identify differences between pairs of group means.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - group_column (str): The column name representing the groups.
    - variable (str): The column name of the variable to test.
    - p_value_threshold (float): The significance level for identifying significant differences.

    Returns:
    - PosthocResults: An object containing significant posthoc test results.
    """
    posthoc_results = PosthocResults()
    tukey = pairwise_tukeyhsd(
        endog=df[variable], groups=df[group_column], alpha=p_value_threshold
    )
    group_stats = (
        df.groupby(group_column)[variable].agg(["mean", "median", "size"]).reset_index()
    )

    pairs = list(combinations(tukey.groupsunique, 2))
    for i, (group_a, group_b) in enumerate(pairs):
        if not tukey.reject[i]:
            continue
        group_a_stats = group_stats.loc[group_stats[group_column] == group_a].iloc[0]
        group_b_stats = group_stats.loc[group_stats[group_column] == group_b].iloc[0]
        groups_info = sorted(
            [
                {
                    "name": group_a,
                    "sample_size": group_a_stats["size"],
                    "mean": group_a_stats["mean"],
                    "median": group_a_stats["median"],
                },
                {
                    "name": group_b,
                    "sample_size": group_b_stats["size"],
                    "mean": group_b_stats["mean"],
                    "median": group_b_stats["median"],
                },
            ],
            key=lambda x: x["mean"],
            reverse=True,
        )
        result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.3f}) > {groups_info[1]['name']} ({groups_info[1]['mean']:.3f})"
        posthoc_results.add_result(
            "Tukey HSD", result_pretty_text, tukey.pvalues[i], groups_info
        )

    return posthoc_results


def get_dunn_posthoc_results(
    df: pd.DataFrame, group_column: str, variable: str, p_value_threshold: float
) -> PosthocResults:
    """
    Performs Dunn's posthoc test following a Kruskal-Wallis H-test to identify differences between pairs of group medians.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - group_column (str): The column name representing the groups.
    - variable (str): The column name of the variable to test.
    - p_value_threshold (float): The significance level for identifying significant differences.

    Returns:
    - PosthocResults: An object containing significant posthoc test results.
    """
    posthoc_results = PosthocResults()
    dunn_results = posthoc_dunn(
        df, val_col=variable, group_col=group_column, p_adjust="holm"
    )
    group_stats = (
        df.groupby(group_column)[variable].agg(["mean", "median", "size"]).reset_index()
    )

    # Iterate only over the upper triangle of the matrix, excluding the diagonal
    for i in range(dunn_results.shape[0]):
        for j in range(i + 1, dunn_results.shape[1]):
            p_value = dunn_results.iloc[i, j]
            if p_value < p_value_threshold:
                group_a_name = group_stats.iloc[i][group_column]
                group_b_name = group_stats.iloc[j][group_column]
                group_a_stats = group_stats.iloc[i]
                group_b_stats = group_stats.iloc[j]

                groups_info = sorted(
                    [
                        {
                            "name": group_a_name,
                            "sample_size": group_a_stats["size"],
                            "mean": group_a_stats["mean"],
                            "median": group_a_stats["median"],
                        },
                        {
                            "name": group_b_name,
                            "sample_size": group_b_stats["size"],
                            "mean": group_b_stats["mean"],
                            "median": group_b_stats["median"],
                        },
                    ],
                    key=lambda x: x["median"],
                    reverse=True,
                )

                result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['median']:.3f}) > {groups_info[1]['name']} ({groups_info[1]['median']:.3f})"
                posthoc_results.add_result(
                    "Dunn", result_pretty_text, p_value, groups_info
                )

    return posthoc_results


def get_cochrans_q_significance(
    df: pd.DataFrame, variables_to_compare: List[str]
) -> float:
    """Performs Cochran's Q test as an omnibus test for >2 dependent dichotomous variables.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - variables_to_compare (List[str]): The column names of dichotomous variables.

    Returns:
    - float: The p-value from Cochran's Q test.
    """
    data = df[variables_to_compare].values.astype(int)
    result = cochrans_q(data)
    return result.pvalue


def get_mcnemar_results(
    df: pd.DataFrame, variables_to_compare: List[str], p_value_threshold: float
) -> Tuple[float, PosthocResults]:
    """
    Performs McNemar's test or its extension for more than two groups and returns the p-value and posthoc results.
    This function is designed to work with binary categorical data to compare the proportions of categories between two related samples.

    Parameters:
    - df (pandas.DataFrame): The dataframe containing the data to analyze.
    - variables_to_compare (List[str]): The column names in df that are to be analyzed for statistical significance.
    - p_value_threshold (float): The threshold for determining statistical significance.

    Returns:
    - Tuple[float, PosthocResults]: A tuple containing the p-value of the McNemar's test and the posthoc results.
    """
    posthoc_results = PosthocResults()
    p_values = []

    # Iterate over all pairs of variables to compare
    for var1, var2 in combinations(variables_to_compare, 2):
        # Create a contingency table for the pair of variables
        contingency_table = pd.crosstab(df[var1], df[var2])
        # Use exact test when discordant cell counts are small
        b = contingency_table.iloc[0, 1]
        c = contingency_table.iloc[1, 0]
        use_exact = (b + c) < 25
        result = mcnemar(contingency_table, exact=use_exact)
        p_value = result.pvalue
        p_values.append(p_value)

        # If the result is significant, add it to the posthoc results
        if p_value < p_value_threshold:
            # Extract statistics for each variable
            groups_info = [
                {
                    "name": variable,
                    "sample_size": df[variable].size,
                    "mean": df[variable].mean(),
                    "median": df[variable].median(),
                }
                for variable in [var1, var2]
            ]
            # Sort groups by mean for consistent presentation
            groups_info.sort(key=lambda x: x["mean"], reverse=True)
            # Format the result text
            result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.3f}) > {groups_info[1]['name']} ({groups_info[1]['mean']:.3f})"
            # Add the result to posthoc results
            posthoc_results.add_result(
                "McNemar", result_pretty_text, p_value, groups_info
            )

    # Apply Bonferroni correction for multiple pairwise comparisons
    if len(p_values) > 1:
        corrected_alpha = p_value_threshold / len(p_values)
        posthoc_results.significant_results = [
            r for r in posthoc_results.significant_results if r.p_value < corrected_alpha
        ]

    omnibus_pvalue = min(p_values) if p_values else 1.0
    return omnibus_pvalue, posthoc_results


def get_repeated_measures_anova_significance(
    df: pd.DataFrame, variables_to_compare: List[str]
) -> float:
    """
    Calculates the significance of differences between conditions using repeated measures ANOVA.
    Applies Greenhouse-Geisser correction when sphericity is violated.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - variables_to_compare (List[str]): The column names representing the conditions to compare.

    Returns:
    - float: The p-value from the repeated measures ANOVA test.
    """
    # Convert the dataframe to long format for ANOVA
    df_long = pd.melt(
        df.reset_index(),
        id_vars=["index"],
        value_vars=variables_to_compare,
        var_name="conditions",
        value_name="value",
    )
    df_long.rename(columns={"index": "subject_id"}, inplace=True)

    # Use pingouin's rm_anova which handles sphericity testing and GG correction
    aov = rm_anova(
        data=df_long,
        dv="value",
        within="conditions",
        subject="subject_id",
        correction=True,
    )

    # Use GG-corrected p-value if sphericity is violated, otherwise uncorrected
    if len(variables_to_compare) > 2 and not aov["sphericity"].iloc[0]:
        p_value = aov["p-GG-corr"].iloc[0]
    else:
        p_value = aov["p-unc"].iloc[0]

    return p_value


def get_repeated_measures_anova_posthoc_results(
    df: pd.DataFrame, variables_to_compare: List[str], p_value_threshold: float
) -> PosthocResults:
    """
    Performs posthoc pairwise comparisons after repeated measures ANOVA to identify significant differences.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - variables_to_compare (List[str]): The column names representing the conditions to compare.
    - p_value_threshold (float): The significance level for identifying significant differences.

    Returns:
    - PosthocResults: An object containing significant posthoc test results.
    """
    posthoc_results = PosthocResults()
    # Generate all pairwise comparisons
    pairwise_comparisons = [
        (var1, var2, *stats.ttest_rel(df[var1], df[var2]))
        for var1, var2 in combinations(variables_to_compare, 2)
    ]

    # Adjust p-values for multiple comparisons using Bonferroni correction
    _, pvals_corrected, _, _ = multipletests(
        [p_val for _, _, _, p_val in pairwise_comparisons],
        alpha=p_value_threshold,
        method="bonferroni",
    )

    # Filter and add significant results to posthoc results
    for (var1, var2, _, p_val), p_val_corrected in zip(
        pairwise_comparisons, pvals_corrected
    ):
        if p_val_corrected < p_value_threshold:
            groups_info = [
                {
                    "name": var,
                    "sample_size": df[var].size,
                    "mean": df[var].mean(),
                    "median": df[var].median(),
                }
                for var in [var1, var2]
            ]
            # Sort groups by mean for consistent presentation
            groups_info.sort(key=lambda x: x["mean"], reverse=True)
            result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.3f}) > {groups_info[1]['name']} ({groups_info[1]['mean']:.3f})"
            posthoc_results.add_result(
                "Repeated Measures ANOVA",
                result_pretty_text,
                p_val_corrected,
                groups_info,
            )

    return posthoc_results


def get_wilcoxon_results(
    df: pd.DataFrame, variables_to_compare: List[str], p_value_threshold: float
) -> Tuple[float, PosthocResults]:
    """
    Performs the Wilcoxon signed-rank test for two paired samples and returns the p-value and posthoc results.
    This function is designed for paired comparisons to test the null hypothesis that two related paired samples
    come from the same distribution.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data to analyze.
    - variables_to_compare (List[str]): A list containing exactly two column names in df that are to be analyzed for statistical significance.
    - p_value_threshold (float): The threshold for determining statistical significance.

    Returns:
    - Tuple[float, PosthocResults]: A tuple containing the p-value of the Wilcoxon signed-rank test and the posthoc results.
    """
    assert (
        len(variables_to_compare) == 2
    ), "Exactly two variables must be compared for the Wilcoxon test."

    posthoc_results = PosthocResults()
    var1, var2 = variables_to_compare
    data1, data2 = df[var1], df[var2]

    _, p_value = wilcoxon(data1, data2)

    groups_info = [
        {
            "name": var,
            "sample_size": df[var].size,
            "mean": df[var].mean(),
            "median": df[var].median(),
        }
        for var in variables_to_compare
    ]
    groups_info.sort(key=lambda x: x["median"], reverse=True)

    if p_value < p_value_threshold:
        result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['median']:.3f}) > {groups_info[1]['name']} ({groups_info[1]['median']:.3f})"
        posthoc_results.add_result("Wilcoxon", result_pretty_text, p_value, groups_info)

    return p_value, posthoc_results


def get_friedman_significance(
    df: pd.DataFrame, variables_to_compare: List[str]
) -> float:
    """
    Performs the Friedman test for repeated measures on ranks and returns the p-value.
    This non-parametric test is used to detect differences in treatments across multiple test attempts.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data to analyze.
    - variables_to_compare (List[str]): The column names in df that are to be analyzed for statistical significance.

    Returns:
    - float: The p-value of the Friedman test, indicating the probability of observing the given result by chance.
    """
    data = [df[variable].values for variable in variables_to_compare]
    _, p_value = friedmanchisquare(*data)
    return p_value


def get_nemenyi_results(
    df: pd.DataFrame, variables_to_compare: List[str], p_value_threshold: float
) -> PosthocResults:
    """
    Performs the Nemenyi posthoc test suitable after the Friedman test to identify differences between pairs of group medians.
    This test is used when the Friedman test has indicated significant differences, to find out which groups differ.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data to analyze.
    - variables_to_compare (List[str]): The column names in df that are to be analyzed for statistical significance.
    - p_value_threshold (float): The threshold for determining statistical significance.

    Returns:
    - PosthocResults: An object containing significant posthoc test results.
    """
    posthoc_results = PosthocResults()
    posthoc = posthoc_nemenyi_friedman(df[variables_to_compare])

    for (i, var1), (j, var2) in combinations(enumerate(variables_to_compare), 2):
        posthoc_p_value = posthoc.iloc[i, j]
        if posthoc_p_value < p_value_threshold:
            groups_info = [
                {
                    "name": var,
                    "sample_size": df[var].size,
                    "mean": df[var].mean(),
                    "median": df[var].median(),
                }
                for var in [var1, var2]
            ]
            groups_info.sort(key=lambda x: x["median"], reverse=True)

            result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['median']:.3f}) > {groups_info[1]['name']} ({groups_info[1]['median']:.3f})"
            posthoc_results.add_result(
                "Nemenyi", result_pretty_text, posthoc_p_value, groups_info
            )

    return posthoc_results
