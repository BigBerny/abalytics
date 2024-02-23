from itertools import combinations

# Define classes to structure the results of posthoc tests
from typing import List, Optional
from typing import Tuple

import pandas as pd
from pingouin import pairwise_gameshowell, welch_anova
from pydantic import BaseModel
from scikit_posthocs import posthoc_dunn, posthoc_nemenyi_friedman
from scipy import stats
from scipy.stats import (
    normaltest,
    chi2_contingency,
    wilcoxon,
    friedmanchisquare,
)
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.contingency_tables import mcnemar
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

    result_pretty_text: str
    p_value: float
    groups: List[GroupResult]


class PosthocResults(BaseModel):
    """Container for multiple posthoc test results."""

    significant_results: List[PosthocResult] = []

    def add_result(
        self, result_pretty_text: str, p_value: float, groups_info: List[dict]
    ):
        groups = [GroupResult(**group) for group in groups_info]
        result = PosthocResult(
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
    """Check if a variable or grouped variables in a DataFrame follow a Gaussian distribution."""
    if group_column:
        return all(
            normaltest(df[df[group_column] == group][variable])[1]
            >= p_value_threshold
            for group in df[group_column].unique()
        )
    return normaltest(df[variable])[1] >= p_value_threshold


def get_chi_square_significance(
    df: pd.DataFrame, group_column: str, variable: str
) -> float:
    """Calculate the chi-square test significance for a given variable and group."""
    contingency_table = pd.crosstab(df[group_column], df[variable])
    return chi2_contingency(contingency_table)[1]


def get_chi_square_posthoc_results(
    df: pd.DataFrame, group_column: str, variable: str, p_value_threshold: float
) -> PosthocResults:
    posthoc_results = PosthocResults()
    contingency_table = pd.crosstab(df[group_column], df[variable])
    group_combinations = [
        (group_a, group_b)
        for i, group_a in enumerate(contingency_table.index)
        for group_b in contingency_table.index[i + 1 :]
    ]
    p_values: List[float] = []

    assert set(contingency_table.columns).issubset(
        {0, 1, False, True}
    ), "Columns must be 0/1 or False/True."

    for group_a, group_b in group_combinations:
        true_or_one = 1 if 1 in contingency_table.columns else True
        count = [
            contingency_table.loc[group_a, true_or_one],
            contingency_table.loc[group_b, true_or_one],
        ]
        nobs = [
            contingency_table.loc[group_a].sum(),
            contingency_table.loc[group_b].sum(),
        ]
        rate_a = (count[0] / nobs[0]) * 100
        rate_b = (count[1] / nobs[1]) * 100

        _, p_value = proportions_ztest(count, nobs)
        p_values.append(p_value)

        groups_info = sorted(
            [
                {"name": group_a, "sample_size": nobs[0], "mean": rate_a},
                {"name": group_b, "sample_size": nobs[1], "mean": rate_b},
            ],
            key=lambda x: x["mean"],
            reverse=True,
        )

        if p_value < p_value_threshold:
            result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.1f}%) > {groups_info[1]['name']} ({groups_info[1]['mean']:.1f}%) (p={p_value:.3f})"
            posthoc_results.add_result(result_pretty_text, p_value, groups_info)

    if len(p_values) == 0:
        return posthoc_results
    # Apply Bonferroni correction
    corrected_alpha = p_value_threshold / len(p_values)
    posthoc_results.significant_results = [
        result
        for result in posthoc_results.significant_results
        if result.p_value < corrected_alpha
    ]

    return posthoc_results


def get_chi_square_posthoc_results_new(
    df: pd.DataFrame, group_column: str, variable: str, p_value_threshold: float
) -> PosthocResults:
    """Perform posthoc analysis for chi-square test results."""
    posthoc_results = PosthocResults()
    contingency_table = pd.crosstab(df[group_column], df[variable])
    group_combinations = list(combinations(contingency_table.index, 2))
    p_values = []

    for group_a, group_b in group_combinations:
        if 1 not in contingency_table.columns:
            continue
        count = [
            contingency_table.loc[group_a, 1],
            contingency_table.loc[group_b, 1],
        ]
        nobs = [
            contingency_table.loc[group_a].sum(),
            contingency_table.loc[group_b].sum(),
        ]
        rate_a, rate_b = (count[0] / nobs[0]) * 100, (count[1] / nobs[1]) * 100
        _, p_value = proportions_ztest(count, nobs)
        p_values.append(p_value)

        if p_value < p_value_threshold:
            groups_info = [
                {"name": group, "sample_size": nobs[i], "mean": rate}
                for i, (group, rate) in enumerate(
                    zip([group_a, group_b], [rate_a, rate_b])
                )
            ]
            result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.1f}%) > {groups_info[1]['name']} ({groups_info[1]['mean']:.1f}%) (p={p_value:.3f})"
            posthoc_results.add_result(result_pretty_text, p_value, groups_info)

    # Apply Bonferroni correction if there are any p-values to correct
    if p_values:
        corrected_alpha = p_value_threshold / len(p_values)
        posthoc_results.significant_results = [
            result
            for result in posthoc_results.significant_results
            if result.p_value < corrected_alpha
        ]

    return posthoc_results


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
        df[df[group_column] == group][variable]
        for group in df[group_column].unique()
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
    return welch_anova(data=df, dv=variable, between=group_column).loc[
        0, "p-unc"
    ]


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
        df.groupby(group_column)[variable]
        .agg(["mean", "median", "size"])
        .reset_index()
    )

    # Filter significant results and add to posthoc results
    significant_results = games_howell_results[
        games_howell_results["pval"] < p_value_threshold
    ]
    for _, row in significant_results.iterrows():
        group_a_stats = group_stats[group_stats[group_column] == row["A"]].iloc[
            0
        ]
        group_b_stats = group_stats[group_stats[group_column] == row["B"]].iloc[
            0
        ]
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
        result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.2f}) > {groups_info[1]['name']} ({groups_info[1]['mean']:.2f}) (p={row['pval']:.3f})"
        posthoc_results.add_result(result_pretty_text, row["pval"], groups_info)

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
        df[df[group_column] == group][variable]
        for group in df[group_column].unique()
    ]
    _, p_value = stats.f_oneway(*group_data)
    return p_value


def get_crushal_wallis_significance(
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
        df[df[group_column] == group][variable]
        for group in df[group_column].unique()
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
        df.groupby(group_column)[variable]
        .agg(["mean", "median", "size"])
        .reset_index()
    )

    significant_results = tukey._results_df[tukey._results_df["reject"]]
    for _, row in significant_results.iterrows():
        group_a, group_b = row["group1"], row["group2"]
        group_a_stats = group_stats.loc[
            group_stats[group_column] == group_a
        ].iloc[0]
        group_b_stats = group_stats.loc[
            group_stats[group_column] == group_b
        ].iloc[0]
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
        result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.2f}) > {groups_info[1]['name']} ({groups_info[1]['mean']:.2f}) (p={row['p-adj']:.3f})"
        posthoc_results.add_result(
            result_pretty_text, row["p-adj"], groups_info
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
        df.groupby(group_column)[variable]
        .agg(["mean", "median", "size"])
        .reset_index()
    )

    for group_pair, p_value in enumerate(dunn_results.values.flatten()):
        group_a, group_b = divmod(group_pair, dunn_results.shape[1])
        if p_value < p_value_threshold:
            group_a_name = group_stats.iloc[group_a][group_column]
            group_b_name = group_stats.iloc[group_b][group_column]
            group_a_stats = group_stats.iloc[group_a]
            group_b_stats = group_stats.iloc[group_b]

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
                key=lambda x: x["mean"],
                reverse=True,
            )

            result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.2f}) > {groups_info[1]['name']} ({groups_info[1]['mean']:.2f}) (p={p_value:.3f})"
            posthoc_results.add_result(result_pretty_text, p_value, groups_info)

    return posthoc_results


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
    # Iterate over all pairs of variables to compare
    for var1, var2 in combinations(variables_to_compare, 2):
        # Create a contingency table for the pair of variables
        contingency_table = pd.crosstab(df[var1], df[var2])
        # Perform McNemar's test
        result = mcnemar(contingency_table, exact=False)
        p_value = result.pvalue

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
            result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.2f}) > {groups_info[1]['name']} ({groups_info[1]['mean']:.2f}) (p={p_value:.3f})"
            # Add the result to posthoc results
            posthoc_results.add_result(result_pretty_text, p_value, groups_info)

    return posthoc_results


def get_repeated_measures_anova_significance(
    df: pd.DataFrame, variables_to_compare: List[str]
) -> float:
    """
    Calculates the significance of differences between conditions using repeated measures ANOVA.

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
    # Perform repeated measures ANOVA
    aovrm = AnovaRM(df_long, "value", "subject_id", within=["conditions"])
    anova_results = aovrm.fit()
    # Extract and return the p-value
    p_value = anova_results.anova_table["Pr > F"]["conditions"]
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
            result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.2f}) > {groups_info[1]['name']} ({groups_info[1]['mean']:.2f}) (p={p_val_corrected:.3f})"
            posthoc_results.add_result(
                result_pretty_text, p_val_corrected, groups_info
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
    groups_info.sort(key=lambda x: x["mean"], reverse=True)

    result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.2f}) > {groups_info[1]['name']} ({groups_info[1]['mean']:.2f}) (p={p_value:.3f})"
    posthoc_results.add_result(result_pretty_text, p_value, groups_info)

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

    for (i, var1), (j, var2) in combinations(
        enumerate(variables_to_compare), 2
    ):
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
            groups_info.sort(key=lambda x: x["mean"], reverse=True)

            result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.2f}) > {groups_info[1]['name']} ({groups_info[1]['mean']:.2f}) (p={posthoc_p_value:.3f})"
            posthoc_results.add_result(
                result_pretty_text, posthoc_p_value, groups_info
            )

    return posthoc_results
