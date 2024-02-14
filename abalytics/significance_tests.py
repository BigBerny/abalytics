import pandas as pd
from scipy import stats
from scipy.stats import normaltest, chi2_contingency, wilcoxon, friedmanchisquare
from scikit_posthocs import posthoc_dunn, posthoc_nemenyi_friedman
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest
from pingouin import pairwise_gameshowell, welch_anova
from itertools import combinations
from typing import List, Dict, Union, Any, Tuple
from statsmodels.sandbox.stats.multicomp import multipletests
from collections import namedtuple


class PosthocResult:
    def __init__(
        self, result_pretty_text: str, p_value: float, groups: List["GroupResult"]
    ):
        self.result_pretty_text: str = result_pretty_text
        self.p_value: float = p_value
        self.groups: List[GroupResult] = groups


class GroupResult:
    def __init__(
        self,
        name: str,
        sample_size: int,
        mean: float,
        median: Union[float, None] = None,
    ):
        self.name: str = name
        self.sample_size: int = sample_size
        self.mean: float = mean
        self.median: Union[float, None] = median


class PosthocResults:
    def __init__(self):
        self.significant_results: List[PosthocResult] = []

    def add_result(
        self, result_pretty_text: str, p_value: float, groups_info: List[Dict[str, Any]]
    ):
        groups: List[GroupResult] = [GroupResult(**group) for group in groups_info]
        result: PosthocResult = PosthocResult(result_pretty_text, p_value, groups)
        self.significant_results.append(result)


def is_dichotomous(df: pd.DataFrame, variable: str) -> bool:
    # Check if the variable has only two unique values
    return len(df[variable].unique()) == 2


def is_numeric(df: pd.DataFrame, variable: str) -> bool:
    # Check if the variable is of numeric type
    return df[variable].dtype in [float, int]


def is_gaussian(
    df: pd.DataFrame, variable: str, p_value_threshold: float, group_column: str = None, 
) -> bool:
    if group_column is None:
        # Test for normality using the specified p-value threshold
        _, p_value = normaltest(df[variable])
        return p_value >= p_value_threshold
    else:
        # Iterate over each unique group in the specified column
        for group in df[group_column].unique():
            # Subset the dataframe for the current group and test for normality
            group_data = df[df[group_column] == group][variable]
            _, p_value = normaltest(group_data)

            # If the p-value is less than the threshold, the data is not Gaussian
            if p_value < p_value_threshold:
                return False  # Early return if any group is not Gaussian

    # If all groups passed the normality test, return True
    return True


def get_chi_square_significance(
    df: pd.DataFrame, group_column: str, variable: str
) -> float:
    contingency_table = pd.crosstab(df[group_column], df[variable])
    _, pvalue, _, _ = chi2_contingency(contingency_table)
    return pvalue


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

    for group_a, group_b in group_combinations:
        if 1 not in contingency_table.columns:
            continue
        count = [contingency_table.loc[group_a, 1], contingency_table.loc[group_b, 1]]
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


def is_levene_significant(
    df: pd.DataFrame, group_column: str, variable: str, p_value_threshold: float
) -> bool:
    # Prepare data for homogeneity of variances test
    groups = df[group_column].unique()
    data_groups = [df[df[group_column] == group][variable] for group in groups]

    # Perform Levene's test for equal variances
    _, p_levene = stats.levene(*data_groups)

    # Check if variances are equal
    return p_levene < p_value_threshold


def get_welch_anova_significance(
    df: pd.DataFrame, group_column: str, variable: str
) -> float:
    # Perform Welch's ANOVA
    welch_anova_results = welch_anova(data=df, dv=variable, between=group_column)
    # Check if the p-value from Welch's ANOVA is significant
    return welch_anova_results.loc[0, "p-unc"]


def get_games_howell_posthoc_results(
    df: pd.DataFrame, group_column: str, variable: str, p_value_threshold: float
) -> PosthocResults:
    posthoc_results = PosthocResults()
    games_howell_results = pairwise_gameshowell(
        data=df, dv=variable, between=group_column
    )
    group_means = df.groupby(group_column)[variable].mean()
    group_medians = df.groupby(group_column)[variable].median()
    group_sizes = df.groupby(group_column).size()

    for _, row in games_howell_results.iterrows():
        p_value = row["pval"]
        if p_value < p_value_threshold:
            group_a, group_b = row["A"], row["B"]
            mean_a, mean_b = group_means[group_a], group_means[group_b]
            median_a, median_b = group_medians[group_a], group_medians[group_b]
            size_a, size_b = group_sizes[group_a], group_sizes[group_b]

            # Order groups so the one with the higher mean is first
            groups_info = sorted(
                [
                    {
                        "name": group_a,
                        "sample_size": size_a,
                        "mean": mean_a,
                        "median": median_a,
                    },
                    {
                        "name": group_b,
                        "sample_size": size_b,
                        "mean": mean_b,
                        "median": median_b,
                    },
                ],
                key=lambda x: x["mean"],
                reverse=True,
            )

            result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.2f}) > {groups_info[1]['name']} ({groups_info[1]['mean']:.2f}) (p={p_value:.3f})"
            posthoc_results.add_result(result_pretty_text, p_value, groups_info)

    return posthoc_results


def get_oneway_anova_significance(
    df: pd.DataFrame, group_column: str, variable: str
) -> float:
    group_names = df[group_column].unique()
    # Perform one-way ANOVA
    group_data = [df[df[group_column] == group][variable] for group in group_names]
    _, pvalue = stats.f_oneway(*group_data)
    return pvalue


def get_crushal_wallis_significance(
    df: pd.DataFrame, group_column: str, variable: str
) -> float:
    group_names = df[group_column].unique()
    # Perform Kruskal-Wallis test
    group_data = [df[df[group_column] == group][variable] for group in group_names]
    _, pvalue = stats.kruskal(*group_data)
    return pvalue


def get_tukeyhsd_posthoc_results(
    df: pd.DataFrame, group_column: str, variable: str, p_value_threshold: float
) -> PosthocResults:
    posthoc_results = PosthocResults()
    # Perform Tukey's HSD test
    tukey = pairwise_tukeyhsd(
        endog=df[variable],
        groups=df[group_column],
        alpha=p_value_threshold,
    )

    # Calculate means and medians for each group
    group_means = df.groupby(group_column)[variable].mean()
    group_medians = df.groupby(group_column)[variable].median()
    group_sizes = df.groupby(group_column).size()

    # Process only significant results
    significant_results = tukey._results_df[tukey._results_df["reject"] == True]
    for _, row in significant_results.iterrows():
        group_a, group_b = row["group1"], row["group2"]
        p_value = row["p-adj"]
        mean_a, mean_b = group_means[group_a], group_means[group_b]
        median_a, median_b = group_medians[group_a], group_medians[group_b]
        size_a, size_b = group_sizes[group_a], group_sizes[group_b]

        # Order groups so the one with the higher mean is first
        groups_info = sorted(
            [
                {
                    "name": group_a,
                    "sample_size": size_a,
                    "mean": mean_a,
                    "median": median_a,
                },
                {
                    "name": group_b,
                    "sample_size": size_b,
                    "mean": mean_b,
                    "median": median_b,
                },
            ],
            key=lambda x: x["mean"],
            reverse=True,
        )

        result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.2f}) > {groups_info[1]['name']} ({groups_info[1]['mean']:.2f}) (p={p_value:.3f})"
        posthoc_results.add_result(result_pretty_text, p_value, groups_info)

    return posthoc_results


def get_dunn_posthoc_results(df, group_column, variable, p_value_threshold):
    posthoc_results = PosthocResults()
    # Perform Dunn's test
    dunn_results = posthoc_dunn(
        df,
        val_col=variable,
        group_col=group_column,
        p_adjust="holm",
    )
    group_means = df.groupby(group_column)[variable].mean()
    group_medians = df.groupby(group_column)[variable].median()
    group_sizes = df.groupby(group_column).size()

    for group_a, group_b in combinations(dunn_results.index, 2):
        p_value = dunn_results.at[group_a, group_b]
        if p_value < p_value_threshold:
            mean_a, mean_b = group_means[group_a], group_means[group_b]
            median_a, median_b = group_medians[group_a], group_medians[group_b]
            size_a, size_b = group_sizes[group_a], group_sizes[group_b]

            # Order groups so the one with the higher mean is first
            groups_info = sorted(
                [
                    {
                        "name": group_a,
                        "sample_size": size_a,
                        "mean": mean_a,
                        "median": median_a,
                    },
                    {
                        "name": group_b,
                        "sample_size": size_b,
                        "mean": mean_b,
                        "median": median_b,
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

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data to analyze.
    variables_to_compare (List[str]): The column names in df that are to be analyzed for statistical significance.
    p_value_threshold (float): The threshold for determining statistical significance.

    Returns:
    Tuple[float, PosthocResults]: A tuple containing the p-value of the McNemar's test and the posthoc results.
    """
    posthoc_results = PosthocResults()
    # Create a contingency table for each pair of variables
    for i in range(len(variables_to_compare) - 1):
        for j in range(i + 1, len(variables_to_compare)):
            var1 = variables_to_compare[i]
            var2 = variables_to_compare[j]
            contingency_table = pd.crosstab(df[var1], df[var2])
            # Perform McNemar's test
            result = mcnemar(contingency_table, exact=False)
            p_value = result.pvalue
            if p_value < p_value_threshold:
                # Add significant result to posthoc results
                groups_info = sorted(
                    [
                        {
                            "name": var1,
                            "sample_size": df[var1].size,
                            "mean": df[var1].mean(),
                            "median": df[var1].median(),
                        },
                        {
                            "name": var2,
                            "sample_size": df[var2].size,
                            "mean": df[var2].mean(),
                            "median": df[var2].median(),
                        },
                    ],
                    key=lambda x: x["mean"],
                    reverse=True,
                )
                result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.2f}) > {groups_info[1]['name']} ({groups_info[1]['mean']:.2f}) (p={p_value:.3f})"

                posthoc_results.add_result(result_pretty_text, p_value, groups_info)
    return posthoc_results


def get_repeated_measures_anova_significance(
    df: pd.DataFrame, variables_to_compare: List[str]
) -> float:
    df_long = pd.melt(
        df.reset_index(),
        id_vars=["index"],
        value_vars=variables_to_compare,
        var_name="conditions",
        value_name="value",
    )
    df_long.rename(columns={"index": "subject_id"}, inplace=True)
    aovrm = AnovaRM(df_long, "value", "subject_id", within=["conditions"])
    anova_results = aovrm.fit()
    p_value = anova_results.anova_table["Pr > F"]["conditions"]
    return p_value


def get_repeated_measures_anova_posthoc(
    df: pd.DataFrame, variables_to_compare: List[str], p_value_threshold: float
) -> PosthocResults:
    posthoc_results = PosthocResults()
    # Perform pairwise comparisons
    pairwise_comparisons = []
    for i, var1 in enumerate(variables_to_compare):
        for j, var2 in enumerate(variables_to_compare):
            if i < j:
                data1 = df[var1]
                data2 = df[var2]
                # Perform paired t-test
                t_stat, p_val = stats.ttest_rel(data1, data2)
                pairwise_comparisons.append((var1, var2, t_stat, p_val))

    # Adjust for multiple comparisons using Bonferroni correction
    num_comparisons = len(pairwise_comparisons)
    reject, pvals_corrected, _, _ = multipletests(
        [x[3] for x in pairwise_comparisons],
        alpha=p_value_threshold,
        method="bonferroni",
    )

    # Add significant results to posthoc results
    for (var1, var2, t_stat, p_val), reject, p_val_corrected in zip(
        pairwise_comparisons, reject, pvals_corrected
    ):
        if reject:
            groups_info = sorted(
                [
                    {
                        "name": var1,
                        "sample_size": df[var1].size,
                        "mean": df[var1].mean(),
                        "median": df[var1].median(),
                    },
                    {
                        "name": var2,
                        "sample_size": df[var2].size,
                        "mean": df[var2].mean(),
                        "median": df[var2].median(),
                    },
                ],
                key=lambda x: x["mean"],
                reverse=True,
            )
            result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.2f}) > {groups_info[1]['name']} ({groups_info[1]['mean']:.2f}) (p={p_val:.3f})"
            posthoc_results.add_result(result_pretty_text, p_val_corrected, groups_info)
    return posthoc_results


def get_wilcoxon_results(
    df: pd.DataFrame, variables_to_compare: List[str], p_value_threshold: float
) -> Tuple[float, PosthocResults]:
    """
    Performs Wilcoxon signed-rank test for two paired samples and returns the p-value and posthoc results.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data to analyze.
    variables_to_compare (List[str]): The column names in df that are to be analyzed for statistical significance.
    p_value_threshold (float): The threshold for determining statistical significance.

    Returns:
    Tuple[float, PosthocResults]: A tuple containing the p-value of the Wilcoxon signed-rank test and the posthoc results.
    """
    posthoc_results = PosthocResults()
    var1, var2 = variables_to_compare

    # Ensure that exactly two variables are provided for comparison
    assert len(variables_to_compare) == 2, "Exactly two variables must be compared."

    # Extract the data for the two variables
    data1 = df[var1]
    data2 = df[var2]

    # Perform the Wilcoxon signed-rank test
    _, p_value = wilcoxon(data1, data2)

    groups_info = sorted(
        [
            {
                "name": var1,
                "sample_size": df[var1].size,
                "mean": df[var1].mean(),
                "median": df[var1].median(),
            },
            {
                "name": var2,
                "sample_size": df[var2].size,
                "mean": df[var2].mean(),
                "median": df[var2].median(),
            },
        ],
        key=lambda x: x["mean"],
        reverse=True,
    )
    # Add significant result to posthoc results
    result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.2f}) > {groups_info[1]['name']} ({groups_info[1]['mean']:.2f}) (p={p_value:.3f})"

    posthoc_results.add_result(
        result_pretty_text, p_value, groups_info
    )

    return p_value, posthoc_results


def get_friedman_significance(
    df: pd.DataFrame, variables_to_compare: List[str]
) -> float:
    """
    Performs Friedman test for repeated measures on ranks and returns the p-value.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data to analyze.
    variables_to_compare (List[str]): The column names in df that are to be analyzed for statistical significance.

    Returns:
    float: The p-value of the Friedman test.
    """
    # Prepare the data for the Friedman test
    data = [df[variable].values for variable in variables_to_compare]
    # Perform the Friedman test
    _, p_value = friedmanchisquare(*data)
    return p_value


def get_nemenyi_results(
    df: pd.DataFrame, variables_to_compare: List[str], p_value_threshold: float
) -> PosthocResults:
    """
    Performs Nemenyi posthoc test suitable after Friedman test and returns the posthoc results.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data to analyze.
    variables_to_compare (List[str]): The column names in df that are to be analyzed for statistical significance.
    p_value_threshold (float): The threshold for determining statistical significance.

    Returns:
    PosthocResults: The posthoc results after performing the Nemenyi test.
    """
    posthoc_results = PosthocResults()
    # Perform Nemenyi posthoc test which is suitable after Friedman test
    posthoc = posthoc_nemenyi_friedman(df[variables_to_compare])
    # Iterate over each pair of variables and check if their comparison is significant
    for i, var1 in enumerate(variables_to_compare):
        for j, var2 in enumerate(variables_to_compare):
            if i < j:
                posthoc_p_value = posthoc.iloc[i, j]
                if posthoc_p_value < p_value_threshold:
                    # Order groups so the one with the higher mean is first
                    groups_info = sorted(
                        [
                            {
                                "name": var1,
                                "sample_size": df[var1].size,
                                "mean": df[var1].mean(),
                                "median": df[var1].median(),
                            },
                            {
                                "name": var2,
                                "sample_size": df[var2].size,
                                "mean": df[var2].mean(),
                                "median": df[var2].median(),
                            },
                        ],
                        key=lambda x: x["mean"],
                        reverse=True,
                    )
                    # Add significant result to posthoc results
                    result_pretty_text = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.2f}) > {groups_info[1]['name']} ({groups_info[1]['mean']:.2f}) (p={posthoc_p_value:.3f})"

                    posthoc_results.add_result(
                        result_pretty_text, posthoc_p_value, groups_info
                    )
    return posthoc_results
