import pandas as pd
from scipy import stats
from scipy.stats import normaltest, chi2_contingency
from scikit_posthocs import posthoc_dunn
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest
from pingouin import pairwise_gameshowell, welch_anova
from itertools import combinations


class PosthocResult:
    def __init__(self, result_pretty, p_value, groups):
        self.result_pretty = result_pretty
        self.p_value = p_value
        self.groups = groups


class GroupResult:
    def __init__(self, name, sample_size, mean, median=None):
        self.name = name
        self.sample_size = sample_size
        self.mean = mean
        self.median = median


class PosthocResults:
    def __init__(self):
        self.significant_results = []

    def add_result(self, result_pretty, p_value, groups_info):
        groups = [GroupResult(**group) for group in groups_info]
        result = PosthocResult(result_pretty, p_value, groups)
        self.significant_results.append(result)


def is_boolean(df, variable):
    # Check if the variable is of boolean type
    return df[variable].dtype == bool


def is_numeric(df, variable):
    # Check if the variable is of numeric type
    return df[variable].dtype in [float, int]


def is_gaussian(df, group_column, variable, p_value_threshold):
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


def get_chi_square_significance(df, group_column, variable):
    contingency_table = pd.crosstab(df[group_column], df[variable])
    _, pvalue, _, _ = chi2_contingency(contingency_table)
    return pvalue


def get_chi_square_posthoc_results(df, group_column, variable, p_value_threshold):
    posthoc_results = PosthocResults()
    contingency_table = pd.crosstab(df[group_column], df[variable])
    group_combinations = [
        (group_a, group_b)
        for i, group_a in enumerate(contingency_table.index)
        for group_b in contingency_table.index[i + 1 :]
    ]
    p_values = []

    for group_a, group_b in group_combinations:
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
                {'name': group_a, 'sample_size': nobs[0], 'mean': rate_a},
                {'name': group_b, 'sample_size': nobs[1], 'mean': rate_b}
            ],
            key=lambda x: x['mean'],
            reverse=True
        )

        if p_value < p_value_threshold:
            result_pretty = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.1f}%) vs {groups_info[1]['name']} ({groups_info[1]['mean']:.1f}%) (p={p_value:.3f})"
            posthoc_results.add_result(result_pretty, p_value, groups_info)

    # Apply Bonferroni correction
    corrected_alpha = p_value_threshold / len(p_values)
    posthoc_results.significant_results = [
        result for result in posthoc_results.significant_results if result.p_value < corrected_alpha
    ]

    return posthoc_results


def is_levene_significant(df, group_column, variable, p_value_threshold):
    # Prepare data for homogeneity of variances test
    groups = df[group_column].unique()
    data_groups = [df[df[group_column] == group][variable] for group in groups]

    # Perform Levene's test for equal variances
    _, p_levene = stats.levene(*data_groups)

    # Check if variances are equal
    return p_levene < p_value_threshold


def get_welch_anova_significance(df, group_column, variable):
    # Perform Welch's ANOVA
    welch_anova_results = welch_anova(data=df, dv=variable, between=group_column)
    # Check if the p-value from Welch's ANOVA is significant
    return welch_anova_results.loc[0, "p-unc"]


def get_games_howell_posthoc_results(df, group_column, variable, p_value_threshold):
    posthoc_results = PosthocResults()
    games_howell_results = pairwise_gameshowell(data=df, dv=variable, between=group_column)
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
                    {'name': group_a, 'sample_size': size_a, 'mean': mean_a, 'median': median_a},
                    {'name': group_b, 'sample_size': size_b, 'mean': mean_b, 'median': median_b}
                ],
                key=lambda x: x['mean'],
                reverse=True
            )

            result_pretty = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.2f}) vs {groups_info[1]['name']} ({groups_info[1]['mean']:.2f}) (p={p_value:.3f})"
            posthoc_results.add_result(result_pretty, p_value, groups_info)

    return posthoc_results


def is_gaussian(df, group_column, variable, p_value_threshold):
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


def get_anova_results(df, group_column, variable):
    # Perform one-way ANOVA
    anova_results = AnovaRM(
        data=df, depvar=variable, subject="session_id", within=[group_column]
    ).fit()
    return anova_results


def get_oneway_anova_significance(df, group_column, variable):
    group_names = df[group_column].unique()
    # Perform one-way ANOVA
    group_data = [df[df[group_column] == group][variable] for group in group_names]
    _, pvalue = stats.f_oneway(*group_data)
    return pvalue


def get_crushal_wallis_significance(df, group_column, variable):
    group_names = df[group_column].unique()
    # Perform Kruskal-Wallis test
    group_data = [df[df[group_column] == group][variable] for group in group_names]
    _, pvalue = stats.kruskal(*group_data)
    return pvalue


def get_tukeyhsd_posthoc_results(df, group_column, variable, p_value_threshold):
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
                {'name': group_a, 'sample_size': size_a, 'mean': mean_a, 'median': median_a},
                {'name': group_b, 'sample_size': size_b, 'mean': mean_b, 'median': median_b}
            ],
            key=lambda x: x['mean'],
            reverse=True
        )

        result_pretty = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.2f}) vs {groups_info[1]['name']} ({groups_info[1]['mean']:.2f}) (p={p_value:.3f})"
        posthoc_results.add_result(result_pretty, p_value, groups_info)

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
                    {'name': group_a, 'sample_size': size_a, 'mean': mean_a, 'median': median_a},
                    {'name': group_b, 'sample_size': size_b, 'mean': mean_b, 'median': median_b}
                ],
                key=lambda x: x['mean'],
                reverse=True
            )
                
            result_pretty = f"{groups_info[0]['name']} ({groups_info[0]['mean']:.2f}) vs {groups_info[1]['name']} ({groups_info[1]['mean']:.2f}) (p={p_value:.3f})"
            posthoc_results.add_result(result_pretty, p_value, groups_info)
    return posthoc_results
