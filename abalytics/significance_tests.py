import pandas as pd
from scipy import stats
from scipy.stats import normaltest, chi2_contingency
from scikit_posthocs import posthoc_dunn
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest
from pingouin import pairwise_gameshowell, welch_anova

p_value_threshold = 0.05  # Set the p-value threshold to 0.05

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

def get_chi_square_posthoc_results(df, group_column, variable):
    results = []
    contingency_table = pd.crosstab(df[group_column], df[variable])
    group_combinations = [(group_a, group_b) for i, group_a in enumerate(contingency_table.index) for group_b in contingency_table.index[i + 1:]]
    p_values = []
        
    for group_a, group_b in group_combinations:
        count = [contingency_table.loc[group_a, 1], contingency_table.loc[group_b, 1]]
        nobs = [contingency_table.loc[group_a].sum(), contingency_table.loc[group_b].sum()]
        rate_a = (count[0] / nobs[0]) * 100
        rate_b = (count[1] / nobs[1]) * 100
            
        _, p_value = proportions_ztest(count, nobs)
        p_values.append(p_value)
            
        if p_value < p_value_threshold:
            if rate_a > rate_b:
                results.append(f"{group_a} ({rate_a:.1f}%) > {group_b} ({rate_b:.1f}%) (p={p_value:.3f})")
            else:
                results.append(f"{group_b} ({rate_b:.1f}%) > {group_a} ({rate_a:.1f}%) (p={p_value:.3f})")
        
    # Apply Bonferroni correction
    corrected_alpha = p_value_threshold / len(p_values)
    results_corrected = [result for result, p_value in zip(results, p_values) if p_value < corrected_alpha]

    return results_corrected if results_corrected else results

def is_levene_significant(df, group_column, variable):
    # Prepare data for homogeneity of variances test
    groups = df[group_column].unique()
    data_groups = [
        df[df[group_column] == group][variable] for group in groups
    ]
            
    # Perform Levene's test for equal variances
    _, p_levene = stats.levene(*data_groups)
            
    # Check if variances are equal
    return p_levene < p_value_threshold

def get_welch_anova_significance(df, group_column, variable):
    # Perform Welch's ANOVA
    welch_anova_results = welch_anova(
        data=df,
        dv=variable,
        between=group_column
    )
    # Check if the p-value from Welch's ANOVA is significant
    return welch_anova_results.loc[0, "p-unc"]

def get_games_howell_posthoc_results(df, group_column, variable):
    results = []
    posthoc_results = pairwise_gameshowell(
        data=df,
        dv=variable,
        between=group_column
    )
    # Iterate through the posthoc results and append significant comparisons
    for _, row in posthoc_results.iterrows():
        if row['pval'] < p_value_threshold:
            group_a = row['A']
            group_b = row['B']
            mean_diff = row['mean(A)'] - row['mean(B)']
            p_value = row['pval']
            mean_a = df[df[group_column] == group_a][variable].mean()
            mean_b = df[df[group_column] == group_b][variable].mean()
            if mean_diff > 0:
                results.append(
                    f"{group_a} ({mean_a:.2f}) > {group_b} ({mean_b:.2f})  (p={p_value:.3f})"
                )
            else:
                results.append(
                    f"{group_b} ({mean_b:.2f}) > {group_a} ({mean_a:.2f})  (p={p_value:.3f})"
                )
    return results


def is_gaussian(df, group_column, variable):
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
        data=df,
        depvar=variable,
        subject="session_id",
        within=[group_column]
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

def get_tukeyhsd_posthoc_results(df, group_column, variable):
    # Perform Tukey's HSD test
    tukey = pairwise_tukeyhsd(
        endog=df[variable],
        groups=df[group_column],
        alpha=p_value_threshold,
    )

    # Calculate means for each group
    group_means = df.groupby(group_column)[
        variable
    ].mean()
    results = []
    # Print only significant results
    significant_results = tukey._results_df[
        tukey._results_df["reject"] == True
    ]
    for _, row in significant_results.iterrows():
        group_a = row["group1"]
        group_b = row["group2"]
        mean_diff = row["meandiff"]
        p_value = row["p-adj"]
        mean_a = group_means[group_a]
        mean_b = group_means[group_b]

        if mean_diff > 0:
            results.append(
                f"{group_a} ({mean_a:.2f}) > {group_b} ({mean_b:.2f})  (p={p_value:.3f})"
            )
        else:
            results.append(
                f"{group_b} ({mean_b:.2f}) > {group_a} ({mean_a:.2f})  (p={p_value:.3f})"
            )
    return results
    
def get_dunn_posthoc_results(df, group_column, variable):
    results = []
    # Perform Dunn's test
    dunn_results = posthoc_dunn(
        df,
        val_col=variable,
        group_col=group_column,
        p_adjust="holm",
    )
    group_means = df.groupby(group_column)[
        variable
    ].mean()
    for i, group_a in enumerate(dunn_results.index):
        for group_b in dunn_results.columns[i + 1 :]:
            p_value = dunn_results.loc[group_a, group_b]
            # Check if the p-value is less than 0.05
            if p_value < p_value_threshold:
                # Compare means of the two groups
                if group_means[group_a] > group_means[group_b]:
                    results.append(
                        f"{group_a} ({group_means[group_a]:.2f}) > {group_b} ({group_means[group_b]:.2f})  (p={p_value:.3f})"
                    )
                else:
                    results.append(
                        f"{group_b} ({group_means[group_b]:.2f}) > {group_a} ({group_means[group_a]:.2f})  (p={p_value:.3f})"
                    )
    return results