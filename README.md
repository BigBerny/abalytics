# ABalytics: Advanced A/B Testing Statistical Analytics

ABalytics is a Python package designed for statistical analysis, particularly for assessing the significance of A/B testing results. Its goal is to provide high-quality analysis by selecting the appropriate statistical tests based on the type of variable being analyzed. It offers a suite of tools to perform various significance tests and posthoc analyses on experimental data.

## Features

- **Boolean and Numeric Analysis**: Supports analysis of both boolean and numeric data types, ensuring the use of correct statistical methods for each.
- **Significance Tests**: Includes a variety of significance tests such as Chi-Square, Welch's ANOVA, and Kruskal-Wallis, to accurately determine the significance of results.
- **Posthoc Analysis**: Offers posthoc analysis methods like Tukey's HSD, Dunn's test, and Games-Howell, for detailed examination following significance tests.
- **Normality and Homogeneity Checks**: Performs checks for Gaussian distribution and homogeneity of variances using Levene's test, which are critical for selecting the right tests.
- **Pretty Text Output**: Generates a formatted text output with the results of the statistical tests, facilitating interpretation and reporting.

## Installation

To install ABalytics, use pip:
```bash
pip install abalytics
```

## Usage

To use ABalytics, import it:
```python
import abalytics
```

### Analyzing Results

To analyze your A/B testing results, you can use the `get_results` function. This function takes a pandas DataFrame, the name of the column containing the variable to analyze, the name of the column containing the grouping variable, and an optional p-value threshold (default is 0.05).

Here's an example of how to use `get_results`:
```python
from abalytics.analysis import get_results
import pandas as pd

#Load your data into a pandas DataFrame
df = pd.read_csv('your_data.csv')

#Analyze the results
analysis_results = get_results(
    df,
    variable_to_analyze = "order_value",
    group_column = "ab_test_group",
    p_value_threshold=0.05,
    min_sample_size=25,
)
```
The `get_results` function will return an `AnalysisResults` object containing the following attributes:
- `significant_results`: A list of results of the statistical significance tests.
- `info`: A string containing information about the data if no results were found.
- `sample_size`: The sample size of the data.
- `dichotomous_flag`: A boolean flag indicating if the data is dichotomous (e.g. boolean).
- `levene_flag`: A boolean flag indicating if Levene's test for homogeneity of variances is significant.
- `gaussian_flag`: A boolean flag indicating if the data has a Gaussian distribution.

### Generating Pretty Text Output

For a formatted text output of your results, you can use the `get_results_pretty_text` function. This function takes the same parameters as `get_results` and additionally allows for identifiers to be included in the output and whether to include a header.

Here's an example of how to use `get_results_pretty_text`:
```python
from abalytics.analysis import get_results_pretty_text

# Assuming 'df' is your pandas DataFrame
pretty_text = get_results_pretty_text(
    df,
    variable_to_analyze = "order_value",
    group_column = "ab_test_group",
    p_value_threshold=0.05,
    min_sample_size=25,
    header=True,
    identifiers=["A/B Test 1", "Mobile"],
)

print(pretty_text)
```
Executing this code will output a neatly formatted table displaying the outcomes of the statistical significance tests. The table includes the sample size, indicators for Levene's test and Gaussian distribution, and the test results.

A further example of how to use ABalytics can be found in `examples/example.py`.

## Contributing

Contributions to ABalytics are welcome. If you have suggestions for improvements or find any issues, please open an issue or submit a pull request.