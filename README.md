# ABalytics: Advanced A/B Testing Statistical Analytics

ABalytics is a Python package designed for statistical analysis, particularly for assessing the significance of A/B testing results. Its goal is to provide high-quality analysis by selecting the appropriate statistical tests based on the type of variable being analyzed. It offers a suite of tools to perform various significance tests and posthoc analyses on experimental data.

## Features

- **Boolean and Numeric Analysis**: Supports analysis of both boolean and numeric data types, ensuring the use of correct statistical methods for each.
- **Significance Tests**: Includes a variety of significance tests such as Chi-Square, Welch's ANOVA, and Kruskal-Wallis, to accurately determine the significance of results.
- **Posthoc Analysis**: Offers posthoc analysis methods like Tukey's HSD, Dunn's test, and Games-Howell, for detailed examination following significance tests.
- **Normality and Homogeneity Checks**: Performs checks for Gaussian distribution and homogeneity of variances using Levene's test, which are critical for selecting the right tests.
- **Data Transformation**: Provides functionality to convert data from long to wide format, facilitating analysis of dependent groups.
- **Pretty Text Output**: Generates a formatted text output with the results of the statistical tests, facilitating interpretation and reporting.

## Installation

To install ABalytics, use pip:
```bash
pip install abalytics
```

## Usage
### Analyzing Results

ABalytics provides two main functions for analyzing A/B testing results: `analyze_independent_groups` and `analyze_dependent_groups`.

#### Independent Groups Analysis

`analyze_independent_groups` is used for analyzing data where the groups are independent of each other. It takes a pandas DataFrame, the name of the column containing the variable to analyze, the name of the column containing the grouping variable, and optional parameters for p-value threshold and minimum sample size.

Example:
```python
from abalytics import analyze_independent_groups
import pandas as pd

# Load your data into a pandas DataFrame
df = pd.read_csv('your_data.csv')

# Analyze the results
analysis_results = analyze_independent_groups(
    df,
    variable_to_analyze="order_value",
    group_column="ab_test_group",
)
```


#### Dependent Groups Analysis

`analyze_dependent_groups` is used for analyzing data where the groups are dependent, such as repeated measures on the same subjects. It requires data in wide format. If your data is in long format, you can use the `convert_long_to_wide` function in `abalytics.utils` to convert it. The `analyze_dependent_groups` function takes a pandas DataFrame, the names of the columns to compare, and optional parameters for p-value threshold and minimum sample size.

Example:
```python
from abalytics import analyze_dependent_groups
import pandas as pd

# Load your data into a pandas DataFrame
df = pd.read_csv('your_data.csv')

# Analyze the results
analysis_results = analyze_dependent_groups(
    df,
    variables_to_compare=["pre_test_score", "post_test_score"],
)
```

### Data Transformation

The `convert_long_to_wide` function in `abalytics.utils` is designed to transform data from long format to wide format, with an option to keep multi-level columns or flatten them. `analyze_dependent_groups` requires data in wide format to operate correctly.

Example:
```python
from abalytics.utils import convert_long_to_wide
import pandas as pd

# Assuming 'df_long' is your pandas DataFrame in long format
df_wide = convert_long_to_wide(
    df_long,
    index_col="subject_id",
    columns_col="condition",
    flatten_columns=True # Set to False if you wish to keep multi-level columns
)
```

### Generating Pretty Text Output

To get a formatted text output of your results, you can use the `utils.format_results_as_table` function.

Example:
```python
from abalytics.utils import format_results_as_table
from abalytics import analyze_independent_groups
import pandas as pd

# Load your data into a pandas DataFrame
df = pd.read_csv('your_data.csv')

# Analyze the results
analysis_results = analyze_independent_groups(
    df,
    variable_to_analyze="order_value",
    group_column="ab_test_group",
)

# Generate pretty text output
pretty_text = format_results_as_table(
    abalytics_results=[analysis_results],
    identifiers_list=[["A/B Test 1", "Mobile"]],
)
print(pretty_text)
```
Executing this code will output a neatly formatted table displaying the outcomes of the statistical significance tests. The table includes the sample size, indicators for Levene's test and Gaussian distribution, and the test results.

A further example of how to use ABalytics can be found in `examples/example.py`.

## Contributing

Contributions to ABalytics are welcome. If you have suggestions for improvements or find any issues, please open an issue or submit a pull request.