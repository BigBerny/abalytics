# ABalytics

ABalytics is a Python package designed for statistical analysis, particularly for assessing the significance of A/B testing results. Its goal is to provide high-quality analysis by selecting the appropriate statistical tests based on the type of variable being analyzed. It offers a suite of tools to perform various significance tests and posthoc analyses on experimental data.

## Features

- **Boolean and Numeric Analysis**: Supports analysis of both boolean and numeric data types, ensuring the use of correct statistical methods for each.
- **Significance Tests**: Includes a variety of significance tests such as Chi-Square, Welch's ANOVA, and Kruskal-Wallis, to accurately determine the significance of results.
- **Posthoc Analysis**: Offers posthoc analysis methods like Tukey's HSD, Dunn's test, and Games-Howell, for detailed examination following significance tests.
- **Normality and Homogeneity Checks**: Performs checks for Gaussian distribution and homogeneity of variances using Levene's test, which are critical for selecting the right tests.
- **Pretty Text Output**: Generates a formatted text output with the results of the statistical tests, facilitating interpretation and reporting.

## Installation

To install ABalytics, use pip:
```pip install abalytics```

## Usage

To use ABalytics, import it:
```import abalytics```

An example of how to use ABalytics can be found in `examples/example.py`.

## Contributing

Contributions to ABalytics are welcome.