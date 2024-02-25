from .significance_tests import PosthocResults
from typing import Optional, List
import pandas as pd
from tabulate import tabulate


def get_table_header(max_identifier_length: int) -> str:
    """
    Returns a text with the results of the statistical significance tests.

    Parameters:
    max_identifier_length (int): The length of the longest identifier string.

    Returns:
    str: A formatted text string with the header of the pretty text table.
    """
    if max_identifier_length > 0:
        return f"{'Identifier':<{max_identifier_length}}{'n':>10}    {'Result'}"
    return f"{'n':>10}    {'Result'}"


def format_results_as_table(
    abalytics_results: List[PosthocResults],
    header: bool = True,
    identifiers_list: Optional[List[dict]] = None,
    show_only_significant_results: bool = False,
    show_details: bool = False,
    output_format: str = "simple",
) -> str:
    """
    Returns a text with the results of the statistical significance tests.

    Parameters:
    abalytics_results (list of PosthocResults): A list of PosthocResults objects.
    header (bool, optional): A boolean flag indicating if the header of the pretty text table should be included. Defaults to True.
    identifiers (list of str, optional): A list of strings that are to be used for identifying the data. Defaults to [].
    show_only_significant_results (bool, optional): A boolean flag indicating if only significant results should be shown. Defaults to False.
    show_details (bool, optional): A boolean flag indicating if the details of the statistical tests should be shown. Defaults to False.
    output_format (str, optional): A string indicating the format of the output. Defaults to "simple". Checl the tabulate documentation: https://github.com/astanin/python-tabulate#table-format

    Returns:
    str: A formatted text string with the results of the statistical significance tests.
    """
    if identifiers_list:
        assert len(identifiers_list) == len(
            abalytics_results
        ), "The number of identifiers must match the number of results."
    else:
        identifiers_list = [{} for _ in range(len(abalytics_results))]

    output_results = []

    for idx, abalytics_result in enumerate(abalytics_results):
        significant_results = abalytics_result.significant_results
        info = abalytics_result.info
        a_priori_test = abalytics_result.a_priori_test
        sample_size = abalytics_result.sample_size
        levene_flag = abalytics_result.levene_flag
        gaussian_flag = abalytics_result.gaussian_flag

        if show_only_significant_results and len(significant_results) == 0:
            continue
        if len(significant_results) > 0:
            for result in significant_results:
                row = identifiers_list[idx].copy()
                row["n"] = sample_size
                row["Result"] = result.result_pretty_text
                row["p-value"] = result.p_value  # round(result.p_value,3)
                if show_details:
                    row["Analysis method"] = result.analysis_method
                    if info:
                        row["info"] = info
                    row["Gaussian"] = gaussian_flag
                    row["Levene"] = levene_flag
                    row["A priori test"] = a_priori_test
                output_results.append(row)
        elif info:
            row = identifiers_list[idx]
            row["n"] = sample_size
            row["Result"] = None
            row["p-value"] = None
            row["info"] = info
            if show_details and a_priori_test:
                row["A priori test"] = a_priori_test
            output_results.append(row)

    table = tabulate(output_results, headers="keys", floatfmt=".3f", tablefmt=output_format)
    return table
