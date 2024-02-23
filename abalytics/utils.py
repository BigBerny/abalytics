from .significance_tests import PosthocResults
from typing import Optional, List
import pandas as pd


def get_table_header(max_identifier_length) -> str:
    """
    Returns a text with the results of the statistical significance tests.

    Parameters:
    max_identifier_length (int): The length of the longest identifier string.

    Returns:
    str: A formatted text string with the header of the pretty text table.
    """
    if max_identifier_length > 0:
        return f"{'Identifier':<{max_identifier_length}}{'n':>10}    {'Levene':<6}    {'Gaussian':<8}    {'Result'}"
    return f"{'n':>10}    {'Levene':<6}    {'Gaussian':<8}    {'Result'}"


def format_results_as_table(
    abalytics_results: List[PosthocResults],
    header: bool = True,
    identifiers_list: Optional[List[List[str]]] = None,
    show_only_significant_results: bool = False,
):
    """
    Returns a text with the results of the statistical significance tests.

    Parameters:
    abalytics_results (list of PosthocResults): A list of PosthocResults objects.
    header (bool, optional): A boolean flag indicating if the header of the pretty text table should be included. Defaults to True.
    identifiers (list of str, optional): A list of strings that are to be used for identifying the data. Defaults to [].
    show_only_significant_results (bool, optional): A boolean flag indicating if only significant results should be shown. Defaults to False.

    Returns:
    str: A formatted text string with the results of the statistical significance tests.
    """
    if identifiers_list:
        assert len(identifiers_list) == len(
            abalytics_results
        ), "The number of identifiers must match the number of results."

        identifier_string_list = []
        for identifiers in identifiers_list:
            identifier_string = ""
            if identifiers:
                identifier_string = "    ".join(identifiers)
                identifier_string += "    "
            identifier_string_list.append(identifier_string)
    else:
        identifier_string_list = [""] * len(abalytics_results)

    # Pad the identifier strings to the same length
    max_identifier_length = max(
        [len(identifier_string) for identifier_string in identifier_string_list]
    )
    identifier_string_list = [
        s.ljust(max_identifier_length) for s in identifier_string_list
    ]

    output_string = ""

    if header:
        output_string += get_table_header(max_identifier_length)
        output_string += "\n"
    output_results = []

    for idx, abalytics_result in enumerate(abalytics_results):
        identifier_string = (
            identifier_string_list[idx] if identifier_string_list else ""
        )
        significant_results = abalytics_result.significant_results
        info = abalytics_result.info
        sample_size = abalytics_result.sample_size
        levene_flag = abalytics_result.levene_flag
        gaussian_flag = abalytics_result.gaussian_flag

        levene_output = "X" if levene_flag else ""
        gaussian_output = "X" if gaussian_flag else ""

        if show_only_significant_results and len(significant_results) == 0:
            continue
        if len(significant_results) > 0:
            for i, result in enumerate(significant_results):
                if i == 0:
                    output_results.append(
                        f"{identifier_string}{sample_size:>10}    {levene_output:<6}    {gaussian_output:<8}    {result.result_pretty_text}"
                    )
                else:
                    output_results.append(
                        f"{len(identifier_string) * ' '}{sample_size:>10}    {levene_output:<6}    {gaussian_output:<8}    {result.result_pretty_text}"
                    )
        else:
            output_results.append(
                f"{identifier_string}{sample_size:>10}    {levene_output:<6}    {gaussian_output:<8}    {info}"
            )
    output_string += "\n".join(output_results)
    return output_string


def convert_long_to_wide(
    df: pd.DataFrame, index_col: str, columns_col: str, flatten_columns: bool = True
) -> pd.DataFrame:
    """
    Converts a DataFrame from long format to wide format, turning all columns into 'column_group' format.
    Optionally flattens multi-level columns into single-level.

    Parameters:
    df (pd.DataFrame): The DataFrame in long format.
    index_col (str): The name of the column to use as the identifier (index) in the wide format.
    columns_col (str): The name of the column that contains the group names in the long format.
    flatten_columns (bool, optional): Whether to flatten multi-level columns into single-level. Defaults to True.

    Returns:
    pd.DataFrame: The DataFrame converted to wide format, with optional column flattening.
    """
    # Create a pivot table with multi-level columns
    wide_df = df.pivot_table(index=index_col, columns=columns_col, aggfunc='first')

    if flatten_columns:
        # Flatten the multi-level columns and create combined column names
        wide_df.columns = ["{}_{}".format(col[0], col[1]) for col in wide_df.columns.values]

    return wide_df.reset_index()
