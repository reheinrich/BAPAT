"""
Utility functions for data processing tasks.

This module contains helper functions for extracting recording filenames and reading data files.
"""

import os
from typing import List
import pandas as pd


def extract_recording_filename(path_column: pd.Series) -> pd.Series:
    """
    Extracts the recording filename from a path column.

    This function takes a pandas Series containing file paths and extracts the base filename
    without the extension for each path.

    Args:
        path_column (pd.Series): Series containing file paths.

    Returns:
        pd.Series: Series containing extracted recording filenames.
    """
    return path_column.apply(
        lambda x: os.path.splitext(os.path.basename(x))[0] if isinstance(x, str) else x
    )


def extract_recording_filename_from_filename(filename_series: pd.Series) -> pd.Series:
    """
    Extracts the recording filename from the source file name.

    This function takes a pandas Series containing filenames and extracts the base filename
    without the extension for each.

    Args:
        filename_series (pd.Series): Series containing file names.

    Returns:
        pd.Series: Series containing extracted recording filenames.
    """
    return filename_series.apply(lambda x: x.split(".")[0] if isinstance(x, str) else x)


def read_and_concatenate_files_in_directory(directory_path: str) -> pd.DataFrame:
    """
    Reads all .txt files from a directory and concatenates them into a single DataFrame.

    This function scans the given directory for all .txt files, reads each one into a DataFrame,
    adds a 'source_file' column with the filename, and concatenates all DataFrames into one.

    Args:
        directory_path (str): Path to the directory containing the .txt files.

    Returns:
        pd.DataFrame: A concatenated DataFrame of all files, or an empty DataFrame if none are found.
    """

    df_list: List[pd.DataFrame] = []
    columns_set = None
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory_path, filename)
            try:
                df = pd.read_csv(filepath, sep="\t", encoding="utf-8")
            except UnicodeDecodeError:
                # Try reading with 'latin-1' encoding
                df = pd.read_csv(filepath, sep="\t", encoding="latin-1")
            if columns_set is None:
                columns_set = set(df.columns)
            elif set(df.columns) != columns_set:
                raise ValueError(
                    f"File {filename} has different columns than the previous files."
                )
            df["source_file"] = filename
            df_list.append(df)
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()
