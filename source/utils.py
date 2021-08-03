from typing import List, Tuple

import pandas as pd


def load_dataset(dataset_path: str) -> "DataFrame":
    """Load a dataset into a pandas dataframe. Transform all categorical values into codes

    Args:
        dataset_path (str): A path to a dataset csv file

    Returns:
        DataFrame: a pandas dataframe with all categorical values transformed into numerical encodings.
    """

    return pd.read_csv(
        dataset_path,
        sep=";",
        quotechar='"',
    )


def get_X_and_y(df: "DataFrame") -> Tuple["DataFrame", "DataFrame"]:
    """Separate a pandas dataframe into X features and y labels.

    Args:
        df (DataFrame): A pandas dataframe containing both features and labels

    Returns:
        Tuple[DataFrame, DataFrame]: one DataFrame for the features and another for the single column of labels
    """
    return df.loc[:, df.columns != "y"], df["y"]
