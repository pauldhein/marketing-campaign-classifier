from typing import List

from sklearn.preprocessing import LabelEncoder
import pandas as pd


def load_dataset(dataset_path: str) -> "DataFrame":
    return pd.read_csv(dataset_path, sep=";", quotechar='"')


def make_categorical_encoder(cat_examples: List[str]) -> LabelEncoder:
    """Creates a label encoder for a specific categorical feature.

    Args:
        cat_examples (List[str]): A list of all the examples for the categorical feature

    Returns:
        LabelEncoder: A Scikit-Learn label encoder that can translate between string labels and integer feature values
    """
    encoder = preprocessing.LabelEncoder()
    encoder.fit(cat_examples)
    return encoder
