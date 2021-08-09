from typing import List, Dict, Tuple

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def load_Xy_dataset(dataset_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load a dataset into a pandas dataframe and then separate the dataframe into X features and y labels.

    Args:
        dataset_path (str): A path to a dataset csv file

    Returns:
        Tuple[DataFrame, DataFrame]: one DataFrame for the features and another for the single column of labels
    """
    df = pd.read_csv(
        dataset_path,
        sep=";",
        quotechar='"',
    )
    return df.loc[:, df.columns != "y"], df["y"]


def compute_metrics(y_t: List[int], y_p: List[int]) -> Dict[str, float]:
    """Computes, prints, and returns the precision, recall, F1 scores.

    Args:
        y_t (List[int]): A list of the true label values
        y_p (List[int]): A list of the predicted label values
    """
    a = accuracy_score(y_t, y_p)
    p = precision_score(y_t, y_p)
    r = recall_score(y_t, y_p)
    f1 = f1_score(y_t, y_p)
    auc = roc_auc_score(y_t, y_p)
    print(
        f"ACCURACY={a:.4f}, PRECISION={p:.4f}, RECALL={r:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
    )
    return {
        "accuracy": a,
        "precision": p,
        "recall": r,
        "f1": f1,
        "roc_auc": auc,
    }
