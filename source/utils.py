from typing import List, Tuple

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


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


def compute_and_show_scores(y_t: List[int], y_p: List[int]) -> None:
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
    return (a, p, r, f1, auc)


def train_and_save_classifier(
    classifier, params, name, train_X, train_y, dev_X, dev_y
):
    # Adding feature scaling into the classifier pipeline
    classifier = GridSearchCV(
        Pipeline(
            steps=[
                ("scaling", StandardScaler()),
                ("classifier", classifier),
            ]
        ),
        params,
        cv=10,
        verbose=2,
        n_jobs=4,
        scoring="balanced_accuracy",
    )

    print(f"\nBeginning classifier fit for:\n{str(classifier)}\n")
    classifier.fit(train_X, train_y)
    dev_preds = classifier.predict(dev_X)

    print(f"Scores for {name} classifier")
    utils.compute_and_show_scores(dev_y, dev_preds)

    pickle.dump(
        classifier, open(f"../data/trained_{name}_classifier.pkl", "wb")
    )
