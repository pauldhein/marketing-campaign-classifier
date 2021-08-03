import argparse

from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score

import utils


def main(args):
    train_file = args.dataset_path + "_train.csv"
    dev_file = args.dataset_path + "_dev.csv"
    test_file = args.dataset_path + "_test.csv"

    print("Loading train/dev/test data-frames")
    train_df = utils.load_dataset(train_file)
    dev_df = utils.load_dataset(dev_file)
    test_df = utils.load_dataset(test_file)

    # TODO: try using variance feature selection after creating
    # one-hot encoded data
    # variance_selector = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))

    train_X, train_y = utils.get_X_and_y(train_df)
    dev_X, dev_y = utils.get_X_and_y(dev_df)
    test_X, test_y = utils.get_X_and_y(test_df)

    print("\nFinished loading and prepping train/dev/test data...")

    # NOTE: Using sensible defaults for now
    # NOTE: Using balanced class weighting to correct for the major class imbalance
    classifier = MLPClassifier(
        hidden_layer_sizes=(50, 20),
        max_iter=500,
        random_state=17,
    )
    # classifier = Pipeline(
    #     [
    #         (
    #             "feature_selection",
    #             SelectFromModel(LinearSVC(penalty="l1", dual=False)),
    #         ),
    #         (
    #             "classification",
    #             RandomForestClassifier(
    #                 max_depth=None,
    #                 n_estimators=50,
    #                 min_samples_split=2,
    #                 max_features="sqrt",
    #                 class_weight="balanced",
    #             ),
    #         ),
    #     ]
    # )

    # NOTE: Current best RandomForest Classifier settings
    # classifier = RandomForestClassifier(
    #     max_depth=None,
    #     n_estimators=50,
    #     min_samples_split=2,
    #     max_features="sqrt",
    #     class_weight="balanced",
    # )

    # NOTE: Current GradientBoostingClassifier settings
    # classifier = GradientBoostingClassifier(
    #     n_estimators=100, learning_rate=1.0, max_depth=1, random_state=17
    # )

    print(f"\nBeginning classifier fit for:\n{str(classifier)}\n")
    classifier.fit(train_X, train_y)
    dev_preds = classifier.predict(dev_X)

    p = precision_score(dev_y, dev_preds)
    r = recall_score(dev_y, dev_preds)
    f = f1_score(dev_y, dev_preds)
    print(f"PRECISION={p:.4f}, RECALL={r:.4f}, F1={f:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the bank telemarketing classifier on a given dataset"
    )
    parser.add_argument(
        "-d",
        "--dataset_path",
        default="../data/bank-partial",
        help="base path to the dataset to use",
    )
    args = parser.parse_args()
    main(args)
